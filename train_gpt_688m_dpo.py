#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) training for 688M GPT model.
Trains on preference pairs to improve response quality.

DPO Loss: -log_sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected))
"""

import os
import sys
import uuid
import json
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------------------------------------------------------
# DPO Hyperparameters

@dataclass
class DPOConfig:
    # Data
    data_dir: str = "data/ultrafeedback"
    sft_checkpoint: str = "logs/f93f0295-8327-49de-9d7c-574d342acf22/sft_step002500_loss1.6624.pt"

    # Training
    batch_size: int = 2  # Small due to 2 models in memory
    num_iterations: int = 2000
    warmup_steps: int = 100
    cooldown_steps: int = 400

    # DPO specific
    beta: float = 0.1  # KL penalty coefficient
    label_smoothing: float = 0.0

    # Learning rates (lower than SFT)
    lr_base: float = 1e-6
    lr_muon: float = 5e-5

    # Validation
    val_every: int = 100
    save_every: int = 250

    # Model
    model_dim: int = 1536
    num_heads: int = 12
    num_layers: int = 12
    head_dim: int = 128
    vocab_size: int = 50257
    max_seq_len: int = 1024

# -----------------------------------------------------------------------------
# Model Components (same as training)

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

def next_multiple_of_n(v, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        d = x.shape[3] // 2
        y1 = x[..., :d] * cos + x[..., d:] * sin
        y2 = x[..., :d] * (-sin) + x[..., d:] * cos
        return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, dim)
        self.c_v = CastedLinear(dim, dim)
        self.c_proj = CastedLinear(dim, dim)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        self.rotary = Rotary(dim // num_heads)
        self.attn_gate = CastedLinear(12, num_heads)

    def forward(self, x: Tensor, v1=None, attn_scale=0.1):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.num_heads, -1)
        k = self.c_k(x).view(B, T, self.num_heads, -1)
        v = self.c_v(x).view(B, T, self.num_heads, -1)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=attn_scale)
        y = y.transpose(1, 2)
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).unsqueeze(-1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, head_dim, num_heads) if layer_idx not in [0, 7] else None
        self.mlp = MLP(dim) if layer_idx != 0 else None
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, v1, x0: Tensor, attn_scale=0.1):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x1, v1 = self.attn(norm(x), v1, attn_scale)
            x = x + x1
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x, v1

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, model_dim),
            h=nn.ModuleList([Block(model_dim, head_dim, num_heads, i) for i in range(num_layers)]),
        ))
        self.lm_head = CastedLinear(model_dim, vocab_size)
        self.smear_gate = CastedLinear(12, 1)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.5 * torch.ones(1))
        self.num_layers = num_layers

    def forward(self, input_seq: Tensor):
        if input_seq.ndim == 1:
            input_seq = input_seq.unsqueeze(0)
        B, T = input_seq.shape

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve_list = [None, ve[1], ve[2]] + [None] * (len(self.transformer.h) - 6) + [ve[0], ve[1], ve[2]]

        x = self.transformer.wte(input_seq)
        smear_gate_out = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)
        x = norm(x)
        x0 = x
        v1 = None

        skip_connections = []
        x_backout = None
        backout_layer = 8

        for i in range(self.num_encoder_layers):
            if ve_list[i] is not None and v1 is None:
                v1 = ve_list[i][None].view(B, T, self.transformer.h[i].attn.num_heads if self.transformer.h[i].attn else 6, -1)
            x, v1 = self.transformer.h[i](x, v1, x0, attn_scale=0.1)
            skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            x = x + self.skip_weights[i] * skip_connections.pop()
            if ve_list[layer_idx] is not None and v1 is None:
                v1 = ve_list[layer_idx][None].view(B, T, 6, -1)
            x, v1 = self.transformer.h[layer_idx](x, v1, x0, attn_scale=0.1)
            if layer_idx == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = norm(x)
        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30)
        return logits

# -----------------------------------------------------------------------------
# DPO Data Loader

class DPODataLoader:
    def __init__(self, data_file, batch_size, device):
        self.batch_size = batch_size
        self.device = device

        # Load all data
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.num_samples = len(self.data)
        self.indices = list(range(self.num_samples))
        random.shuffle(self.indices)
        self.current_idx = 0

    def __len__(self):
        return self.num_samples // self.batch_size

    def get_batch(self):
        batch_data = []

        for _ in range(self.batch_size):
            if self.current_idx >= self.num_samples:
                random.shuffle(self.indices)
                self.current_idx = 0

            idx = self.indices[self.current_idx]
            batch_data.append(self.data[idx])
            self.current_idx += 1

        return batch_data

# -----------------------------------------------------------------------------
# DPO Loss

def compute_log_probs(logits, labels, mask):
    """Compute per-token log probabilities."""
    # logits: (B, T, V), labels: (B, T), mask: (B, T)
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs for actual tokens
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Apply mask and sum
    masked_log_probs = token_log_probs * mask
    return masked_log_probs.sum(dim=-1)  # (B,)

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta, label_smoothing=0.0):
    """
    Compute DPO loss.

    loss = -log_sigmoid(beta * (log_pi_chosen - log_pi_rejected - log_ref_chosen + log_ref_rejected))
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    logits = chosen_rewards - rejected_rewards

    if label_smoothing > 0:
        losses = -F.logsigmoid(logits) * (1 - label_smoothing) - F.logsigmoid(-logits) * label_smoothing
    else:
        losses = -F.logsigmoid(logits)

    # Metrics
    chosen_reward = chosen_rewards.mean().item()
    rejected_reward = rejected_rewards.mean().item()
    accuracy = (logits > 0).float().mean().item()

    return losses.mean(), chosen_reward, rejected_reward, accuracy

# -----------------------------------------------------------------------------
# Optimizer (Muon with memory-efficient Newton-Schulz)

def zeropower_via_newtonschulz5(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        del A, B  # Free memory immediately
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, max_size=4096):
        self.max_size = max_size  # Skip Newton-Schulz for matrices larger than this
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Apply Newton-Schulz only if matrix not too large (memory constraint)
                if g.ndim >= 2 and g.shape[0] <= self.max_size:
                    g = zeropower_via_newtonschulz5(g)

                p.add_(g, alpha=-lr)

# -----------------------------------------------------------------------------
# Main Training

def main():
    config = DPOConfig()

    # Setup
    device = "cuda"
    torch.set_float32_matmul_precision('high')

    run_id = str(uuid.uuid4())
    log_dir = f"logs/{run_id}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"DPO Training Run ID: {run_id}")
    print(f"SFT Checkpoint: {config.sft_checkpoint}")

    # Load SFT checkpoint
    print("Loading SFT checkpoint...")
    checkpoint = torch.load(config.sft_checkpoint, map_location="cpu", weights_only=False)

    # Create policy model (trainable)
    policy_model = GPT(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        model_dim=config.model_dim,
        max_seq_len=config.max_seq_len
    )

    # Create reference model (frozen)
    ref_model = GPT(
        vocab_size=config.vocab_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        model_dim=config.model_dim,
        max_seq_len=config.max_seq_len
    )

    # Load weights
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model'].items()}
    policy_model.load_state_dict(state_dict, strict=False)
    ref_model.load_state_dict(state_dict, strict=False)

    policy_model = policy_model.to(device).train()
    ref_model = ref_model.to(device).eval()

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    # No torch.compile - saves memory and avoids recompilation issues with grad_mode switching
    print(f"Models loaded. Policy trainable, Reference frozen.")

    # Setup optimizers
    muon_params = []
    base_params = []

    for name, param in policy_model.named_parameters():
        if param.ndim >= 2:
            muon_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.AdamW(base_params, lr=config.lr_base, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    optimizer2 = Muon(muon_params, lr=config.lr_muon, momentum=0.95, max_size=4096)

    # Load data
    train_file = f"{config.data_dir}/train.jsonl"
    val_file = f"{config.data_dir}/val.jsonl"

    if not os.path.exists(train_file):
        print(f"ERROR: {train_file} not found. Run prepare_dpo_data.py first.")
        sys.exit(1)

    train_loader = DPODataLoader(train_file, config.batch_size, device)
    val_loader = DPODataLoader(val_file, config.batch_size, device)

    print(f"Train samples: {train_loader.num_samples}")
    print(f"Val samples: {val_loader.num_samples}")

    # Training loop
    print(f"\nStarting DPO training for {config.num_iterations} iterations...")

    for step in range(1, config.num_iterations + 1):
        # Learning rate schedule
        if step < config.warmup_steps:
            lr_mult = step / config.warmup_steps
        elif step > config.num_iterations - config.cooldown_steps:
            lr_mult = (config.num_iterations - step) / config.cooldown_steps
        else:
            lr_mult = 1.0

        for pg in optimizer.param_groups:
            pg['lr'] = config.lr_base * lr_mult
        for pg in optimizer2.param_groups:
            pg['lr'] = config.lr_muon * lr_mult

        # Get batch
        batch = train_loader.get_batch()

        total_loss = 0
        total_chosen_reward = 0
        total_rejected_reward = 0
        total_accuracy = 0

        optimizer.zero_grad()
        optimizer2.zero_grad()

        for sample in batch:
            prompt_tokens = sample["prompt_tokens"]
            chosen_tokens = sample["chosen_tokens"]
            rejected_tokens = sample["rejected_tokens"]

            # Create full sequences
            chosen_seq = prompt_tokens + chosen_tokens
            rejected_seq = prompt_tokens + rejected_tokens

            # Pad to same length
            max_len = max(len(chosen_seq), len(rejected_seq))
            chosen_seq = chosen_seq + [0] * (max_len - len(chosen_seq))
            rejected_seq = rejected_seq + [0] * (max_len - len(rejected_seq))

            # Create tensors
            chosen_input = torch.tensor(chosen_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            chosen_target = torch.tensor(chosen_seq[1:], dtype=torch.long, device=device).unsqueeze(0)
            rejected_input = torch.tensor(rejected_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
            rejected_target = torch.tensor(rejected_seq[1:], dtype=torch.long, device=device).unsqueeze(0)

            # Create masks (only for response tokens)
            prompt_len = len(prompt_tokens)
            chosen_mask = torch.zeros(1, max_len - 1, device=device)
            rejected_mask = torch.zeros(1, max_len - 1, device=device)
            chosen_mask[0, prompt_len-1:prompt_len-1+len(sample["chosen_tokens"])] = 1
            rejected_mask[0, prompt_len-1:prompt_len-1+len(sample["rejected_tokens"])] = 1

            # Forward pass - policy model
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                policy_chosen_logits = policy_model(chosen_input)
                policy_rejected_logits = policy_model(rejected_input)

            # Forward pass - reference model (no grad)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    ref_chosen_logits = ref_model(chosen_input)
                    ref_rejected_logits = ref_model(rejected_input)

            # Compute log probs
            policy_chosen_logps = compute_log_probs(policy_chosen_logits, chosen_target, chosen_mask)
            policy_rejected_logps = compute_log_probs(policy_rejected_logits, rejected_target, rejected_mask)
            ref_chosen_logps = compute_log_probs(ref_chosen_logits, chosen_target, chosen_mask)
            ref_rejected_logps = compute_log_probs(ref_rejected_logits, rejected_target, rejected_mask)

            # DPO loss
            loss, chosen_reward, rejected_reward, accuracy = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                config.beta, config.label_smoothing
            )

            # Backward
            (loss / config.batch_size).backward()

            total_loss += loss.item()
            total_chosen_reward += chosen_reward
            total_rejected_reward += rejected_reward
            total_accuracy += accuracy

        # Update
        optimizer.step()
        optimizer2.step()

        # Clear cache to avoid fragmentation
        if step % 10 == 0:
            torch.cuda.empty_cache()

        # Average metrics
        avg_loss = total_loss / config.batch_size
        avg_chosen = total_chosen_reward / config.batch_size
        avg_rejected = total_rejected_reward / config.batch_size
        avg_accuracy = total_accuracy / config.batch_size

        # Logging
        if step % 10 == 0:
            print(f"step:{step}/{config.num_iterations} loss:{avg_loss:.4f} acc:{avg_accuracy:.3f} "
                  f"chosen:{avg_chosen:.3f} rejected:{avg_rejected:.3f}")

        # Validation
        if step % config.val_every == 0:
            policy_model.eval()
            val_losses = []
            val_accs = []

            with torch.no_grad():
                for _ in range(min(50, len(val_loader))):
                    batch = val_loader.get_batch()

                    for sample in batch:
                        prompt_tokens = sample["prompt_tokens"]
                        chosen_tokens = sample["chosen_tokens"]
                        rejected_tokens = sample["rejected_tokens"]

                        chosen_seq = prompt_tokens + chosen_tokens
                        rejected_seq = prompt_tokens + rejected_tokens
                        max_len = max(len(chosen_seq), len(rejected_seq))
                        chosen_seq = chosen_seq + [0] * (max_len - len(chosen_seq))
                        rejected_seq = rejected_seq + [0] * (max_len - len(rejected_seq))

                        chosen_input = torch.tensor(chosen_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
                        chosen_target = torch.tensor(chosen_seq[1:], dtype=torch.long, device=device).unsqueeze(0)
                        rejected_input = torch.tensor(rejected_seq[:-1], dtype=torch.long, device=device).unsqueeze(0)
                        rejected_target = torch.tensor(rejected_seq[1:], dtype=torch.long, device=device).unsqueeze(0)

                        prompt_len = len(prompt_tokens)
                        chosen_mask = torch.zeros(1, max_len - 1, device=device)
                        rejected_mask = torch.zeros(1, max_len - 1, device=device)
                        chosen_mask[0, prompt_len-1:prompt_len-1+len(sample["chosen_tokens"])] = 1
                        rejected_mask[0, prompt_len-1:prompt_len-1+len(sample["rejected_tokens"])] = 1

                        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                            policy_chosen_logits = policy_model(chosen_input)
                            policy_rejected_logits = policy_model(rejected_input)
                            ref_chosen_logits = ref_model(chosen_input)
                            ref_rejected_logits = ref_model(rejected_input)

                        policy_chosen_logps = compute_log_probs(policy_chosen_logits, chosen_target, chosen_mask)
                        policy_rejected_logps = compute_log_probs(policy_rejected_logits, rejected_target, rejected_mask)
                        ref_chosen_logps = compute_log_probs(ref_chosen_logits, chosen_target, chosen_mask)
                        ref_rejected_logps = compute_log_probs(ref_rejected_logits, rejected_target, rejected_mask)

                        loss, _, _, acc = dpo_loss(
                            policy_chosen_logps, policy_rejected_logps,
                            ref_chosen_logps, ref_rejected_logps,
                            config.beta
                        )
                        val_losses.append(loss.item())
                        val_accs.append(acc)

            val_loss = sum(val_losses) / len(val_losses)
            val_acc = sum(val_accs) / len(val_accs)
            print(f"step:{step}/{config.num_iterations} val_loss:{val_loss:.4f} val_acc:{val_acc:.3f}")
            policy_model.train()

        # Save checkpoint
        if step % config.save_every == 0:
            ckpt_path = f"{log_dir}/dpo_step{step:06d}_loss{avg_loss:.4f}.pt"
            torch.save({
                'model': {k: v for k, v in policy_model.state_dict().items()},
                'step': step,
                'config': config.__dict__,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

    # Final save
    final_path = f"{log_dir}/dpo_final.pt"
    torch.save({
        'model': {k: v for k, v in policy_model.state_dict().items()},
        'step': config.num_iterations,
        'config': config.__dict__,
    }, final_path)
    print(f"\nDPO Training complete! Saved: {final_path}")

if __name__ == "__main__":
    main()
