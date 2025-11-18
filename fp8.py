#!/usr/bin/env python3
"""
Improved GPT-2 training with:
- FlexAttention with document boundary masking
- FP8 (e4m3fn) Linear Layers for RTX 4090 speedup (Forward pass only)
- Reduced Context (1024) for efficiency
- Progressive attention window (curriculum learning)
- Muon Optimizer
"""
import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()
import copy
import glob
import math
import threading
import time
import uuid
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward()

import torch._dynamo as dynamo
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import triton
import triton.language as tl
from torch import Tensor, nn

dynamo.config.recompile_limit = 64

# Compile FlexAttention
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(pid, M, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(configs=_get_autotune_configs(), key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"])
@triton.jit
def XXT_kernel(A_ptr, C_ptr, M, K, a_stride_b, a_stride_r, a_stride_c, c_stride_b, c_stride_r, c_stride_c,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
               GROUP_SIZE_M: tl.constexpr, LOWER_UPPER: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c
    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def XXT(A: torch.Tensor, out: torch.Tensor):
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0
    grid = lambda meta: (batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),)
    XXT_kernel[grid](A_ptr=A, C_ptr=out, M=M, K=K, a_stride_b=input_batch_stride, a_stride_r=A.stride(-2),
                     a_stride_c=A.stride(-1), c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1))
    return out

@triton.autotune(configs=_get_autotune_configs(), key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"])
@triton.jit
def ba_plus_cAA_kernel(A_ptr, C_ptr, M, a_stride_b, a_stride_r, a_stride_c, c_stride_b, c_stride_r, c_stride_c,
                       alpha, beta, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                       GROUP_SIZE_M: tl.constexpr, LOWER_UPPER: tl.constexpr):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)
    accumulator *= alpha
    accumulator += a_add * beta
    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K
    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0
    grid = lambda meta: (batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),)
    ba_plus_cAA_kernel[grid](A_ptr=A, C_ptr=out, M=M, a_stride_b=input_batch_stride, a_stride_r=A.stride(-2),
                             a_stride_c=A.stride(-1), c_stride_b=output_batch_stride, c_stride_r=out.stride(-2),
                             c_stride_c=out.stride(-1), alpha=alpha, beta=beta)
    return out

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
]

@torch.compile(dynamic=False, fullgraph=True)
def polar_express(G: torch.Tensor):
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)
    aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm
    for a, b, c in polar_express_coeffs:
        XXT(X, out=A)
        ba_plus_cAA(A, alpha=c, beta=b, out=B)
        aX_plus_BX(X, B, X, beta=a, out=C)
        X, C = C, X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# -----------------------------------------------------------------------------
# Muon Optimizer with DDP-style gradient distribution (even for single GPU)

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalization."""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.data.add_(g, alpha=-lr)

# -----------------------------------------------------------------------------
# Model Components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    """Standard Linear that casts weights to BF16 for forward pass."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()  # Zero init

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x))

class FP8Matmul(torch.autograd.Function):
    """
    Autograd function to handle FP8 matmul in forward pass 
    and standard BF16 matmul in backward pass.
    """
    @staticmethod
    def forward(ctx, x, weight):
        # x: [M, K], weight: [N, K] (Linear stores Transposed)
        
        # 1. Compute scales
        scale_x = x.abs().max() / 448.0
        scale_w = weight.abs().max() / 448.0
        
        # Safety
        scale_x = torch.max(scale_x, torch.tensor(1e-6, device=x.device))
        scale_w = torch.max(scale_w, torch.tensor(1e-6, device=x.device))
        
        # 2. Quantize
        x_fp8 = (x / scale_x).to(torch.float8_e4m3fn)
        w_fp8 = (weight / scale_w).to(torch.float8_e4m3fn)
        
        # 3. Matmul
        # torch._scaled_mm expects (M, K) and (K, N)
        # Our weight is [N, K]. w_fp8.T is [K, N].
        out = torch._scaled_mm(x_fp8, w_fp8.T, scale_a=scale_x, scale_b=scale_w, out_dtype=x.dtype)
        
        # Save inputs for backward
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = grad_weight = None
        
        # grad_output is usually BF16
        # weight is usually FP32 (master weights)
        
        # 1. dL/dX = dL/dY @ W
        if ctx.needs_input_grad[0]:
            # We must cast weight to match grad_output's dtype (BF16)
            # to avoid "expected mat1 and mat2 to have the same dtype"
            grad_x = grad_output @ weight.to(grad_output.dtype)
            
        # 2. dL/dW = (dL/dY).T @ X
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.T @ x
            
        return grad_x, grad_weight

class FP8Linear(nn.Linear):
    """
    Linear layer that performs matmul in FP8 (e4m3fn) using custom autograd function.
    Keeps master weights in BF16/FP32 (compatible with Muon).
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = hasattr(torch, 'float8_e4m3fn') and hasattr(torch, '_scaled_mm')

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.zero_()

    def forward(self, x: Tensor):
        if not self.use_fp8:
            return F.linear(x, self.weight.type_as(x))
        
        shape = x.shape
        x_2d = x.view(-1, shape[-1])
        out = FP8Matmul.apply(x_2d, self.weight)
        return out.view(*shape[:-1], -1)

class Rotary(torch.nn.Module):
    """Standard Rotary Position Embeddings."""
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
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, n_head, flex_kernel_options=None):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        # Use FP8Linear for heavy computations
        self.c_q = FP8Linear(dim, dim)
        self.c_k = FP8Linear(dim, dim)
        self.c_v = FP8Linear(dim, dim)
        # Initialize q/k/v with proper init
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_q.weight.uniform_(-bound, bound)
            self.c_k.weight.uniform_(-bound, bound)
            self.c_v.weight.uniform_(-bound, bound)
        # Value residual lambda (simple scalar)
        self.lamb = nn.Parameter(torch.tensor(0.5))
        # Rotary embeddings
        self.rotary = Rotary(dim // n_head)
        # Output projection (FP8)
        self.c_proj = FP8Linear(dim, dim)
        # Gated attention (Keep standard BF16 for small dims to avoid overhead)
        self.attn_gate = CastedLinear(12, n_head)
        # FlexAttention kernel options
        self.flex_kernel_options = flex_kernel_options

    def forward(self, x, v1, block_mask, attn_scale=0.1):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q = self.c_q(x).view(B, T, self.n_head, -1)
        k = self.c_k(x).view(B, T, self.n_head, -1)
        v = self.c_v(x).view(B, T, self.n_head, -1)
        if v1 is None:
            v1 = v
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            block_mask=block_mask,
            scale=attn_scale,
            kernel_options=self.flex_kernel_options
        )
        y = y.transpose(1, 2).contiguous()  # (B, T, n_head, head_dim)
        # Apply gated attention (your architecture) - gate each head separately
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).unsqueeze(-1)  # (B, T, n_head, 1)
        y = y * gate
        y = y.view_as(x)  # (B, T, n_embd)
        y = self.c_proj(y)
        return y, v1

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Use FP8Linear
        self.c_fc = FP8Linear(dim, 4 * dim)
        self.c_proj = FP8Linear(4 * dim, dim)
        # Initialize c_fc properly
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.c_fc.weight.uniform_(-bound, bound)
            # c_proj stays zero-init

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # Keep your architecture: skip attention on layers 0 and 7
        self.attn = CausalSelfAttention(config.n_embd, config.n_head, config.flex_kernel_options) if layer_idx not in [0, 7] else None
        self.mlp = MLP(config.n_embd) if layer_idx != 0 else None
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask, attn_scale):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x1, v1 = self.attn(norm(x), v1, block_mask, attn_scale)
            x = x + x1
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size: int = 50304  # Padded to multiple of 128
    n_layer: int = 12
    n_head: int = 6  # head dim 128
    n_embd: int = 768
    flex_kernel_options: dict = None

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # U-net design
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        ))
        # Use FP8Linear for the big head
        self.lm_head = FP8Linear(config.n_embd, config.vocab_size)

        # YOUR ARCHITECTURE: smear_gate and value_embeds
        # Keep gate in higher precision/standard linear as it is small
        self.smear_gate = CastedLinear(12, 1)
        self.value_embeds = nn.ModuleList([nn.Embedding(config.vocab_size, config.n_embd) for _ in range(3)])

        # Learnable scalars for smear and backout
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, idx, target, attn_blocksize):
        # Document boundary masking with progressive window
        docs = (idx == 50256).cumsum(0)

        def document_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            window_mask = q_idx - kv_idx < attn_blocksize
            return causal_mask & document_mask & window_mask

        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        # Forward pass
        x = self.transformer.wte(idx[None])

        # YOUR ARCHITECTURE: smear gate
        smear_gate_out = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)

        x = norm(x)
        x0 = x
        v1 = None

        # YOUR ARCHITECTURE: value embeddings
        ve = [value_embed(idx) for value_embed in self.value_embeds]
        ve_list = [None, ve[1], ve[2]] + [None] * (len(self.transformer.h) - 6) + [ve[0], ve[1], ve[2]]

        # Store outputs for U-Net skip connections
        skip_connections = []
        x_backout = None
        backout_layer = 8

        # Encoder pass
        for i in range(self.num_encoder_layers):
            # Inject value embeddings by updating v1 if applicable
            if ve_list[i] is not None:
                if v1 is None:
                    v1 = ve_list[i][None].view(1, S, self.transformer.h[i].attn.n_head if self.transformer.h[i].attn else 6, -1)
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask, attn_scale=0.1)
            skip_connections.append(x)
            if i == backout_layer:
                x_backout = x

        # Decoder pass with weighted skip connections
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            x = x + self.skip_weights[i] * skip_connections.pop()
            if ve_list[layer_idx] is not None:
                if v1 is None:
                    v1 = ve_list[layer_idx][None].view(1, S, 6, -1)
            x, v1 = self.transformer.h[layer_idx](x, v1, x0, block_mask, attn_scale=0.1)
            if layer_idx == backout_layer:
                x_backout = x

        # YOUR ARCHITECTURE: backout lambda
        if x_backout is not None:
            x = x - self.backout_lambda * x_backout

        x = norm(x)
        logits = self.lm_head(x)
        # IMPROVED: tanh logit scaling instead of sigmoid
        logits = 30 * torch.tanh(logits / 30)
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

# -----------------------------------------------------------------------------
# Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = torch.frombuffer(f.read(256*4), dtype=torch.int32)
    assert header[0] == 20240520
    assert header[1] == 1
    return int(header[2])

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = torch.frombuffer(f.read(256*4), dtype=torch.int32)
        assert header[0] == 20240520
        assert header[1] == 1
        ntok = header[2]
        tokens = torch.frombuffer(f.read(), dtype=torch.uint16)
    assert len(tokens) == ntok
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, T, process_rank=0, num_processes=1):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total
        self.reset()

    def reset(self):
        self.current_shard = -1
        self.advance()

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        batch_size = self.T * self.num_processes
        buf = self.tokens[self.current_position:self.current_position+self.T+1]
        buf = torch.tensor(buf.numpy().astype('int32'), dtype=torch.long)
        x = buf[:-1]
        y = buf[1:]
        self.current_position += batch_size
        if self.current_position + batch_size >= len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# Hyperparameters

@dataclass
class Hyperparameters:
    input_bin: str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size: int = 128  # Accumulation steps
    sequence_length: int = 1024  # Context length
    num_iterations: int = 3000   # Iterations
    warmup_iters: int = 0
    cooldown_iters: int = 640
    weight_decay: float = 0
    val_loss_every: int = 125
    val_tokens: int = 10485760
    save_every: int = 0

args = Hyperparameters()
model_config = GPTConfig(
    flex_kernel_options={
        "BLOCK_M": 64, "BLOCK_N": 64,  # forward
        "BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32  # backwards
    }
)

# Single GPU setup (fake DDP for compatibility)
assert torch.cuda.is_available()
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
device = 'cuda:0'
torch.cuda.set_device(device)
print(f"using device: {device}")

# Logging
run_id = str(uuid.uuid4())
logdir = f'logs/{run_id}/'
os.makedirs(logdir, exist_ok=True)
logfile = f'logs/{run_id}.txt'

def print0(s, logonly=False):
    with open(logfile, "a") as f:
        if not logonly:
            print(s)
        f.write(s+'\n')

with open(logfile, "w") as f:
    f.write(code)
    f.write('='*100 + '\n')

print0(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
import subprocess
result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
print0(f'{result.stdout}', logonly=True)
print0('='*100, logonly=True)

T = args.sequence_length
assert args.val_tokens % T == 0
val_steps = args.val_tokens // T
train_accumulation_steps = args.batch_size

# Load data
train_loader = DistributedDataLoader(args.input_bin, T, 0, 1)
val_loader = DistributedDataLoader(args.input_val_bin, T, 0, 1)
print0(f"Training DataLoader: total number of tokens: {train_loader.ntok_total}")
print0(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total}")
print0('='*100, logonly=True)
x, y = train_loader.next_batch()

# Create model
model = GPT(model_config)
model = model.cuda().bfloat16()
for m in model.modules():
    if isinstance(m, CastedLinear) or isinstance(m, FP8Linear):
        m.float() # Master weights in float/bf16

model = torch.compile(model)

# IMPROVED: Separate optimizers with different learning rates
# Like the reference: wte=0.6, lm_head=0.008, Muon=0.05, scalars=0.04
optimizer1 = torch.optim.Adam([model.transformer.wte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)
for ve in model.value_embeds:
    optimizer1.add_param_group({'params': [ve.weight], 'lr': 0.6})

optimizer2 = torch.optim.Adam([model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)

# Matrix params for Muon
params = list(model.transformer.h.parameters())
matrix_params = [p for p in params if p.ndim == 2]
scalar_params = [p for p in params if p.ndim < 2] + [model.skip_weights, model.smear_lambda, model.backout_lambda]

optimizer3 = Muon(matrix_params, lr=0.05, momentum=0.95)
optimizer4 = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True)

optimizers = [optimizer1, optimizer2, optimizer3, optimizer4]

# Learning rate schedule: no warmup, linear cooldown
def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    elif it < args.num_iterations - args.cooldown_iters:
        return 1.0
    else:
        decay_ratio = (args.num_iterations - it) / args.cooldown_iters
        return decay_ratio

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# Training loop
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 10:
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 11 else (step - 10) + 1

    # IMPROVED: Progressive attention window (64 to 1024)
    # Cap at 1024 since sequence_length is 1024
    attn_blocksize = torch.tensor(64 * ((step / args.num_iterations * (1024 - 64) + 64) // 64), dtype=torch.int, device='cuda')

    # Validation
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            with torch.no_grad():
                x_val, y_val = val_loader.next_batch()
                val_loss += model(x_val, y_val, attn_blocksize=attn_blocksize)
        val_loss /= val_steps
        print0(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')

        # Save checkpoint
        if step > 0:
            checkpoint_path = f'{logdir}checkpoint_step{step:06d}_loss{val_loss:.4f}.pt'
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, checkpoint_path)
            print0(f"Saved: {checkpoint_path}")

        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # Training
    model.train()
    for i in range(1, train_accumulation_steps + 1):
        loss = model(x, y, attn_blocksize=attn_blocksize)
        x, y = train_loader.next_batch()
        loss.backward()
        train_loss = loss.detach()

    for p in model.parameters():
        if p.grad is not None:
            p.grad /= train_accumulation_steps

    # IMPROVED: Muon momentum warmup (0.85 -> 0.95 over 300 steps)
    frac = min(step / 300, 1)
    optimizer3.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95

    # Step optimizers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)

    approx_time = training_time_ms + 1000 * (time.time() - t0)
    print0(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")

print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
print0(f"Training complete! Final run_id: {run_id}")