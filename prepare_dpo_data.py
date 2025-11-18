#!/usr/bin/env python3
"""
Prepare UltraFeedback dataset for DPO training.
Downloads from HuggingFace, applies ChatML template, tokenizes preference pairs.

Output: JSON lines with tokenized (prompt, chosen, rejected) triplets.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import tiktoken

# ChatML special tokens
IM_START_ID = 50257
IM_END_ID = 50258
EOT_ID = 50256

def tokenize_chatml_message(role, content, enc):
    """Tokenize a single message in ChatML format."""
    tokens = [IM_START_ID]
    tokens.extend(enc.encode(role + "\n", allowed_special=set()))
    tokens.extend(enc.encode(content, allowed_special=set()))
    tokens.append(IM_END_ID)
    tokens.extend(enc.encode("\n", allowed_special=set()))
    return tokens

def format_prompt_tokens(messages, enc):
    """Format conversation history as prompt tokens."""
    tokens = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        tokens.extend(tokenize_chatml_message(role, content, enc))

    # Add assistant turn start
    tokens.append(IM_START_ID)
    tokens.extend(enc.encode("assistant\n", allowed_special=set()))
    return tokens

def format_response_tokens(content, enc):
    """Format response tokens (without the im_start|assistant prefix)."""
    tokens = enc.encode(content, allowed_special=set())
    tokens.append(IM_END_ID)
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/ultrafeedback")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=1024, help="Max total sequence length")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    enc = tiktoken.get_encoding("gpt2")

    print("Loading UltraFeedback dataset...")
    from datasets import load_dataset

    # Use the binarized version with clear chosen/rejected
    dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
    print(f"Loaded {len(dataset)} samples")

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Shuffle
    dataset = dataset.shuffle(seed=42)

    # Process samples
    processed = []
    skipped = 0

    for sample in tqdm(dataset, desc="Processing"):
        try:
            # Extract prompt and responses
            prompt = sample.get("prompt", "")
            chosen_response = sample.get("chosen", [])
            rejected_response = sample.get("rejected", [])

            # Handle different formats
            if isinstance(chosen_response, list) and len(chosen_response) > 0:
                # Format: [{"role": "...", "content": "..."}]
                chosen_text = chosen_response[-1].get("content", "") if isinstance(chosen_response[-1], dict) else str(chosen_response[-1])
            else:
                chosen_text = str(chosen_response)

            if isinstance(rejected_response, list) and len(rejected_response) > 0:
                rejected_text = rejected_response[-1].get("content", "") if isinstance(rejected_response[-1], dict) else str(rejected_response[-1])
            else:
                rejected_text = str(rejected_response)

            if not prompt or not chosen_text or not rejected_text:
                skipped += 1
                continue

            # Tokenize
            prompt_messages = [{"role": "user", "content": prompt}]
            prompt_tokens = format_prompt_tokens(prompt_messages, enc)
            chosen_tokens = format_response_tokens(chosen_text, enc)
            rejected_tokens = format_response_tokens(rejected_text, enc)

            # Check length
            max_response_len = args.max_length - len(prompt_tokens)
            if max_response_len < 10:
                skipped += 1
                continue

            # Truncate responses if needed
            if len(chosen_tokens) > max_response_len:
                chosen_tokens = chosen_tokens[:max_response_len-1] + [IM_END_ID]
            if len(rejected_tokens) > max_response_len:
                rejected_tokens = rejected_tokens[:max_response_len-1] + [IM_END_ID]

            processed.append({
                "prompt_tokens": prompt_tokens,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
            })

        except Exception as e:
            skipped += 1
            continue

    print(f"\nProcessed: {len(processed)}, Skipped: {skipped}")

    # Split train/val
    n_val = int(len(processed) * args.val_ratio)
    n_train = len(processed) - n_val

    train_data = processed[:n_train]
    val_data = processed[n_train:]

    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")

    # Save as JSON lines
    train_file = f"{args.output_dir}/train.jsonl"
    val_file = f"{args.output_dir}/val.jsonl"

    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    # Stats
    avg_prompt_len = sum(len(x["prompt_tokens"]) for x in processed) / len(processed)
    avg_chosen_len = sum(len(x["chosen_tokens"]) for x in processed) / len(processed)
    avg_rejected_len = sum(len(x["rejected_tokens"]) for x in processed) / len(processed)

    print(f"\nSaved to {args.output_dir}/")
    print(f"Average prompt length: {avg_prompt_len:.1f} tokens")
    print(f"Average chosen length: {avg_chosen_len:.1f} tokens")
    print(f"Average rejected length: {avg_rejected_len:.1f} tokens")

if __name__ == "__main__":
    main()
