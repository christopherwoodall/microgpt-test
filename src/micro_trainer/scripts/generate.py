#!/usr/bin/env python
"""Generate text from trained microgpt"""

import sys
import pickle
import random
import argparse
from pathlib import Path

from micro_trainer.microgpt import Value, linear, softmax, rmsnorm
from micro_trainer import config

random.seed(42)


def load_checkpoint(filepath):
    """Load checkpoint"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint["state_dict"], checkpoint["vocab"]


def gpt(token_id, pos_id, keys, values, state_dict):
    """Forward pass - SAME as train.py"""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(config.n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []
        for h in range(config.n_head):
            hs = h * config.head_dim
            q_h = q[hs:hs+config.head_dim]
            k_h = [ki[hs:hs+config.head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+config.head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(config.head_dim)) / config.head_dim**0.5 
                          for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) 
                       for j in range(config.head_dim)]
            x_attn.extend(head_out)
        
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


def generate_sample(state_dict, uchars, prompt=None, temperature=0.7, max_length=500):
    """Generate single sample - COPY pattern from microgpt.py lines 110-120"""
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    
    keys = [[] for _ in range(config.n_layer)]
    values = [[] for _ in range(config.n_layer)]
    
    # Start tokens
    if prompt:
        tokens = [BOS] + [uchars.index(ch) for ch in prompt if ch in uchars]
    else:
        tokens = [BOS]
    
    # Generate
    for pos_id in range(len(tokens) - 1, max_length):
        token_id = tokens[-1]
        logits = gpt(token_id, pos_id, keys, values, state_dict)
        probs = softmax([l / temperature for l in logits])
        next_token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        if next_token == BOS:
            break
        
        tokens.append(next_token)
    
    # Detokenize
    text = ''.join([uchars[t] for t in tokens[1:] if t < len(uchars)])
    return text


def main():
    parser = argparse.ArgumentParser(description="Generate text")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--output", type=str, default=None, help="Output file for samples")
    args = parser.parse_args()
    
    print(f"Loading {args.checkpoint}...")
    state_dict, uchars = load_checkpoint(args.checkpoint)
    print(f"Vocab size: {len(uchars) + 1}\n")
    
    samples = []
    print("=" * 70)
    for i in range(args.num_samples):
        sample = generate_sample(
            state_dict, uchars,
            prompt=args.prompt,
            temperature=args.temperature,
            max_length=args.max_length
        )
        samples.append(sample)
        print(f"\nSample {i+1}:")
        print("-" * 70)
        print(sample)
        print("-" * 70)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(samples):
                f.write(f"Sample {i+1}:\n")
                f.write("-" * 70 + "\n")
                f.write(sample + "\n")
                f.write("-" * 70 + "\n\n")
        print(f"\nSamples saved to {args.output}")


if __name__ == "__main__":
    main()
