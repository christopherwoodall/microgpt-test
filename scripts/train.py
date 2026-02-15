#!/usr/bin/env python
"""Train microgpt on Gutenberg corpus"""

import os
import sys
import pickle
import random
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from microgpt import Value, linear, softmax, rmsnorm, matrix
import config

random.seed(42)


def load_corpus(corpus_name, split="train"):
    """Load documents from data/processed/{corpus_name}_{split}.txt"""
    filepath = Path(config.data_dir) / f"{corpus_name}_{split}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def initialize_model(vocab_size):
    """Initialize model parameters - COPY from microgpt.py lines 47-56"""
    state_dict = {
        'wte': matrix(vocab_size, config.n_embd),
        'wpe': matrix(config.block_size, config.n_embd),
        'lm_head': matrix(vocab_size, config.n_embd)
    }
    for i in range(config.n_layer):
        state_dict[f'layer{i}.attn_wq'] = matrix(config.n_embd, config.n_embd)
        state_dict[f'layer{i}.attn_wk'] = matrix(config.n_embd, config.n_embd)
        state_dict[f'layer{i}.attn_wv'] = matrix(config.n_embd, config.n_embd)
        state_dict[f'layer{i}.attn_wo'] = matrix(config.n_embd, config.n_embd)
        state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * config.n_embd, config.n_embd)
        state_dict[f'layer{i}.mlp_fc2'] = matrix(config.n_embd, 4 * config.n_embd)
    return state_dict


def save_checkpoint(state_dict, step, corpus_name, vocab):
    """Save checkpoint using pickle"""
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": state_dict, "step": step, "vocab": vocab}
    filepath = Path(config.checkpoint_dir) / f"{corpus_name}_step_{step}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint: {filepath}")


def load_checkpoint(filepath):
    """Load checkpoint from pickle"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint["state_dict"], checkpoint["step"], checkpoint["vocab"]


def gpt(token_id, pos_id, keys, values, state_dict):
    """Forward pass - COPY from microgpt.py lines 59-83, add state_dict parameter"""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(config.n_layer):
        # Attention block
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        
        # Multi-head attention
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
        
        # MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


def forward_pass(tokens, state_dict):
    """Forward pass returning loss - COPY pattern from microgpt.py lines 85-96"""
    keys = [[] for _ in range(config.n_layer)]
    values = [[] for _ in range(config.n_layer)]
    losses = []
    
    for pos_id in range(len(tokens) - 1):
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values, state_dict)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    
    loss = (1 / len(losses)) * sum(losses)
    return loss


def evaluate_validation(state_dict, val_docs, uchars, BOS):
    """Compute validation loss on sample of documents"""
    sample_docs = random.sample(val_docs, min(10, len(val_docs)))
    val_losses = []
    
    for doc in sample_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
        n = min(config.block_size, len(tokens) - 1)
        tokens = tokens[:n+1]
        loss = forward_pass(tokens, state_dict)
        val_losses.append(loss.data)
    
    return sum(val_losses) / len(val_losses)


def training_loop(corpus_name, num_steps, resume_from=None):
    """Main training loop"""
    print(f"Loading {corpus_name} corpus...")
    train_docs = load_corpus(corpus_name, "train")
    val_docs = load_corpus(corpus_name, "val")
    print(f"Train docs: {len(train_docs)}, Val docs: {len(val_docs)}")
    
    # Build vocabulary
    uchars = sorted(set(''.join(train_docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize or resume
    start_step = 0
    if resume_from:
        print(f"Resuming from {resume_from}")
        state_dict, start_step, uchars = load_checkpoint(resume_from)
        BOS = len(uchars)
    else:
        print("Initializing model...")
        state_dict = initialize_model(vocab_size)
    
    # Flatten parameters
    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"Parameters: {len(params)}")
    
    # Adam buffers
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    
    # Training history for visualization
    history = {
        'train_loss': [],
        'val_loss': [],
        'steps': [],
        'learning_rates': [],
        'timestamps': [],
        'corpus': corpus_name,
        'total_params': len(params),
        'vocab_size': vocab_size
    }
    
    print(f"\nTraining for {num_steps} steps...\n")
    start_time = datetime.now()
    
    # Training loop
    for step in range(start_step, num_steps):
        # Sample and tokenize
        doc = random.choice(train_docs)
        tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
        n = min(config.block_size, len(tokens) - 1)
        tokens = tokens[:n+1]
        
        # Forward + backward
        loss = forward_pass(tokens, state_dict)
        loss.backward()
        
        # Adam update - COPY from microgpt.py lines 102-109
        lr_t = config.learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = config.beta1 * m[i] + (1 - config.beta1) * p.grad
            v[i] = config.beta2 * v[i] + (1 - config.beta2) * p.grad ** 2
            m_hat = m[i] / (1 - config.beta1 ** (step + 1))
            v_hat = v[i] / (1 - config.beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + config.eps_adam)
            p.grad = 0
        
        # Progress
        if (step + 1) % 100 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f} | lr {lr_t:.6f} | time {elapsed:.1f}s")
            
            # Record history
            history['train_loss'].append(loss.data)
            history['steps'].append(step + 1)
            history['learning_rates'].append(lr_t)
            history['timestamps'].append(elapsed)
        
        # Checkpoint
        if (step + 1) % config.checkpoint_interval == 0:
            save_checkpoint(state_dict, step + 1, corpus_name, uchars)
        
        # Validation
        if (step + 1) % config.val_interval == 0:
            val_loss = evaluate_validation(state_dict, val_docs, uchars, BOS)
            print(f"       validation loss: {val_loss:.4f}")
            history['val_loss'].append(val_loss)
    
    # Final checkpoint
    save_checkpoint(state_dict, num_steps, corpus_name, uchars)
    
    # Save training history
    history_path = Path(config.checkpoint_dir) / f"{corpus_name}_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\nTraining history saved to {history_path}")
    print("\nTraining complete!")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train microgpt")
    parser.add_argument("--corpus", required=True, choices=["wilde", "lovecraft", "mixed"])
    parser.add_argument("--steps", type=int, default=config.num_steps)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    training_loop(args.corpus, args.steps, args.resume)


if __name__ == "__main__":
    main()
