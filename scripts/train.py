#!/usr/bin/env python
"""Train microgpt on Gutenberg corpus with VERBOSE output"""

import os
import sys
import json
import pickle
import random
import argparse
import time
from pathlib import Path
from datetime import datetime

# Increase recursion limit for deep backward passes with 2-layer model
sys.setrecursionlimit(10000)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from microgpt import Value, linear, softmax, rmsnorm, matrix
import config

random.seed(42)

# Spinner characters for showing movement
SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']


def load_corpus(corpus_name, split="train"):
    """Load documents from data/processed/{corpus_name}_{split}.txt"""
    filepath = Path(config.data_dir) / f"{corpus_name}_{split}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def initialize_model(vocab_size):
    """Initialize model parameters"""
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
    print(f"\n💾 CHECKPOINT SAVED: {filepath}\n")


def load_checkpoint(filepath):
    """Load checkpoint from pickle"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint["state_dict"], checkpoint["step"], checkpoint["vocab"]


def gpt(token_id, pos_id, keys, values, state_dict):
    """Forward pass with state_dict parameter"""
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
    """Forward pass returning loss"""
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


def format_time(seconds):
    """Format seconds into readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def training_loop(corpus_name, num_steps, resume_from=None):
    """Main training loop with CONSTANT VERBOSE OUTPUT"""
    print("\n" + "="*78)
    print(f"🚀 MICROGPT TRAINING - {corpus_name.upper()}")
    print(f"   Steps: {num_steps} | Arch: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
    print("="*78)
    
    print(f"\n📚 Loading corpus...")
    train_docs = load_corpus(corpus_name, "train")
    val_docs = load_corpus(corpus_name, "val")
    print(f"   ✓ Train: {len(train_docs):,} docs")
    print(f"   ✓ Val:   {len(val_docs):,} docs")
    
    # Build vocabulary
    uchars = sorted(set(''.join(train_docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"   ✓ Vocab: {vocab_size} tokens")
    
    # Initialize or resume
    start_step = 0
    if resume_from:
        print(f"\n📂 Resuming from: {resume_from}")
        state_dict, start_step, uchars = load_checkpoint(resume_from)
        BOS = len(uchars)
    else:
        print("\n🎲 Initializing fresh model...")
        state_dict = initialize_model(vocab_size)
    
    # Flatten parameters
    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"   ✓ Parameters: {len(params):,}")
    
    # Adam buffers
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    
    # Setup logging
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    json_log_path = Path(config.checkpoint_dir) / f"{corpus_name}_training_log.jsonl"
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'steps': [],
        'learning_rates': [],
        'timestamps': [],
        'corpus': corpus_name,
        'total_params': len(params),
        'vocab_size': vocab_size,
    }
    
    print(f"\n{'─'*78}")
    print("🔥 TRAINING STARTED - Watching loss converge in real-time...")
    print(f"{'─'*78}\n")
    
    start_time = time.time()
    last_log_time = start_time
    step_times = []  # Track recent step times for ETA
    
    # Training loop
    for step in range(start_step, num_steps):
        step_start = time.time()
        
        # Sample and tokenize
        doc = random.choice(train_docs)
        tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
        n = min(config.block_size, len(tokens) - 1)
        tokens = tokens[:n+1]
        
        # Forward + backward
        loss = forward_pass(tokens, state_dict)
        loss.backward()
        
        # Adam update
        lr_t = config.learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = config.beta1 * m[i] + (1 - config.beta1) * p.grad
            v[i] = config.beta2 * v[i] + (1 - config.beta2) * p.grad ** 2
            m_hat = m[i] / (1 - config.beta1 ** (step + 1))
            v_hat = v[i] / (1 - config.beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + config.eps_adam)
            p.grad = 0
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        if len(step_times) > 50:
            step_times.pop(0)  # Keep last 50 for moving average
        
        current_time = time.time()
        elapsed = current_time - start_time
        steps_done = step - start_step + 1
        steps_remaining = num_steps - step - 1
        avg_step_time = sum(step_times) / len(step_times) if step_times else step_time
        eta = steps_remaining * avg_step_time
        
        # Progress bar
        progress = (step + 1) / num_steps
        bar_width = 30
        filled = int(bar_width * progress)
        bar = '█' * filled + '░' * (bar_width - filled)
        
        # Spinner
        spinner = SPINNER[step % len(SPINNER)]
        
        # Build status line
        status = f"\r{spinner} Step {step+1:>6}/{num_steps} [{bar}] {progress*100:5.1f}% | Loss: {loss.data:7.4f} | {avg_step_time:5.2f}s/step | ETA: {format_time(eta)}"
        
        # Print in-place (no newline)
        print(status, end='', flush=True)
        
        # Every 100 steps, add a newline and log
        if (step + 1) % 100 == 0:
            print()  # New line
            
            # Record history
            history['train_loss'].append(loss.data)
            history['steps'].append(step + 1)
            history['learning_rates'].append(lr_t)
            history['timestamps'].append(elapsed)
            
            # Write JSON log
            log_entry = {
                'step': step + 1,
                'train_loss': loss.data,
                'learning_rate': lr_t,
                'elapsed_seconds': elapsed,
                'steps_per_second': 1.0 / avg_step_time if avg_step_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            with open(json_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Checkpoint
        if (step + 1) % config.checkpoint_interval == 0:
            print(f"\n💾 Saving checkpoint at step {step+1}...")
            save_checkpoint(state_dict, step + 1, corpus_name, uchars)
        
        # Validation
        if (step + 1) % config.val_interval == 0:
            print(f"\n📊 Running validation...")
            val_loss = evaluate_validation(state_dict, val_docs, uchars, BOS)
            print(f"✓ Validation loss: {val_loss:.4f}\n")
            history['val_loss'].append(val_loss)
            
            log_entry = {
                'step': step + 1,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            }
            with open(json_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
    
    # Final newline if we ended mid-line
    print()
    
    # Final checkpoint
    save_checkpoint(state_dict, num_steps, corpus_name, uchars)
    
    # Save training history
    history_path = Path(config.checkpoint_dir) / f"{corpus_name}_history.pkl"
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    # Save summary JSON
    summary = {
        'corpus': corpus_name,
        'total_steps': num_steps,
        'total_params': len(params),
        'vocab_size': vocab_size,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else None,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        'total_time_seconds': elapsed,
        'config': {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
        },
        'logs': {
            'jsonl': str(json_log_path),
            'history_pkl': str(history_path)
        }
    }
    summary_path = Path(config.checkpoint_dir) / f"{corpus_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*78}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*78}")
    print(f"⏱️  Total time: {format_time(elapsed)}")
    if history['train_loss']:
        print(f"📉 Final loss: {history['train_loss'][-1]:.4f}")
    print(f"📊 Logs: {config.checkpoint_dir}/")
    print(f"{'='*78}\n")
    
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
