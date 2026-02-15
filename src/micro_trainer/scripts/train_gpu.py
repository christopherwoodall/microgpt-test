#!/usr/bin/env python
"""
GPU-accelerated training for MicroGPT using PyTorch CUDA
~100-400x faster than pure Python version
"""

import os
import sys
import json
import pickle
import random
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from micro_trainer import config

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Check for GPU and SCREAM if not available
if not torch.cuda.is_available():
    print("\n" + "="*78)
    print("🔥🔥🔥 GPU NOT DETECTED 🔥🔥🔥")
    print("="*78)
    print("\n❌ ERROR: No CUDA-capable GPU found!")
    print("\nThis script requires a GPU for training. You have options:")
    print("\n1. 💻 Use CPU training instead:")
    print("   python scripts/train.py --corpus wilde --steps 1000")
    print("\n2. 🎮 Install PyTorch with CUDA:")
    print("   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\n3. 🐳 Use a Docker container with GPU support")
    print("\n4. ☁️  Run on Google Colab (free GPU):")
    print("   https://colab.research.google.com")
    print("\n" + "="*78 + "\n")
    sys.exit(1)

# Device configuration
DEVICE = torch.device('cuda')
print(f"\n✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention"""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # Key, query, value projections
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)


class MLP(nn.Module):
    """Feed-forward network with ReLU"""
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)
    
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.ln1(x))
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x


class MicroGPT(nn.Module):
    """Complete GPT model"""
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=256):
        super().__init__()
        self.block_size = block_size
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size)
            for _ in range(n_layer)
        ])
        
        # Final layer norm and output
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        idx: (B, T) tensor of token indices
        targets: (B, T) tensor of target indices (optional)
        """
        B, T = idx.size()
        
        # Get embeddings
        token_embeddings = self.token_emb(idx)
        pos_embeddings = self.pos_emb(torch.arange(T, device=idx.device))
        x = token_embeddings + pos_embeddings
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens"""
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


def load_corpus(corpus_name, split="train"):
    """Load documents from data/processed/{corpus_name}_{split}.txt"""
    filepath = Path(config.data_dir) / f"{corpus_name}_{split}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_checkpoint_gpu(model, optimizer, step, corpus_name, vocab, loss=None):
    """Save GPU checkpoint"""
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'vocab': vocab,
        'loss': loss,
        'config': {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
        }
    }
    
    filepath = Path(config.checkpoint_dir) / f"{corpus_name}_step_{step}_gpu.pkl"
    torch.save(checkpoint, filepath)
    print(f"\n💾 GPU CHECKPOINT SAVED: {filepath}\n")
    return filepath


def load_checkpoint_gpu(filepath):
    """Load GPU checkpoint"""
    checkpoint = torch.load(filepath, map_location=DEVICE)
    return checkpoint


def create_batches(docs, uchars, BOS, batch_size=4, block_size=256):
    """Create training batches with padding"""
    batch_docs = random.sample(docs, min(batch_size, len(docs)))
    
    sequences = []
    for doc in batch_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
        sequences.append(tokens)
    
    # Find max length in batch
    max_len = min(max(len(s) for s in sequences), block_size + 1)
    
    # Pad sequences
    x_batch = []
    y_batch = []
    
    for tokens in sequences:
        # Truncate if too long
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # Input is tokens[:-1], target is tokens[1:]
        x = tokens[:-1]
        y = tokens[1:]
        
        # Pad to max_len - 1
        x = x + [BOS] * (max_len - 1 - len(x))
        y = y + [BOS] * (max_len - 1 - len(y))
        
        x_batch.append(x)
        y_batch.append(y)
    
    return torch.tensor(x_batch, dtype=torch.long, device=DEVICE), \
           torch.tensor(y_batch, dtype=torch.long, device=DEVICE)


def training_loop_gpu(corpus_name, num_steps, resume_from=None, batch_size=4):
    """Main GPU training loop with batching"""
    print("\n" + "="*78)
    print(f"🚀 MICROGPT GPU TRAINING - {corpus_name.upper()}")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Steps: {num_steps} | Batch Size: {batch_size}")
    print(f"   Architecture: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
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
        print(f"\n📂 Resuming from GPU checkpoint: {resume_from}")
        checkpoint = load_checkpoint_gpu(resume_from)
        
        # Create model from saved config
        cfg = checkpoint['config']
        model = MicroGPT(vocab_size, cfg['n_embd'], cfg['n_head'], 
                        cfg['n_layer'], cfg['block_size']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_step = checkpoint['step']
        
        # Restore optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, 
                                    betas=(config.beta1, config.beta2), 
                                    eps=config.eps_adam)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("\n🎲 Initializing fresh GPU model...")
        model = MicroGPT(vocab_size, config.n_embd, config.n_head, 
                        config.n_layer, config.block_size).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                    betas=(config.beta1, config.beta2),
                                    eps=config.eps_adam)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parameters: {total_params:,}")
    print(f"   ✓ GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    
    # Learning rate scheduler (linear decay)
    def lr_lambda(step):
        return 1.0 - (step / num_steps)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Setup logging
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    json_log_path = Path(config.checkpoint_dir) / f"{corpus_name}_training_log_gpu.jsonl"
    
    print(f"\n{'─'*78}")
    print("🔥 GPU TRAINING STARTED - This is gonna be FAST...")
    print(f"{'─'*78}\n")
    
    model.train()
    start_time = time.time()
    losses = []
    
    # Training loop with progress bar
    pbar = tqdm(range(start_step, num_steps), initial=start_step, total=num_steps,
                desc="Training", unit="step", ncols=78)
    
    for step in pbar:
        # Create batch
        x, y = create_batches(train_docs, uchars, BOS, batch_size, config.block_size)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track loss
        loss_item = loss.item()
        losses.append(loss_item)
        
        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss_item:.4f}',
            'lr': f'{current_lr:.6f}',
            'gpu': f'{torch.cuda.memory_allocated() / 1e9:.2f}GB'
        })
        
        # Log every 100 steps
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1 - start_step) / elapsed
            
            log_entry = {
                'step': step + 1,
                'train_loss': loss_item,
                'learning_rate': current_lr,
                'elapsed_seconds': elapsed,
                'steps_per_second': steps_per_sec,
                'gpu_memory_gb': torch.cuda.memory_allocated() / 1e9,
                'timestamp': datetime.now().isoformat()
            }
            with open(json_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Checkpoint
        if (step + 1) % config.checkpoint_interval == 0:
            save_checkpoint_gpu(model, optimizer, step + 1, corpus_name, uchars, loss_item)
        
        # Validation
        if (step + 1) % config.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_docs_sample = random.sample(val_docs, min(10, len(val_docs)))
                val_losses = []
                
                for doc in val_docs_sample:
                    tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
                    if len(tokens) > 1:
                        x_val = torch.tensor([tokens[:-1]], dtype=torch.long, device=DEVICE)
                        y_val = torch.tensor([tokens[1:]], dtype=torch.long, device=DEVICE)
                        _, vloss = model(x_val, y_val)
                        val_losses.append(vloss.item())
                
                avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
                tqdm.write(f"\n📊 Validation Loss: {avg_val_loss:.4f}\n")
                
                log_entry = {
                    'step': step + 1,
                    'val_loss': avg_val_loss,
                    'timestamp': datetime.now().isoformat()
                }
                with open(json_log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            
            model.train()
    
    # Final checkpoint
    save_checkpoint_gpu(model, optimizer, num_steps, corpus_name, uchars, losses[-1] if losses else None)
    
    # Save summary
    elapsed = time.time() - start_time
    summary = {
        'corpus': corpus_name,
        'total_steps': num_steps,
        'total_params': total_params,
        'vocab_size': vocab_size,
        'final_train_loss': losses[-1] if losses else None,
        'total_time_seconds': elapsed,
        'avg_steps_per_second': (num_steps - start_step) / elapsed,
        'config': {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
            'batch_size': batch_size,
        },
        'device': torch.cuda.get_device_name(0),
    }
    
    summary_path = Path(config.checkpoint_dir) / f"{corpus_name}_summary_gpu.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*78}")
    print("✅ GPU TRAINING COMPLETE!")
    print(f"{'='*78}")
    print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
    print(f"🚀 Speed: {(num_steps - start_step) / elapsed:.1f} steps/sec")
    if losses:
        print(f"📉 Final loss: {losses[-1]:.4f}")
    print(f"📊 Summary: {summary_path}")
    print(f"{'='*78}\n")


def main():
    parser = argparse.ArgumentParser(description="Train microgpt on GPU with PyTorch CUDA")
    parser.add_argument("--corpus", required=True, choices=["wilde", "lovecraft", "mixed"],
                       help="Training corpus")
    parser.add_argument("--steps", type=int, default=config.num_steps,
                       help="Number of training steps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to GPU checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for training (default: 4)")
    args = parser.parse_args()
    
    training_loop_gpu(args.corpus, args.steps, args.resume, args.batch_size)


if __name__ == "__main__":
    main()
