#!/usr/bin/env python
"""
Alternating corpus trainer for MicroGPT GPU
Trains on Wilde and Lovecraft (or any corpora) in random chunks
Perfect for creating a hybrid writing style!
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from micro_trainer import config

# Set random seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Check for GPU
if not torch.cuda.is_available():
    print("\n" + "="*78)
    print("🔥🔥🔖 GPU NOT DETECTED 🔥🔥🔖")
    print("="*78)
    print("\n❌ This script requires a GPU. Use train_gpu.py for single-corpus training.")
    print("\n" + "="*78 + "\n")
    sys.exit(1)

DEVICE = torch.device('cuda')

# Corpus colors for terminal output
CORPUS_COLORS = {
    'wilde': '\033[95m',      # Purple
    'lovecraft': '\033[94m',   # Blue
    'mixed': '\033[96m',       # Cyan
    'reset': '\033[0m'
}

SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']


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
        
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(out)


class MLP(nn.Module):
    """Feed-forward network"""
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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MicroGPT(nn.Module):
    """Complete GPT model"""
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=256):
        super().__init__()
        self.block_size = block_size
        
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size)
            for _ in range(n_layer)
        ])
        
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        token_embeddings = self.token_emb(idx)
        pos_embeddings = self.pos_emb(torch.arange(T, device=idx.device))
        x = token_embeddings + pos_embeddings
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


def load_corpus(corpus_name, split="train"):
    """Load documents"""
    filepath = Path(config.data_dir) / f"{corpus_name}_{split}.txt"
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def build_vocab(docs_list):
    """Build vocabulary from multiple corpora"""
    all_chars = set()
    for docs in docs_list:
        all_chars.update(''.join(docs))
    uchars = sorted(all_chars)
    return uchars


def create_batches(docs, uchars, BOS, batch_size=4, block_size=256):
    """Create training batches"""
    batch_docs = random.sample(docs, min(batch_size, len(docs)))
    
    sequences = []
    for doc in batch_docs:
        tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
        sequences.append(tokens)
    
    max_len = min(max(len(s) for s in sequences), block_size + 1)
    
    x_batch = []
    y_batch = []
    
    for tokens in sequences:
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        x = tokens[:-1]
        y = tokens[1:]
        
        x = x + [BOS] * (max_len - 1 - len(x))
        y = y + [BOS] * (max_len - 1 - len(y))
        
        x_batch.append(x)
        y_batch.append(y)
    
    return torch.tensor(x_batch, dtype=torch.long, device=DEVICE), \
           torch.tensor(y_batch, dtype=torch.long, device=DEVICE)


def evaluate_on_corpus(model, docs, uchars, BOS, num_samples=10):
    """Evaluate model on a corpus"""
    model.eval()
    sample_docs = random.sample(docs, min(num_samples, len(docs)))
    losses = []
    
    with torch.no_grad():
        for doc in sample_docs:
            tokens = [BOS] + [uchars.index(ch) for ch in doc if ch in uchars] + [BOS]
            if len(tokens) > 1:
                x = torch.tensor([tokens[:-1]], dtype=torch.long, device=DEVICE)
                y = torch.tensor([tokens[1:]], dtype=torch.long, device=DEVICE)
                _, loss = model(x, y)
                losses.append(loss.item())
    
    model.train()
    return sum(losses) / len(losses) if losses else 0.0


def format_time(seconds):
    """Format seconds into readable time"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def training_loop_alternating(corpora, total_steps, min_chunk, max_chunk, batch_size=4, validate_interval=500):
    """Main alternating training loop"""
    
    # Validate corpora
    for corpus in corpora:
        if corpus not in ["wilde", "lovecraft", "mixed"]:
            print(f"❌ Unknown corpus: {corpus}")
            sys.exit(1)
    
    print("\n" + "="*78)
    print(f"🎭 ALTERNATING CORPUS TRAINING")
    print(f"   Corpora: {', '.join(corpora)}")
    print(f"   Total Steps: {total_steps}")
    print(f"   Chunk Size: {min_chunk}-{max_chunk} steps")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print("="*78)
    
    # Load all corpora
    print(f"\n📚 Loading corpora...")
    corpora_data = {}
    for corpus in corpora:
        train_docs = load_corpus(corpus, "train")
        val_docs = load_corpus(corpus, "val")
        corpora_data[corpus] = {
            'train': train_docs,
            'val': val_docs
        }
        color = CORPUS_COLORS.get(corpus, '')
        reset = CORPUS_COLORS['reset']
        print(f"   {color}✓ {corpus.upper()}{reset}: {len(train_docs):,} train, {len(val_docs):,} val")
    
    # Build unified vocabulary from all corpora
    all_train_docs = [corpora_data[c]['train'] for c in corpora]
    uchars = build_vocab(all_train_docs)
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"\n   ✓ Unified vocabulary: {vocab_size} tokens")
    
    # Initialize model
    print("\n🎲 Initializing model...")
    model = MicroGPT(vocab_size, config.n_embd, config.n_head, 
                    config.n_layer, config.block_size).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                betas=(config.beta1, config.beta2),
                                eps=config.eps_adam)
    
    # Setup logging
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    corpus_names = '_'.join(corpora)
    json_log_path = Path(config.checkpoint_dir) / f"{corpus_names}_alternating_training_log.jsonl"
    
    # Pick random starting corpus
    current_corpus = random.choice(corpora)
    steps_in_current_chunk = 0
    current_chunk_size = random.randint(min_chunk, max_chunk)
    
    print(f"\n{'─'*78}")
    print(f"🔥 TRAINING STARTED - Alternating between {', '.join(corpora)}...")
    print(f"{'─'*78}\n")
    
    start_time = time.time()
    chunk_number = 1
    
    pbar = tqdm(range(total_steps), total=total_steps, desc="Training", unit="step", ncols=78)
    
    for step in pbar:
        # Check if we should switch corpus
        if steps_in_current_chunk >= current_chunk_size:
            # Switch to random different corpus
            new_corpus = random.choice([c for c in corpora if c != current_corpus])
            current_corpus = new_corpus
            steps_in_current_chunk = 0
            current_chunk_size = random.randint(min_chunk, max_chunk)
            chunk_number += 1
            
            color = CORPUS_COLORS.get(current_corpus, '')
            reset = CORPUS_COLORS['reset']
            tqdm.write(f"\n{color}🔄 Switching to {current_corpus.upper()} corpus (chunk {chunk_number}, size {current_chunk_size}){reset}\n")
        
        # Get data for current corpus
        train_docs = corpora_data[current_corpus]['train']
        
        # Create batch
        x, y = create_batches(train_docs, uchars, BOS, batch_size, config.block_size)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_item = loss.item()
        steps_in_current_chunk += 1
        
        # Update progress bar with corpus indicator
        color = CORPUS_COLORS.get(current_corpus, '')
        reset = CORPUS_COLORS['reset']
        pbar.set_postfix({
            'loss': f'{loss_item:.4f}',
            'corpus': f"{color}{current_corpus[:3].upper()}{reset}",
            'chunk': f'{steps_in_current_chunk}/{current_chunk_size}'
        })
        
        # Log every 100 steps
        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            
            log_entry = {
                'step': step + 1,
                'train_loss': loss_item,
                'corpus': current_corpus,
                'chunk_number': chunk_number,
                'chunk_progress': steps_in_current_chunk,
                'chunk_size': current_chunk_size,
                'elapsed_seconds': elapsed,
                'steps_per_second': steps_per_sec,
                'timestamp': datetime.now().isoformat()
            }
            with open(json_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        # Validation on both corpora
        if (step + 1) % validate_interval == 0:
            tqdm.write(f"\n📊 Running validation on all corpora at step {step+1}...")
            
            for corpus in corpora:
                val_loss = evaluate_on_corpus(model, corpora_data[corpus]['val'], 
                                             uchars, BOS, num_samples=10)
                color = CORPUS_COLORS.get(corpus, '')
                reset = CORPUS_COLORS['reset']
                tqdm.write(f"   {color}{corpus.upper()}{reset} val loss: {val_loss:.4f}")
                
                log_entry = {
                    'step': step + 1,
                    'val_loss': val_loss,
                    'corpus': corpus,
                    'timestamp': datetime.now().isoformat()
                }
                with open(json_log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            
            tqdm.write("")
    
    # Save final checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': total_steps,
        'vocab': uchars,
        'corpora': corpora,
        'config': {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
        }
    }
    
    checkpoint_path = Path(config.checkpoint_dir) / f"{corpus_names}_alternating_step_{total_steps}_gpu.pkl"
    torch.save(checkpoint, checkpoint_path)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*78}")
    print("✅ ALTERNATING TRAINING COMPLETE!")
    print(f"{'='*78}")
    print(f"⏱️  Total time: {format_time(elapsed)}")
    print(f"🎭 Corpora: {', '.join(corpora)}")
    print(f"📊 Total chunks: {chunk_number}")
    print(f"💾 Checkpoint: {checkpoint_path}")
    print(f"{'='*78}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train MicroGPT on alternating corpora (GPU only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Alternate between Wilde and Lovecraft
  python train_alternating.py --corpora wilde lovecraft --total-steps 20000
  
  # All three corpora with custom chunk sizes
  python train_alternating.py --corpora wilde lovecraft mixed \\
    --total-steps 30000 --min-chunk 200 --max-chunk 800
  
  # Faster training with larger batches
  python train_alternating.py --corpora wilde lovecraft \\
    --total-steps 20000 --batch-size 8
        """
    )
    
    parser.add_argument("--corpora", nargs='+', required=True,
                       choices=["wilde", "lovecraft", "mixed"],
                       help="List of corpora to alternate between (e.g., wilde lovecraft)")
    parser.add_argument("--total-steps", type=int, required=True,
                       help="Total number of training steps")
    parser.add_argument("--min-chunk", type=int, default=100,
                       help="Minimum chunk size (steps per corpus)")
    parser.add_argument("--max-chunk", type=int, default=500,
                       help="Maximum chunk size (steps per corpus)")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--validate-interval", type=int, default=500,
                       help="Validate on all corpora every N steps")
    
    args = parser.parse_args()
    
    if len(args.corpora) < 2:
        print("❌ Error: Need at least 2 corpora to alternate between!")
        sys.exit(1)
    
    if args.min_chunk >= args.max_chunk:
        print("❌ Error: min-chunk must be less than max-chunk!")
        sys.exit(1)
    
    training_loop_alternating(
        corpora=args.corpora,
        total_steps=args.total_steps,
        min_chunk=args.min_chunk,
        max_chunk=args.max_chunk,
        batch_size=args.batch_size,
        validate_interval=args.validate_interval
    )


if __name__ == "__main__":
    main()
