#!/usr/bin/env python
"""
Visualization suite for microgpt training
Generate ASCII art, progress bars, and training charts
"""

import sys
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class Visualizer:
    """Heavy visualization engine for microgpt"""
    
    def __init__(self):
        self.colors = config.VIZ_COLORS
        self.arch = config.ARCH_VIZ
        self.width = 70
    
    def _color(self, text, color_key):
        """Add ANSI color codes"""
        color_map = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'reset': '\033[0m'
        }
        return f"{color_map.get(color_key, '')}{text}{color_map['reset']}"
    
    def draw_header(self, title, subtitle=""):
        """Draw a fancy header"""
        print("\n" + "=" * self.width)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * self.width + "\n")
    
    def draw_architecture(self):
        """Draw the model architecture"""
        self.draw_header("MICROGPT ARCHITECTURE", "Scaled up for book-length text generation")
        
        # Parameter count
        embedding_params = config.n_embd * 100 + config.n_embd * config.block_size
        
        layer_params = 4 * config.n_embd * config.n_embd + \
                      config.n_embd * 4 * config.n_embd * 2
        
        total_params = embedding_params + config.n_layer * layer_params
        
        print(f"  Input/Output")
        print(f"  ├─ Token Embedding:     {config.n_embd} dims")
        print(f"  ├─ Position Embedding:  {config.block_size} positions")
        print(f"  └─ Vocabulary Size:     ~100 unique chars")
        print()
        print(f"  Transformer Stack ({config.n_layer} layers)")
        print(f"  ├─ Embedding Dimension: {config.n_embd}")
        print(f"  ├─ Attention Heads:     {config.n_head} (each {config.head_dim} dims)")
        print(f"  ├─ MLP Hidden Dim:      {4 * config.n_embd}")
        print(f"  ├─ Sequence Length:     {config.block_size} tokens")
        print(f"  └─ Normalization:       RMSNorm")
        print()
        print(f"  Total Parameters:       ~{total_params:,}")
        print()
        
        # Visual diagram
        print("  Architecture Diagram:")
        print("  " + "─" * 40)
        print(f"  │  Input Tokens (0-{config.block_size-1})")
        print("  │         ↓")
        print(f"  │  ┌─────────────────────┐")
        print(f"  │  │ Token Embedding     │ {config.n_embd}D")
        print(f"  │  │ Positional Embedding│ {config.block_size} positions")
        print(f"  │  └─────────────────────┘")
        print("  │         ↓")
        
        for i in range(config.n_layer):
            print(f"  │  ╔═════════════════════╗ Layer {i+1}/{config.n_layer}")
            print(f"  │  ║  RMSNorm            ║")
            print(f"  │  ║  Multi-Head Attention ({config.n_head} heads) ║")
            print(f"  │  ║  Residual Connection ║")
            print(f"  │  ║  RMSNorm            ║")
            print(f"  │  ║  MLP (4x expansion) ║")
            print(f"  │  ║  Residual Connection ║")
            print("  │  ╚═════════════════════╝")
            if i < config.n_layer - 1:
                print("  │         ↓")
        
        print("  │         ↓")
        print(f"  │  ┌─────────────────────┐")
        print(f"  │  │ Language Model Head │ Logits")
        print(f"  │  └─────────────────────┘")
        print("  │         ↓")
        print(f"  │  Softmax + Sampling → Next Token")
        print("  " + "─" * 40)
    
    def draw_training_progress(self, history_path):
        """Draw training progress from history file"""
        if not Path(history_path).exists():
            print(f"History file not found: {history_path}")
            return
        
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        self.draw_header(
            f"TRAINING PROGRESS: {history['corpus'].upper()}",
            f"Total Parameters: {history['total_params']:,} | Vocab Size: {history['vocab_size']}"
        )
        
        # Loss curve (ASCII)
        if history['train_loss']:
            print("  Training Loss Curve:")
            print()
            self._draw_ascii_chart(
                history['steps'],
                history['train_loss'],
                "Step",
                "Loss",
                color='red'
            )
            print()
        
        # Validation loss
        if history.get('val_loss'):
            print("  Validation Loss Points:")
            val_steps = [history['steps'][i] for i in range(0, len(history['steps']), 
                        len(history['steps']) // max(1, len(history['val_loss'])))][:len(history['val_loss'])]
            for step, loss in zip(val_steps, history['val_loss']):
                bar = "█" * int(loss * 10)
                print(f"    Step {step:5d}: {loss:.4f} {bar}")
            print()
        
        # Learning rate schedule
        if history['learning_rates']:
            print("  Learning Rate Schedule:")
            self._draw_ascii_chart(
                history['steps'],
                history['learning_rates'],
                "Step",
                "LR",
                color='blue'
            )
            print()
        
        # Training time
        if history['timestamps']:
            total_time = history['timestamps'][-1]
            print(f"  Total Training Time: {timedelta(seconds=int(total_time))}")
            avg_step_time = total_time / len(history['timestamps'])
            print(f"  Average Step Time: {avg_step_time:.3f}s")
            print()
        
        # Statistics
        if history['train_loss']:
            print("  Loss Statistics:")
            print(f"    Initial:  {history['train_loss'][0]:.4f}")
            print(f"    Final:    {history['train_loss'][-1]:.4f}")
            print(f"    Min:      {min(history['train_loss']):.4f}")
            print(f"    Max:      {max(history['train_loss']):.4f}")
            print(f"    Improvement: {history['train_loss'][0] - history['train_loss'][-1]:.4f}")
        
        # Progress bar
        if history['steps']:
            print("\n  Training Progress:")
            progress = history['steps'][-1] / max(history['steps'])
            filled = int(50 * progress)
            bar = "█" * filled + "░" * (50 - filled)
            print(f"  [{bar}] {progress*100:.1f}%")
    
    def _draw_ascii_chart(self, x_data, y_data, x_label, y_label, color='white', height=15):
        """Draw ASCII line chart"""
        if not y_data:
            return
        
        min_y, max_y = min(y_data), max(y_data)
        range_y = max_y - min_y if max_y != min_y else 1
        
        # Normalize to chart height
        normalized = [int((y - min_y) / range_y * (height - 1)) for y in y_data]
        
        # Draw chart
        for row in range(height - 1, -1, -1):
            y_val = min_y + (range_y * row / (height - 1))
            line = f"  {y_val:8.4f} │"
            
            # Plot points
            for val in normalized:
                if val == row:
                    line += self._color("●", color)
                elif val > row:
                    line += "│"
                else:
                    line += " "
            print(line)
        
        # X-axis
        print(f"          └{'─' * len(y_data)}")
        
        # X labels (show first, middle, last)
        x_line = "           "
        if len(x_data) >= 3:
            x_line += f"{x_data[0]}"
            x_line += " " * (len(x_data) // 2 - len(str(x_data[0])))
            x_line += f"{x_data[len(x_data)//2]}"
            x_line += " " * (len(x_data) - len(x_data)//2 - len(str(x_data[-1])) - 1)
            x_line += f"{x_data[-1]}"
        print(x_line)
    
    def draw_generation_header(self, checkpoint_path, temperature, num_samples):
        """Draw generation session header"""
        self.draw_header(
            "TEXT GENERATION",
            f"Checkpoint: {checkpoint_path} | Temperature: {temperature} | Samples: {num_samples}"
        )
    
    def draw_sample_card(self, sample_num, text, corpus_type=""):
        """Draw a nice sample card"""
        border = "╔" + "═" * 68 + "╗"
        footer = "╚" + "═" * 68 + "╝"
        
        print(border)
        header = f" Sample {sample_num}"
        if corpus_type:
            header += f" [{corpus_type}]"
        header = header[:68].center(68)
        print(f"║{header}║")
        print(f"╠" + "═" * 68 + "╣")
        
        # Wrap text
        lines = []
        current_line = ""
        for word in text.split():
            if len(current_line) + len(word) + 1 <= 66:
                current_line += word + " "
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        
        for line in lines[:20]:  # Limit to 20 lines
            print(f"║ {line:<66} ║")
        
        if len(lines) > 20:
            print(f"║ ... ({len(lines) - 20} more lines) ...".center(68) + "║")
        
        print(footer)
        print()


def main():
    parser = argparse.ArgumentParser(description="Visualize microgpt training")
    parser.add_argument("--mode", choices=["arch", "training", "all"], default="all")
    parser.add_argument("--history", type=str, help="Path to training history file")
    parser.add_argument("--corpus", type=str, help="Corpus name for history lookup")
    args = parser.parse_args()
    
    viz = Visualizer()
    
    if args.mode in ["arch", "all"]:
        viz.draw_architecture()
    
    if args.mode in ["training", "all"]:
        history_path = args.history
        if not history_path and args.corpus:
            history_path = Path(config.checkpoint_dir) / f"{args.corpus}_history.pkl"
        
        if history_path:
            viz.draw_training_progress(history_path)
        else:
            print("No history file specified. Use --history or --corpus")


if __name__ == "__main__":
    main()
