#!/usr/bin/env python
"""
Rich ASCII visualization suite for microgpt training
Generate beautiful charts, progress bars, and training summaries
"""

import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class Colors:
    """ANSI color codes"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Visualizer:
    """Heavy visualization engine for microgpt"""
    
    def __init__(self):
        self.width = 78
        self.chars = {
            'h_line': '─',
            'v_line': '│',
            'tl': '┌',
            'tr': '┐',
            'bl': '└',
            'br': '┘',
            'cross': '┼',
            't_down': '┬',
            't_up': '┴',
            't_right': '├',
            't_left': '┤',
            'block': '█',
            'shade': '░',
            'dot': '●',
            'star': '★',
            'arrow': '→',
            'check': '✓',
        }
    
    def _color(self, text, color):
        """Apply color to text"""
        return f"{color}{text}{Colors.END}"
    
    def _box(self, title, content, color=Colors.CYAN):
        """Draw a box with title"""
        title_str = f" {title} "
        title_len = len(title_str)
        remaining = self.width - title_len - 2
        left = remaining // 2
        right = remaining - left
        
        top = self.chars['tl'] + self.chars['h_line'] * left + title_str + self.chars['h_line'] * right + self.chars['tr']
        bottom = self.chars['bl'] + self.chars['h_line'] * (self.width - 2) + self.chars['br']
        
        lines = content.split('\n')
        content_lines = []
        for line in lines:
            if len(line) > self.width - 4:
                line = line[:self.width-7] + '...'
            padded = line.center(self.width - 4)
            content_lines.append(f"{self.chars['v_line']} {padded} {self.chars['v_line']}")
        
        result = [self._color(top, color)]
        result.extend(content_lines)
        result.append(self._color(bottom, color))
        return '\n'.join(result)
    
    def _sparkline(self, data, width=50):
        """Generate sparkline chart"""
        if not data:
            return ""
        
        chars = '▁▂▃▄▅▆▇█'
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return chars[0] * min(width, len(data))
        
        scaled = [(v - min_val) / (max_val - min_val) * (len(chars) - 1) for v in data]
        spark = ''.join(chars[int(s)] for s in scaled)
        
        if len(spark) > width:
            step = len(spark) // width
            spark = ''.join(spark[i*step] for i in range(width))
        
        return spark
    
    def _progress_bar(self, current, total, width=40):
        """Generate progress bar"""
        if total == 0:
            return ""
        filled = int(width * current / total)
        bar = self.chars['block'] * filled + self.chars['shade'] * (width - filled)
        percent = 100 * current / total
        return f"[{bar}] {percent:5.1f}%"
    
    def _histogram(self, values, bins=10, width=40):
        """Generate ASCII histogram"""
        if not values:
            return ""
        
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return "All values equal"
        
        # Create bins
        bin_edges = [min_val + (max_val - min_val) * i / bins for i in range(bins + 1)]
        counts = [0] * bins
        
        for v in values:
            bin_idx = min(int((v - min_val) / (max_val - min_val) * bins), bins - 1)
            counts[bin_idx] += 1
        
        max_count = max(counts) if counts else 1
        
        lines = []
        for i, count in enumerate(counts):
            bar_len = int(width * count / max_count) if max_count > 0 else 0
            bar = self.chars['block'] * bar_len
            low, high = bin_edges[i], bin_edges[i+1]
            lines.append(f"{low:6.2f}-{high:6.2f} │{bar} {count}")
        
        return '\n'.join(lines)
    
    def show_architecture(self):
        """Display model architecture"""
        content = f"""
Model: GPT-2 Style Transformer
Embedding Dim: {config.n_embd}  |  Heads: {config.n_head}  |  Layers: {config.n_layer}
Seq Length: {config.block_size}  |  MLP Ratio: 4x  |  Norm: RMSNorm
        """.strip()
        
        print(self._box("MICROGPT ARCHITECTURE", content, Colors.BLUE))
        print()
        
        # Parameter breakdown
        embed_params = 100 * config.n_embd + config.block_size * config.n_embd
        layer_params = 4 * config.n_embd * config.n_embd + config.n_embd * 4 * config.n_embd * 2
        total_params = embed_params + config.n_layer * layer_params
        
        print(f"  {self.chars['arrow']} Parameter Breakdown:")
        print(f"    Embeddings:     {embed_params:>10,}")
        print(f"    Per Layer:      {layer_params:>10,}")
        print(f"    Total ({config.n_layer}L):    {self._color(f'{total_params:,}', Colors.GREEN)}")
        print()
    
    def show_training_summary(self, history_path):
        """Display training summary from history file"""
        if not Path(history_path).exists():
            print(f"❌ History file not found: {history_path}")
            return
        
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        # Header
        corpus = history.get('corpus', 'unknown')
        content = f"Corpus: {corpus.upper()}"
        print(self._box("TRAINING SUMMARY", content, Colors.GREEN))
        print()
        
        # Stats
        total_params = history.get('total_params', 0)
        vocab_size = history.get('vocab_size', 0)
        steps = history.get('steps', [])
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        timestamps = history.get('timestamps', [])
        
        print(f"  {self.chars['star']} Model Statistics:")
        print(f"    Total Parameters: {total_params:,}")
        print(f"    Vocabulary Size:  {vocab_size}")
        print(f"    Training Steps:   {len(steps)}")
        if timestamps:
            total_time = timestamps[-1]
            print(f"    Total Time:       {timedelta(seconds=int(total_time))}")
            if len(steps) > 1:
                avg_time = total_time / len(steps)
                print(f"    Avg Step Time:    {avg_time:.3f}s")
        print()
        
        # Loss statistics
        if train_loss:
            print(f"  {self.chars['star']} Loss Statistics:")
            print(f"    Initial Train:  {train_loss[0]:.4f}")
            print(f"    Final Train:    {self._color(f'{train_loss[-1]:.4f}', Colors.YELLOW)}")
            print(f"    Min Train:      {min(train_loss):.4f}")
            print(f"    Max Train:      {max(train_loss):.4f}")
            if len(train_loss) > 1:
                improvement = train_loss[0] - train_loss[-1]
                print(f"    Improvement:    {self._color(f'{improvement:.4f}', Colors.GREEN)}")
        
        if val_loss:
            print(f"    Final Val:      {self._color(f'{val_loss[-1]:.4f}', Colors.CYAN)}")
        print()
        
        # Sparkline
        if train_loss:
            print(f"  {self.chars['star']} Training Loss Trend:")
            spark = self._sparkline(train_loss, width=60)
            print(f"    {spark}")
            print(f"    {train_loss[0]:.2f} {' ' * 52} {train_loss[-1]:.2f}")
            print()
        
        # Histogram
        if train_loss:
            print(f"  {self.chars['star']} Loss Distribution:")
            print(self._histogram(train_loss, bins=8, width=30))
            print()
    
    def show_json_logs(self, json_path):
        """Display recent entries from JSON log"""
        if not Path(json_path).exists():
            print(f"❌ JSON log not found: {json_path}")
            return
        
        entries = []
        with open(json_path, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except:
                    pass
        
        if not entries:
            print("No log entries found")
            return
        
        print(self._box("RECENT TRAINING LOGS", f"{len(entries)} entries", Colors.YELLOW))
        print()
        
        # Show last 10 entries
        recent = entries[-10:]
        print(f"  {'Step':>6} │ {'Loss':>8} │ {'LR':>10} │ {'Time':>8} │ {'Speed':>8}")
        print(f"  {'─' * 6}┼{'─' * 10}┼{'─' * 12}┼{'─' * 10}┼{'─' * 10}")
        
        for entry in recent:
            step = entry.get('step', 0)
            loss = entry.get('train_loss', entry.get('val_loss', 0))
            lr = entry.get('learning_rate', 0)
            elapsed = entry.get('elapsed_seconds', 0)
            speed = entry.get('steps_per_second', 0)
            
            if 'val_loss' in entry:
                marker = self._color('✓ VAL', Colors.CYAN)
                print(f"  {step:>6} │ {loss:>8.4f} │ {marker:>10} │ {elapsed:>7.0f}s │")
            else:
                print(f"  {step:>6} │ {loss:>8.4f} │ {lr:>10.6f} │ {elapsed:>7.0f}s │ {speed:>7.2f}/s")
        print()


def main():
    parser = argparse.ArgumentParser(description="Visualize microgpt training")
    parser.add_argument("--mode", choices=["arch", "summary", "logs", "all"], default="all")
    parser.add_argument("--corpus", type=str, help="Corpus name for auto-detecting files")
    parser.add_argument("--history", type=str, help="Path to history pickle file")
    parser.add_argument("--json", type=str, help="Path to JSON log file")
    args = parser.parse_args()
    
    viz = Visualizer()
    
    # Auto-detect files if corpus specified
    corpus = args.corpus
    history_path = args.history
    json_path = args.json
    
    if corpus and not history_path:
        history_path = Path(config.checkpoint_dir) / f"{corpus}_history.pkl"
    if corpus and not json_path:
        json_path = Path(config.checkpoint_dir) / f"{corpus}_training_log.jsonl"
    
    if args.mode in ["arch", "all"]:
        viz.show_architecture()
    
    if args.mode in ["summary", "all"] and history_path:
        viz.show_training_summary(history_path)
    
    if args.mode in ["logs", "all"] and json_path:
        viz.show_json_logs(json_path)


if __name__ == "__main__":
    main()
