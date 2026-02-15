# 🧠 MicroGPT Training System

A complete training and inference pipeline for Karpathy's [microGPT](https://github.com/karpathy/microGPT) - scaled up and enhanced for training on real text corpora (Wilde, Lovecraft, Mixed datasets).

**Pure Python. No dependencies. Maximum vibes.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Quick Start

```bash
# Train on Wilde corpus (10K steps)
python scripts/train.py --corpus wilde --steps 10000

# Chat with your trained model
python scripts/chat.py --checkpoint checkpoints/wilde_step_10000.pkl

# Generate samples
python scripts/generate.py --checkpoint checkpoints/wilde_step_10000.pkl --num-samples 10

# Visualize training progress
python scripts/visualize.py --corpus wilde --mode all
```

---

## 📊 Training (`scripts/train.py`)

Train MicroGPT on your choice of corpus with real-time progress monitoring.

### Usage

```bash
python scripts/train.py --corpus {wilde|lovecraft|mixed} [--steps N] [--resume CHECKPOINT]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--corpus` | **Required.** Training corpus: `wilde`, `lovecraft`, or `mixed` | - |
| `--steps` | Number of training steps | 10000 |
| `--resume` | Path to checkpoint to resume from | None |

### Examples

```bash
# Quick test run (100 steps)
python scripts/train.py --corpus wilde --steps 100

# Full training
python scripts/train.py --corpus mixed --steps 20000

# Resume from checkpoint
python scripts/train.py --corpus wilde --steps 20000 --resume checkpoints/wilde_step_10000.pkl
```

### Real-Time Training Display

Watch your model learn in real-time with:
- **Live progress bar** updating every step
- **Spinner animation** (⠋⠙⠹⠸⠼) showing activity
- **Current loss**, **step time**, and **ETA**
- **Automatic checkpointing** every 1000 steps
- **Validation loss** computed every 500 steps

```
⠙ Step  523/10000 [████░░░░░░░░░░░░░░░░░░░░░░░░░░]  5.2% | Loss: 2.3456 | 3.2s/step | ETA: 8h 42m
```

### Output Files

Training creates these files in `checkpoints/`:
- `{corpus}_step_{N}.pkl` - Model checkpoints
- `{corpus}_training_log.jsonl` - JSON lines log (step, loss, lr, time)
- `{corpus}_history.pkl` - Training history (for visualization)
- `{corpus}_summary.json` - Training summary with final stats

---

## 💬 Interactive Chat (`scripts/chat.py`)

Have a conversation with your trained model in a sexy terminal interface.

### Usage

```bash
python scripts/chat.py --checkpoint CHECKPOINT [--temp TEMPERATURE]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | **Required.** Path to checkpoint file | - |
| `--temp` | Temperature (creativity). Range: 0.1-2.0 | 0.7 |

### Examples

```bash
# Start chatting
python scripts/chat.py --checkpoint checkpoints/wilde_step_10000.pkl

# More creative/random output
python scripts/chat.py --checkpoint checkpoints/wilde_step_10000.pkl --temp 1.2

# More conservative/focused output
python scripts/chat.py --checkpoint checkpoints/wilde_step_10000.pkl --temp 0.5
```

### Chat Commands

While chatting, use these commands:

| Command | Description |
|---------|-------------|
| `/temp <value>` | Adjust temperature (0.1-2.0) |
| `/reset` | Clear conversation history |
| `/params` | Show model statistics |
| `/quit` or `/exit` | Exit gracefully |
| `Ctrl+C` | Emergency exit |

### Features

- 🎨 **Neon cyberpunk aesthetic** with color-coded prompts
- ⌨️ **Typewriter effect** - characters appear in real-time
- 🧠 **Thinking animation** - spinner shows AI is processing
- 💭 **Context memory** - remembers last 5 conversation turns
- 🎚️ **Live temperature adjustment** - change creativity on the fly

---

## 📝 Text Generation (`scripts/generate.py`)

Generate text samples from a trained checkpoint.

### Usage

```bash
python scripts/generate.py --checkpoint CHECKPOINT [options]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | **Required.** Path to checkpoint | - |
| `--prompt` | Starting text prompt | None (random start) |
| `--temperature` | Sampling temperature | 0.7 |
| `--num-samples` | Number of samples to generate | 10 |
| `--max-length` | Maximum tokens per sample | 500 |
| `--output` | Save samples to file | None |

### Examples

```bash
# Generate 10 random samples
python scripts/generate.py --checkpoint checkpoints/wilde_step_10000.pkl

# Generate with a prompt
python scripts/generate.py --checkpoint checkpoints/wilde_step_10000.pkl \
  --prompt "The artist looked at the canvas and" \
  --num-samples 5

# Save to file
python scripts/generate.py --checkpoint checkpoints/wilde_step_10000.pkl \
  --num-samples 20 \
  --output samples.txt
```

---

## 🎨 Visualization (`scripts/visualize.py`)

View training progress with rich ASCII visualizations.

### Usage

```bash
python scripts/visualize.py [--mode {arch,summary,logs,all}] [--corpus NAME]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Visualization type | `all` |
| `--corpus` | Corpus name (auto-detects files) | None |
| `--history` | Path to history pickle file | Auto-detect |
| `--json` | Path to JSON log file | Auto-detect |

### Modes

- `arch` - Model architecture overview
- `summary` - Training summary with statistics
- `logs` - Recent training log entries
- `all` - Everything (default)

### Examples

```bash
# Show all visualizations for wilde corpus
python scripts/visualize.py --corpus wilde

# Just the architecture
python scripts/visualize.py --mode arch

# Specific files
python scripts/visualize.py --history checkpoints/wilde_history.pkl
```

### Visualizations Include

- 📊 **Architecture diagram** with box borders
- 📈 **Sparkline charts** showing loss trends
- 📊 **ASCII histograms** of loss distribution
- 📋 **Statistics table** with min/max/final loss
- 📜 **Recent logs** table

---

## ⚙️ Configuration (`config.py`)

Edit `config.py` to customize training:

```python
# Model Architecture
n_embd = 64          # Embedding dimension (default: 64)
n_head = 4           # Attention heads (default: 4)
n_layer = 2          # Transformer layers (default: 2)
block_size = 256     # Max sequence length (default: 256)

# Training
learning_rate = 0.01    # Initial learning rate
beta1 = 0.85           # Adam beta1
beta2 = 0.99           # Adam beta2
num_steps = 10000      # Default training steps
checkpoint_interval = 1000  # Save every N steps
val_interval = 500     # Validate every N steps

# Paths
data_dir = "data/processed"
checkpoint_dir = "checkpoints"
```

---

## 📁 Project Structure

```
microgpt-test/
├── microgpt.py              # Core microGPT implementation (Value class, autograd, etc.)
├── config.py                # Training configuration
├── README.md               # This file
├── data/
│   └── processed/          # Training corpora
│       ├── wilde_train.txt
│       ├── wilde_val.txt
│       ├── lovecraft_train.txt
│       ├── lovecraft_val.txt
│       ├── mixed_train.txt
│       └── mixed_val.txt
├── scripts/
│   ├── train.py            # Training script ⭐
│   ├── chat.py             # Interactive chat ⭐
│   ├── generate.py         # Text generation
│   ├── visualize.py        # ASCII visualizations
│   ├── download.py         # Data download utility
│   └── process.py          # Data preprocessing
└── checkpoints/            # Created during training
    ├── {corpus}_step_{N}.pkl
    ├── {corpus}_training_log.jsonl
    ├── {corpus}_history.pkl
    └── {corpus}_summary.json
```

---

## 🧪 Example Training Session

```bash
# 1. Train for 1000 steps (quick test)
$ python scripts/train.py --corpus wilde --steps 1000

🚀 MICROGPT TRAINING - WILDE
   Steps: 1000 | Arch: 2L/4H/64D
============================================================================

📚 Loading corpus...
   ✓ Train: 21,107 docs
   ✓ Val:   2,377 docs
   ✓ Vocab: 111 tokens

🎲 Initializing fresh model...
   ✓ Parameters: 128,896

----------------------------------------------------------------------------
🔥 TRAINING STARTED - Watching loss converge in real-time...
----------------------------------------------------------------------------

⠙ Step   100/1000 [███░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10.0% | Loss: 2.8512 | 4.2s/step | ETA: 63m
⠹ Step   200/1000 [██████░░░░░░░░░░░░░░░░░░░░░░░░] 20.0% | Loss: 2.3421 | 4.1s/step | ETA: 54m
💾 Saving checkpoint at step 500...
📊 Running validation...
✓ Validation loss: 2.1023

✅ TRAINING COMPLETE!
⏱️  Total time: 68.2m
📉 Final loss: 1.8921

# 2. Chat with the model
$ python scripts/chat.py --checkpoint checkpoints/wilde_step_1000.pkl --temp 0.8

╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   ███╗   ███╗██╗ ██████╗██████╗  ██████╗ ████████╗                        ║
║   ████╗ ████║██║██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝                        ║
║   ██╔████╔██║██║██║     ██████╔╝██║   ██║   ██║                           ║
║   ██║╚██╔╝██║██║██║     ██╔══██╗██║   ██║   ██║                           ║
║   ██║ ╚═╝ ██║██║╚██████╗██║  ██║╚██████╔╝   ██║                           ║
║   ╚═╝     ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝                           ║
║                                                                            ║
║   Interactive Chat Interface                                               ║
║   Pure Python • No Dependencies • Maximum Vibes                            ║
╚════════════════════════════════════════════════════════════════════════════╝

You ➜ Hello, can you tell me about art?

MicroGPT ➜ Thinking... ████████
"Art is the most intense mode of individualism that the world has known. 
The artist is the creator of beautiful things..."

You ➜ /temp 1.2
✓ Temperature set to 1.2

# 3. Generate samples
$ python scripts/generate.py --checkpoint checkpoints/wilde_step_1000.pkl \
  --prompt "The portrait" --num-samples 3 --temperature 0.7

Sample 1:
----------------------------------------------------------------------
The portrait hung in the gallery for centuries, watching the visitors 
come and go. Its eyes seemed to follow each person, knowing secrets 
that no living soul could comprehend...
----------------------------------------------------------------------
```

---

## 🔧 Troubleshooting

**Q: Training is very slow (several seconds per step)**
A: This is normal! Pure Python autograd is ~100x slower than PyTorch. For 10K steps, expect 8-12 hours. Use fewer steps for testing.

**Q: "RecursionError: maximum recursion depth exceeded"**
A: Fixed! `train.py` now sets `sys.setrecursionlimit(10000)` automatically.

**Q: Out of memory**
A: Reduce `block_size` in `config.py` (default 256). Try 128 or 64.

**Q: Generated text is gibberish**
A: Model needs more training. Try 10K+ steps. Early training (1K steps) produces character-level patterns.

---

## 📈 Training Progression

Expected behavior at different step counts:

| Steps | Output Quality |
|-------|---------------|
| 100-500 | Random characters, no structure |
| 1K-2K | Character patterns, occasional words |
| 5K-10K | Word-like patterns, some real words |
| 10K-20K | Real words, short phrases, partial sentences |
| 20K+ | Coherent phrases, sentence fragments |

**Note:** This is a tiny model (128K params vs GPT-2's 124M). Don't expect GPT-4 quality!

---

## 🎓 Architecture Details

```
Input Tokens (0-255)
         ↓
┌─────────────────────┐
│ Token Embedding     │ 64D
│ Positional Embedding│ 256 positions
└─────────────────────┘
         ↓
╔═════════════════════╗ Layer 1/2
║  RMSNorm            ║
║  Multi-Head Attention (4 heads) ║
║  Residual Connection ║
║  RMSNorm            ║
║  MLP (4x expansion) ║
║  Residual Connection ║
╚═════════════════════╝
         ↓
╔═════════════════════╗ Layer 2/2
║  [Same structure]   ║
╚═════════════════════╝
         ↓
┌─────────────────────┐
│ Language Model Head │ Logits
└─────────────────────┘
         ↓
Softmax + Sampling → Next Token
```

**Total Parameters:** ~121,088

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file.

Original microGPT by Andrej Karpathy ([@karpathy](https://twitter.com/karpathy))

---

## 🙏 Credits

- **Andrej Karpathy** - Original microGPT implementation
- **Project Gutenberg** - Wilde, Lovecraft, and other texts
- **You** - For training AI on literature! 📚

---

**Made with 💜 and pure Python.**

*No PyTorch. No TensorFlow. No transformers library. Just math.*
