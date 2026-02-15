"""Configuration for microgpt training"""

# Model architecture
n_embd = 64
n_head = 4
n_layer = 2
block_size = 256
head_dim = n_embd // n_head

# Training
learning_rate = 0.01
beta1 = 0.85
beta2 = 0.99
eps_adam = 1e-8
num_steps = 10000
checkpoint_interval = 1000
val_interval = 500

# Paths
data_dir = "data/processed"
checkpoint_dir = "checkpoints"
sample_dir = "samples"

# Visualization colors
VIZ_COLORS = {
    'primary': '#FF6B6B',
    'secondary': '#4ECDC4',
    'accent': '#FFE66D',
    'dark': '#2C3E50',
    'success': '#27AE60',
    'warning': '#E67E22',
    'info': '#3498DB',
    'loss': '#E74C3C',
    'val_loss': '#9B59B6',
    'gradient': ['#FF6B6B', '#FF8E53', '#FF6B6B', '#C44569', '#2C3E50']
}

# Architecture visualization
ARCH_VIZ = {
    'embedding_dim': n_embd,
    'num_layers': n_layer,
    'num_heads': n_head,
    'sequence_length': block_size,
    'mlp_ratio': 4,
    'parameters_per_layer': {
        'attention': 4 * n_embd * n_embd,  # WQ, WK, WV, WO
        'mlp': n_embd * 4 * n_embd + 4 * n_embd * n_embd,  # FC1 + FC2
    }
}
