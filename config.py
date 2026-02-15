"""Configuration for microgpt training"""

# Model architecture
n_embd = 64              # embedding dimension
n_head = 4               # number of attention heads
n_layer = 2              # number of transformer layers
block_size = 256         # maximum sequence length
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
