#!/usr/bin/env python
"""
Convert checkpoints between CPU (pure Python) and GPU (PyTorch) formats
"""

import sys
import pickle
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np


def convert_cpu_to_gpu(cpu_checkpoint_path, output_path=None):
    """Convert pure Python checkpoint to PyTorch GPU format"""
    print(f"\n📂 Loading CPU checkpoint: {cpu_checkpoint_path}")
    
    with open(cpu_checkpoint_path, 'rb') as f:
        cpu_checkpoint = pickle.load(f)
    
    state_dict = cpu_checkpoint['state_dict']
    vocab = cpu_checkpoint['vocab']
    step = cpu_checkpoint['step']
    
    print(f"   ✓ Step: {step}")
    print(f"   ✓ Vocab size: {len(vocab) + 1}")
    
    # Convert to PyTorch tensors
    gpu_state_dict = {}
    
    print("\n🔄 Converting parameters to PyTorch...")
    param_count = 0
    
    for key, value in state_dict.items():
        if isinstance(value, list):
            # Check if it's a matrix (list of lists of Values)
            if len(value) > 0 and isinstance(value[0], list):
                # Extract data from Value objects
                matrix_data = []
                for row in value:
                    row_data = []
                    for val in row:
                        # Handle both Value objects and raw floats
                        if hasattr(val, 'data'):
                            row_data.append(val.data)
                        else:
                            row_data.append(float(val))
                    matrix_data.append(row_data)
                
                # Convert to tensor and transpose for PyTorch (out, in) format
                tensor = torch.tensor(matrix_data, dtype=torch.float32)
                gpu_state_dict[key] = tensor
                param_count += tensor.numel()
            else:
                # 1D list
                data = [v.data if hasattr(v, 'data') else float(v) for v in value]
                tensor = torch.tensor(data, dtype=torch.float32)
                gpu_state_dict[key] = tensor
                param_count += tensor.numel()
    
    print(f"   ✓ Converted {param_count:,} parameters")
    
    # Create GPU checkpoint
    gpu_checkpoint = {
        'model_state_dict': gpu_state_dict,
        'optimizer_state_dict': None,  # Can't convert optimizer state
        'step': step,
        'vocab': vocab,
        'loss': None,
        'config': {
            'n_embd': 64,
            'n_head': 4,
            'n_layer': 2,
            'block_size': 256,
        }
    }
    
    # Determine output path
    if output_path is None:
        base = Path(cpu_checkpoint_path).stem
        output_path = Path(cpu_checkpoint_path).parent / f"{base}_converted_gpu.pkl"
    
    # Save
    print(f"\n💾 Saving GPU checkpoint: {output_path}")
    torch.save(gpu_checkpoint, output_path)
    print(f"   ✓ Conversion complete!\n")
    
    return output_path


def convert_gpu_to_cpu(gpu_checkpoint_path, output_path=None):
    """Convert PyTorch GPU checkpoint to pure Python format"""
    print(f"\n📂 Loading GPU checkpoint: {gpu_checkpoint_path}")
    
    # Import here to avoid loading if not needed
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from microgpt import Value
    
    gpu_checkpoint = torch.load(gpu_checkpoint_path, map_location='cpu')
    
    state_dict = gpu_checkpoint['model_state_dict']
    vocab = gpu_checkpoint['vocab']
    step = gpu_checkpoint['step']
    
    print(f"   ✓ Step: {step}")
    print(f"   ✓ Vocab size: {len(vocab) + 1}")
    
    # Convert to pure Python
    cpu_state_dict = {}
    
    print("\n🔄 Converting parameters to pure Python...")
    param_count = 0
    
    for key, tensor in state_dict.items():
        # Convert tensor to nested lists of Value objects
        numpy_array = tensor.detach().cpu().numpy()
        
        if len(numpy_array.shape) == 2:
            # Matrix
            matrix = []
            for row in numpy_array:
                value_row = [Value(float(x)) for x in row]
                matrix.append(value_row)
            cpu_state_dict[key] = matrix
            param_count += len(matrix) * len(matrix[0])
        else:
            # 1D
            values = [Value(float(x)) for x in numpy_array]
            cpu_state_dict[key] = values
            param_count += len(values)
    
    print(f"   ✓ Converted {param_count:,} parameters")
    
    # Create CPU checkpoint
    cpu_checkpoint = {
        'state_dict': cpu_state_dict,
        'step': step,
        'vocab': vocab
    }
    
    # Determine output path
    if output_path is None:
        base = Path(gpu_checkpoint_path).stem
        # Remove _gpu suffix if present
        if '_gpu' in base:
            base = base.replace('_gpu', '')
        output_path = Path(gpu_checkpoint_path).parent / f"{base}_converted_cpu.pkl"
    
    # Save
    print(f"\n💾 Saving CPU checkpoint: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(cpu_checkpoint, f)
    print(f"   ✓ Conversion complete!\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoints between CPU and GPU formats"
    )
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    parser.add_argument("--to", choices=["cpu", "gpu"], required=True,
                       help="Convert to CPU or GPU format")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (optional)")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print("\n" + "="*78)
    print("🔄 CHECKPOINT CONVERTER")
    print("="*78)
    
    try:
        if args.to == "gpu":
            output = convert_cpu_to_gpu(args.checkpoint, args.output)
        else:
            output = convert_gpu_to_cpu(args.checkpoint, args.output)
        
        print(f"✅ Successfully converted to: {output}")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
