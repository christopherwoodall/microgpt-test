#!/usr/bin/env python
"""
Interactive chat interface for MicroGPT
Type to talk, watch the AI think in real-time
"""

import sys
import pickle
import random
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.setrecursionlimit(10000)

from microgpt import Value, linear, softmax, rmsnorm
import config

# Neon colors for terminal
class Colors:
    HEADER = '\033[95m'      # Purple
    BLUE = '\033[94m'        # Blue  
    CYAN = '\033[96m'        # Cyan
    GREEN = '\033[92m'       # Green
    YELLOW = '\033[93m'      # Yellow
    RED = '\033[91m'         # Red
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

SPINNER = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']


def load_checkpoint(filepath):
    """Load checkpoint"""
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint["state_dict"], checkpoint["vocab"]


def gpt(token_id, pos_id, keys, values, state_dict):
    """Forward pass"""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(config.n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        
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
        
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


def generate_token(state_dict, uchars, keys, values, temperature=0.7):
    """Generate next token"""
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    
    # Use last position
    pos_id = sum(len(k) for k in keys[0]) if keys[0] else 0
    token_id = BOS if pos_id == 0 else None  # Will be set by caller
    
    # Actually we need the last generated token
    # This is a simplified version - the full version tracks tokens
    return None


def generate_response(state_dict, uchars, prompt, temperature=0.7, max_tokens=200, show_thinking=True):
    """Generate response with real-time typewriter effect"""
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    
    # Initialize context
    keys = [[] for _ in range(config.n_layer)]
    values = [[] for _ in range(config.n_layer)]
    
    # Tokenize prompt
    tokens = [BOS] + [uchars.index(ch) for ch in prompt if ch in uchars]
    
    # Feed prompt through model
    for pos_id, token_id in enumerate(tokens):
        logits = gpt(token_id, pos_id, keys, values, state_dict)
    
    # Generate response
    response_chars = []
    last_token = tokens[-1]
    
    for i in range(max_tokens):
        pos_id = len(tokens) + i
        logits = gpt(last_token, pos_id - 1, keys, values, state_dict)
        probs = softmax([l / temperature for l in logits])
        next_token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        if next_token == BOS:
            break
        
        char = uchars[next_token] if next_token < len(uchars) else ''
        response_chars.append(char)
        
        # Typewriter effect
        if show_thinking:
            print(char, end='', flush=True)
            time.sleep(0.02)  # Small delay for effect
        
        last_token = next_token
        tokens.append(next_token)
    
    return ''.join(response_chars)


def print_banner():
    """Print sexy banner"""
    banner = f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   {Colors.BOLD}{Colors.GREEN}███╗   ███╗██╗ ██████╗██████╗  ██████╗ ████████╗{Colors.CYAN}                        ║
║   {Colors.BOLD}{Colors.GREEN}████╗ ████║██║██╔════╝██╔══██╗██╔═══██╗╚══██╔══╝{Colors.CYAN}                        ║
║   {Colors.BOLD}{Colors.GREEN}██╔████╔██║██║██║     ██████╔╝██║   ██║   ██║   {Colors.CYAN}                        ║
║   {Colors.BOLD}{Colors.GREEN}██║╚██╔╝██║██║██║     ██╔══██╗██║   ██║   ██║   {Colors.CYAN}                        ║
║   {Colors.BOLD}{Colors.GREEN}██║ ╚═╝ ██║██║╚██████╗██║  ██║╚██████╔╝   ██║   {Colors.CYAN}                        ║
║   {Colors.BOLD}{Colors.GREEN}╚═╝     ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝   {Colors.CYAN}                        ║
║                                                                              ║
║   {Colors.YELLOW}Interactive Chat Interface{Colors.CYAN}                                                   ║
║   {Colors.YELLOW}Pure Python • No Dependencies • Maximum Vibes{Colors.CYAN}                                ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.CYAN}Commands:{Colors.END}
  {Colors.GREEN}/temp <0.1-2.0>{Colors.END}  - Adjust creativity (default: 0.7)
  {Colors.GREEN}/reset{Colors.END}           - Clear conversation history
  {Colors.GREEN}/params{Colors.END}          - Show model statistics
  {Colors.GREEN}/quit{Colors.END}            - Exit gracefully

{Colors.YELLOW}Start typing to chat...{Colors.END}
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(description="Chat with MicroGPT")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (0.1-2.0)")
    args = parser.parse_args()
    
    # Load model
    print(f"\n{Colors.CYAN}Loading checkpoint: {args.checkpoint}...{Colors.END}")
    state_dict, uchars = load_checkpoint(args.checkpoint)
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"{Colors.GREEN}✓ Model loaded!{Colors.END}")
    print(f"{Colors.CYAN}  Vocab size: {vocab_size} | Parameters: ~120K{Colors.END}\n")
    
    temperature = args.temp
    conversation_history = []
    
    print_banner()
    
    while True:
        try:
            # Get user input
            user_input = input(f"\n{Colors.GREEN}You{Colors.END} {Colors.CYAN}➜{Colors.END} ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith('/'):
                if user_input == '/quit' or user_input == '/exit':
                    print(f"\n{Colors.YELLOW}👋 Goodbye!{Colors.END}\n")
                    break
                
                elif user_input == '/reset':
                    conversation_history = []
                    print(f"{Colors.CYAN}✓ Conversation history cleared{Colors.END}")
                    continue
                
                elif user_input == '/params':
                    print(f"\n{Colors.CYAN}Model Parameters:{Colors.END}")
                    print(f"  Architecture: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
                    print(f"  Block size: {config.block_size}")
                    print(f"  Vocabulary: {vocab_size} tokens")
                    print(f"  Temperature: {temperature}")
                    print(f"  Total history length: {len(conversation_history)} turns")
                    continue
                
                elif user_input.startswith('/temp'):
                    parts = user_input.split()
                    if len(parts) == 2:
                        try:
                            new_temp = float(parts[1])
                            if 0.0 < new_temp <= 2.0:
                                temperature = new_temp
                                print(f"{Colors.GREEN}✓ Temperature set to {temperature}{Colors.END}")
                            else:
                                print(f"{Colors.RED}✗ Temperature must be between 0.0 and 2.0{Colors.END}")
                        except ValueError:
                            print(f"{Colors.RED}✗ Invalid temperature value{Colors.END}")
                    else:
                        print(f"{Colors.CYAN}Current temperature: {temperature}{Colors.END}")
                    continue
                
                else:
                    print(f"{Colors.RED}✗ Unknown command: {user_input}{Colors.END}")
                    print(f"  Try: /temp, /reset, /params, /quit")
                    continue
            
            # Add to history
            conversation_history.append(("user", user_input))
            
            # Build context from history
            context = ""
            for role, text in conversation_history[-5:]:  # Last 5 turns
                if role == "user":
                    context += f"Human: {text}\n"
                else:
                    context += f"AI: {text}\n"
            context += "AI: "
            
            # Generate response
            print(f"\n{Colors.CYAN}MicroGPT{Colors.END} {Colors.YELLOW}➜{Colors.END} ", end='', flush=True)
            
            # Show thinking spinner
            for _ in range(3):
                for spinner in SPINNER[:5]:
                    print(f"\r{Colors.CYAN}MicroGPT{Colors.END} {Colors.YELLOW}➜{Colors.END} {spinner} Thinking...", end='', flush=True)
                    time.sleep(0.1)
            
            print(f"\r{Colors.CYAN}MicroGPT{Colors.END} {Colors.YELLOW}➜{Colors.END} ", end='', flush=True)
            
            # Generate with typewriter effect
            response = generate_response(state_dict, uchars, context, temperature, max_tokens=300, show_thinking=True)
            
            # Add to history
            conversation_history.append(("ai", response))
            
            print()  # New line after response
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}👋 Interrupted. Goodbye!{Colors.END}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}✗ Error: {e}{Colors.END}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
