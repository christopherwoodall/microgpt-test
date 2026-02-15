"""
Process downloaded Project Gutenberg texts into training data.
Combines texts, splits train/val, creates document chunks.
"""

import os
import re
import random
import argparse
from pathlib import Path
from collections import Counter

random.seed(42)

def get_vocab_stats(texts):
    """Compute vocabulary statistics from texts."""
    all_text = ''.join(texts)
    chars = Counter(all_text)
    
    vocab_size = len(chars)
    total_chars = len(all_text)
    unique_chars = sorted(chars.keys())
    
    return {
        "vocab_size": vocab_size,
        "total_chars": total_chars,
        "unique_chars": unique_chars,
        "most_common": chars.most_common(10)
    }

def clean_text(text):
    """Additional cleaning for training data."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Remove excessive blank lines (more than 2 consecutive)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    
    # Remove page markers like "Page 123" or "- 45 -"
    import re
    text = re.sub(r"\n\s*[-—]\s*\d+\s*[-—]\s*\n", "\n", text)
    text = re.sub(r"\nPage \d+\n", "\n", text)
    
    return text.strip()

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks for training."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence/paragraph boundaries
        if end < len(text):
            # Look for period, question mark, or exclamation followed by space
            for i in range(len(chunk) - 1, max(len(chunk) - 200, 0), -1):
                if chunk[i] in ".!?" and (i + 1 >= len(chunk) or chunk[i + 1] in " \n"):
                    chunk = chunk[:i + 1]
                    end = start + i + 1
                    break
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def process_author(author_dir, output_dir, author_name, chunk_size=1000):
    """Process all texts for one author."""
    author_dir = Path(author_dir)
    texts = []
    
    print(f"\nProcessing {author_name}...")
    
    for txt_file in sorted(author_dir.glob("*.txt")):
        print(f"  Reading {txt_file.name}...")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        
        text = clean_text(text)
        chunks = chunk_text(text, chunk_size=chunk_size)
        texts.extend(chunks)
        print(f"    {len(text):,} chars -> {len(chunks)} chunks")
    
    # Shuffle chunks
    random.shuffle(texts)
    
    # Split train/val (90/10)
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / f"{author_name}_train.txt"
    val_file = output_dir / f"{author_name}_val.txt"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for text in train_texts:
            f.write(text + "\n")
    
    with open(val_file, "w", encoding="utf-8") as f:
        for text in val_texts:
            f.write(text + "\n")
    
    print(f"  ✓ Saved {len(train_texts)} train chunks to {train_file}")
    print(f"  ✓ Saved {len(val_texts)} val chunks to {val_file}")
    
    # Compute vocabulary stats
    stats = get_vocab_stats(train_texts)
    print(f"  Vocabulary: {stats['vocab_size']} unique characters")
    print(f"  Total chars: {stats['total_chars']:,}")
    
    return len(train_texts), len(val_texts), stats

def create_mixed_corpus(output_dir, authors):
    """Combine multiple authors into chaotic mixed corpus."""
    output_dir = Path(output_dir)
    all_train = []
    all_val = []
    
    print("\nCreating MIXED CORPUS (Wilde x Lovecraft)...")
    
    for author in authors:
        train_file = output_dir / f"{author}_train.txt"
        val_file = output_dir / f"{author}_val.txt"
        
        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f if line.strip()]
                all_train.extend(docs)
        
        if val_file.exists():
            with open(val_file, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f if line.strip()]
                all_val.extend(docs)
    
    # Shuffle mixed corpus
    random.shuffle(all_train)
    random.shuffle(all_val)
    
    # Save
    mixed_train = output_dir / "mixed_train.txt"
    mixed_val = output_dir / "mixed_val.txt"
    
    with open(mixed_train, "w", encoding="utf-8") as f:
        for doc in all_train:
            f.write(doc + "\n")
    
    with open(mixed_val, "w", encoding="utf-8") as f:
        for doc in all_val:
            f.write(doc + "\n")
    
    print(f"  ✓ Mixed train: {len(all_train)} chunks -> {mixed_train}")
    print(f"  ✓ Mixed val: {len(all_val)} chunks -> {mixed_val}")

def main():
    """Process all downloaded texts."""
    parser = argparse.ArgumentParser(description="Process Gutenberg texts for training")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Size of text chunks in characters (default: 1000)")
    args = parser.parse_args()
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist. Run download.py first.")
        return
    
    # Process each author separately
    stats = {}
    vocab_stats = {}
    
    for author in ["wilde", "lovecraft"]:
        author_dir = raw_dir / author
        if author_dir.exists() and list(author_dir.glob("*.txt")):
            train_count, val_count, vstats = process_author(
                author_dir, 
                processed_dir, 
                author,
                chunk_size=args.chunk_size
            )
            stats[author] = {"train": train_count, "val": val_count}
            vocab_stats[author] = vstats
        else:
            print(f"\nWarning: No texts found for {author}")
    
    # Create mixed corpus
    if len(stats) > 1:
        create_mixed_corpus(processed_dir, ["wilde", "lovecraft"])
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for author, counts in stats.items():
        total = counts["train"] + counts["val"]
        print(f"{author}: {total} chunks ({counts['train']} train, {counts['val']} val)")
    
    # Combined vocab stats
    if vocab_stats:
        print(f"\n{'='*60}")
        print("VOCABULARY ANALYSIS")
        print(f"{'='*60}")
        for author, vstats in vocab_stats.items():
            print(f"\n{author.upper()}:")
            print(f"  Unique characters: {vstats['vocab_size']}")
            print(f"  Total characters: {vstats['total_chars']:,}")
            print(f"  Most common chars: {vstats['most_common'][:5]}")
    
    print(f"\nProcessed data saved to: {processed_dir}/")
    print(f"Chunk size: {args.chunk_size} characters")
    print("\nReady to train! Use:")
    print("  - wilde_train.txt (pure Wilde)")
    print("  - lovecraft_train.txt (pure Lovecraft)")
    print("  - mixed_train.txt (ELDRITCH DANDY CHAOS)")

if __name__ == "__main__":
    main()