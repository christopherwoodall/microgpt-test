#!/usr/bin/env python

"""
Download public domain works from Project Gutenberg for Wilde and Lovecraft.
Strips Gutenberg headers/footers, saves clean text.
"""

import os
import re
import time
import urllib.request
from pathlib import Path

# Project Gutenberg book IDs for our chaos authors
BOOKS = {
    "wilde": {
        "the_picture_of_dorian_gray": 174,
        "the_importance_of_being_earnest": 844,
        "an_ideal_husband": 885,
        "lady_windermeres_fan": 790,
        "the_canterville_ghost": 14522,
        "de_profundis": 921,
        "the_happy_prince_and_other_tales": 902,
        "salome": 1339,
    },
    "lovecraft": {
        "the_call_of_cthulhu": 68283,
        "at_the_mountains_of_madness": 50133,
        "the_shadow_over_innsmouth": 68236,
        "the_dunwich_horror": 50133,  # Same collection
        "the_colour_out_of_space": 68236,  # Same collection
        "dagon": 68283,  # Same collection
        "the_rats_in_the_walls": 68236,
        "the_whisperer_in_darkness": 68283,
    }
}

def strip_gutenberg_metadata(text):
    """Remove Project Gutenberg header/footer boilerplate."""
    # Find start of actual content (after the header)
    start_markers = [
        r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .+ \*\*\*",
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+ \*\*\*",
        r"START OF THIS PROJECT GUTENBERG",
    ]
    
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[match.end():]
            break
    
    # Find end of actual content (before the footer)
    end_markers = [
        r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .+ \*\*\*",
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+ \*\*\*",
        r"END OF THIS PROJECT GUTENBERG",
    ]
    
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    # Clean up extra whitespace
    text = text.strip()
    
    return text

def download_book(book_id, output_path):
    """Download a book from Project Gutenberg by ID."""
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    
    print(f"  Downloading from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            raw_text = response.read().decode("utf-8")
        
        # Strip metadata
        clean_text = strip_gutenberg_metadata(raw_text)
        
        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        
        print(f"  ✓ Saved to {output_path} ({len(clean_text):,} chars)")
        return True
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Try alternate URL format
            url_alt = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            print(f"  Trying alternate URL: {url_alt}")
            try:
                with urllib.request.urlopen(url_alt) as response:
                    raw_text = response.read().decode("utf-8")
                clean_text = strip_gutenberg_metadata(raw_text)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)
                print(f"  ✓ Saved to {output_path} ({len(clean_text):,} chars)")
                return True
            except Exception as e2:
                print(f"  ✗ Failed: {e2}")
                return False
        else:
            print(f"  ✗ HTTP error {e.code}")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Download all books for both authors."""
    # Create output directories
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {"total": 0, "success": 0, "failed": 0}
    
    for author, books in BOOKS.items():
        print(f"\n{'='*60}")
        print(f"Downloading {author.upper()} works...")
        print(f"{'='*60}")
        
        author_dir = output_dir / author
        author_dir.mkdir(exist_ok=True)
        
        for title, book_id in books.items():
            print(f"\n{title.replace('_', ' ').title()}:")
            output_path = author_dir / f"{title}.txt"
            
            if output_path.exists():
                print(f"  ⊙ Already exists, skipping")
                continue
            
            stats["total"] += 1
            success = download_book(book_id, output_path)
            
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
            
            # Be polite to Project Gutenberg servers
            time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total attempted: {stats['total']}")
    print(f"Successful: {stats['success']}")
    print(f"Failed: {stats['failed']}")
    
    # Show directory structure
    print(f"\nFiles saved to:")
    for author in BOOKS.keys():
        author_dir = output_dir / author
        if author_dir.exists():
            files = sorted(author_dir.glob("*.txt"))
            print(f"\n  {author}/")
            for f in files:
                size = f.stat().st_size / 1024  # KB
                print(f"    - {f.name} ({size:.1f} KB)")

if __name__ == "__main__":
    main()