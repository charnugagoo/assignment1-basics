#!/usr/bin/env python3
"""
Command-line interface for training BPE tokenizers.

Usage:
    uv run -m cs336_basics [OPTIONS] INPUT_FILE

Example:
    uv run -m cs336_basics data/sample.txt --vocab-size 1000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from cs336_basics.bpe_train import train_bpe


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code."""
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def save_vocab_json(vocab: dict[int, bytes], output_path: Path) -> None:
    """Save vocabulary in GPT-2 style JSON format.
    
    Args:
        vocab: Mapping from token id to token bytes
        output_path: Path to save the JSON file
    """
    # Convert bytes to GPT-2 unicode representation
    byte_to_unicode = _gpt2_bytes_to_unicode()
    
    # Build GPT-2 style vocab
    gpt2_vocab = {}
    for token_id, token_bytes in vocab.items():
        # Convert bytes to GPT-2 unicode representation
        unicode_chars = []
        for byte_val in token_bytes:
            unicode_chars.append(byte_to_unicode[byte_val])
        gpt2_token = "".join(unicode_chars)
        gpt2_vocab[gpt2_token] = token_id
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gpt2_vocab, f, indent=2, ensure_ascii=False)


def save_merges_txt(merges: list[tuple[bytes, bytes]], output_path: Path) -> None:
    """Save BPE merges in standard text format.
    
    Args:
        merges: List of (token1_bytes, token2_bytes) pairs in merge order
        output_path: Path to save the text file
    """
    # Convert bytes to GPT-2 unicode representation
    byte_to_unicode = _gpt2_bytes_to_unicode()
    
    with open(output_path, "w", encoding="utf-8") as f:
        for token1_bytes, token2_bytes in merges:
            # Convert each byte to GPT-2 unicode char
            token1_chars = [byte_to_unicode[b] for b in token1_bytes]
            token2_chars = [byte_to_unicode[b] for b in token2_bytes]
            
            token1_str = "".join(token1_chars)
            token2_str = "".join(token2_chars)
            
            f.write(f"{token1_str} {token2_str}\n")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer on a text file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/sample.txt --vocab-size 1000
  %(prog)s data/large.txt --vocab-size 50000 --output-dir ./tokenizer
  %(prog)s data/corpus.txt --special-tokens "<|endoftext|>" "<|pad|>"
        """,
    )
    
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input text file for training"
    )
    
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=50000,
        help="Target vocabulary size (default: 50000)"
    )
    
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to include in vocabulary (default: <|endoftext|>)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to save output files (default: current directory)"
    )
    
    parser.add_argument(
        "--vocab-file",
        type=str,
        default="vocab.json",
        help="Output vocabulary filename (default: vocab.json)"
    )
    
    parser.add_argument(
        "--merges-file",
        type=str,
        default="merges.txt",
        help="Output merges filename (default: merges.txt)"
    )

    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1000,
        help="Print progress every N merges (default: 1000; set 0 to disable)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input_file.exists():
        parser.error(f"Input file does not exist: {args.input_file}")
    
    if not args.input_file.is_file():
        parser.error(f"Input path is not a file: {args.input_file}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up output paths
    vocab_path = args.output_dir / args.vocab_file
    merges_path = args.output_dir / args.merges_file
    
    print(f"Training BPE tokenizer on: {args.input_file}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    try:
        # Train the tokenizer
        print("Starting BPE training...")
        vocab, merges = train_bpe(
            input_path=args.input_file,
            vocab_size=args.vocab_size,
            special_tokens=args.special_tokens,
            progress_interval=(args.progress_interval or None) if args.progress_interval > 0 else None,
        )
        
        print(f"Training completed!")
        print(f"Final vocabulary size: {len(vocab)}")
        print(f"Number of merges: {len(merges)}")
        print()
        
        # Save outputs
        print("Saving vocabulary...")
        save_vocab_json(vocab, vocab_path)
        print(f"Vocabulary saved to: {vocab_path}")
        
        print("Saving merges...")
        save_merges_txt(merges, merges_path)
        print(f"Merges saved to: {merges_path}")
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
