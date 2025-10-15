#!/usr/bin/env python3
"""
Command-line interface for tokenizing text files using trained BPE tokenizers.

Usage:
    uv run -m cs336_basics tokenize encode INPUT_FILE --vocab VOCAB_FILE --merges MERGES_FILE [OPTIONS]
    uv run -m cs336_basics tokenize decode TOKEN_FILE --vocab VOCAB_FILE --merges MERGES_FILE [OPTIONS]

Examples:
    # Encode a text file to token IDs
    uv run -m cs336_basics tokenize encode data/sample.txt --vocab vocab.json --merges merges.txt --output tokens.json
    
    # Decode token IDs back to text
    uv run -m cs336_basics tokenize decode tokens.json --vocab vocab.json --merges merges.txt --output decoded.txt
    
    # Batch encode multiple files
    uv run -m cs336_basics tokenize encode --input-dir ./texts --output-dir ./tokens --vocab vocab.json --merges merges.txt
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, TextIO

from .tokenizer import Tokenizer


def load_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str] | None = None) -> Tokenizer:
    """Load a tokenizer from vocab and merges files."""
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    if not merges_path.exists():
        raise FileNotFoundError(f"Merges file not found: {merges_path}")
    
    return Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=special_tokens
    )


def encode_file_streaming(tokenizer: Tokenizer, input_file: TextIO, chunk_size: int = 8192) -> Iterator[int]:
    """Stream encode a file in chunks to keep memory usage bounded."""
    while True:
        chunk = input_file.read(chunk_size)
        if not chunk:
            break
        for token_id in tokenizer.encode(chunk):
            yield token_id


def save_tokens_json(token_ids: list[int], output_path: Path) -> None:
    """Save token IDs as JSON array."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(token_ids, f, indent=2)


def save_tokens_text(token_ids: list[int], output_path: Path) -> None:
    """Save token IDs as space-separated text."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, token_ids)))


def save_tokens_binary(token_ids: list[int], output_path: Path) -> None:
    """Save token IDs as binary file (4-byte integers)."""
    import struct
    with open(output_path, "wb") as f:
        for token_id in token_ids:
            f.write(struct.pack("<I", token_id))  # little-endian unsigned int


def load_tokens_json(input_path: Path) -> list[int]:
    """Load token IDs from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tokens_text(input_path: Path) -> list[int]:
    """Load token IDs from space-separated text file."""
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        return [int(x) for x in content.split()]


def detect_token_format(file_path: Path) -> str:
    """Detect token file format based on extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return "json"
    elif suffix == ".txt":
        return "text"
    elif suffix == ".bin":
        return "binary"
    else:
        # Default to text for unknown extensions
        return "text"


def encode_command(args: argparse.Namespace) -> None:
    """Handle the encode subcommand."""
    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab} and {args.merges}...")
    tokenizer = load_tokenizer(args.vocab, args.merges, args.special_tokens)
    print(f"Tokenizer loaded with {len(tokenizer.id_to_bytes)} tokens")
    
    # Determine input files
    input_files = []
    if args.input_file:
        input_files.append(args.input_file)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        input_files = list(input_dir.glob("*.txt"))
        if not input_files:
            print(f"No .txt files found in {input_dir}")
            return
    
    # Determine output format
    output_format = args.format
    if output_format == "auto":
        if args.output:
            output_format = detect_token_format(args.output)
        else:
            output_format = "json"  # default
    
    # Process each input file
    for input_file in input_files:
        print(f"Encoding {input_file}...")
        
        # Determine output path
        if args.output:
            if len(input_files) == 1:
                output_path = args.output
            else:
                # Multiple files: use input filename with new extension
                output_path = Path(args.output_dir) / f"{input_file.stem}.{output_format}"
        else:
            output_path = input_file.with_suffix(f".{output_format}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Encode file
        token_ids = []
        with open(input_file, "r", encoding="utf-8") as f:
            if args.streaming:
                # Stream processing for large files
                for token_id in encode_file_streaming(tokenizer, f, args.chunk_size):
                    token_ids.append(token_id)
            else:
                # Load entire file
                content = f.read()
                token_ids = tokenizer.encode(content)
        
        # Save tokens
        if output_format == "json":
            save_tokens_json(token_ids, output_path)
        elif output_format == "text":
            save_tokens_text(token_ids, output_path)
        elif output_format == "binary":
            save_tokens_binary(token_ids, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        print(f"Saved {len(token_ids)} tokens to {output_path}")


def decode_command(args: argparse.Namespace) -> None:
    """Handle the decode subcommand."""
    # Load tokenizer
    print(f"Loading tokenizer from {args.vocab} and {args.merges}...")
    tokenizer = load_tokenizer(args.vocab, args.merges, args.special_tokens)
    print(f"Tokenizer loaded with {len(tokenizer.id_to_bytes)} tokens")
    
    # Determine input files
    input_files = []
    if args.input_file:
        input_files.append(args.input_file)
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        # Look for token files
        input_files = list(input_dir.glob("*.json")) + list(input_dir.glob("*.txt")) + list(input_dir.glob("*.bin"))
        if not input_files:
            print(f"No token files found in {input_dir}")
            return
    
    # Process each input file
    for input_file in input_files:
        print(f"Decoding {input_file}...")
        
        # Determine input format
        input_format = args.format
        if input_format == "auto":
            input_format = detect_token_format(input_file)
        
        # Load token IDs
        if input_format == "json":
            token_ids = load_tokens_json(input_file)
        elif input_format == "text":
            token_ids = load_tokens_text(input_file)
        elif input_format == "binary":
            # Binary format not implemented for loading yet
            raise NotImplementedError("Binary token loading not yet implemented")
        else:
            raise ValueError(f"Unsupported input format: {input_format}")
        
        # Decode tokens
        decoded_text = tokenizer.decode(token_ids)
        
        # Determine output path
        if args.output:
            if len(input_files) == 1:
                output_path = args.output
            else:
                # Multiple files: use input filename with .txt extension
                output_path = Path(args.output_dir) / f"{input_file.stem}.txt"
        else:
            output_path = input_file.with_suffix(".txt")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save decoded text
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(decoded_text)
        
        print(f"Decoded {len(token_ids)} tokens to {output_path}")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the tokenize command."""
    parser = argparse.ArgumentParser(
        description="Tokenize text files using trained BPE tokenizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode a single file
  %(prog)s encode input.txt --vocab vocab.json --merges merges.txt --output tokens.json
  
  # Decode token IDs back to text
  %(prog)s decode tokens.json --vocab vocab.json --merges merges.txt --output decoded.txt
  
  # Batch encode multiple files
  %(prog)s encode --input-dir ./texts --output-dir ./tokens --vocab vocab.json --merges merges.txt
  
  # Use different output formats
  %(prog)s encode input.txt --vocab vocab.json --merges merges.txt --format text --output tokens.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Encode subcommand
    encode_parser = subparsers.add_parser("encode", help="Encode text files to token IDs")
    encode_parser.add_argument("input_file", nargs="?", type=Path, help="Input text file")
    encode_parser.add_argument("--input-dir", type=Path, help="Input directory containing text files")
    encode_parser.add_argument("--vocab", type=Path, required=True, help="Vocabulary file (vocab.json)")
    encode_parser.add_argument("--merges", type=Path, required=True, help="Merges file (merges.txt)")
    encode_parser.add_argument("--output", type=Path, help="Output file path")
    encode_parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Output directory for batch processing")
    encode_parser.add_argument("--format", choices=["json", "text", "binary", "auto"], default="auto", 
                              help="Output format (default: auto-detect from output file)")
    encode_parser.add_argument("--special-tokens", nargs="*", help="Additional special tokens")
    encode_parser.add_argument("--streaming", action="store_true", help="Use streaming for large files")
    encode_parser.add_argument("--chunk-size", type=int, default=8192, help="Chunk size for streaming (default: 8192)")
    
    # Decode subcommand
    decode_parser = subparsers.add_parser("decode", help="Decode token IDs back to text")
    decode_parser.add_argument("input_file", nargs="?", type=Path, help="Input token file")
    decode_parser.add_argument("--input-dir", type=Path, help="Input directory containing token files")
    decode_parser.add_argument("--vocab", type=Path, required=True, help="Vocabulary file (vocab.json)")
    decode_parser.add_argument("--merges", type=Path, required=True, help="Merges file (merges.txt)")
    decode_parser.add_argument("--output", type=Path, help="Output text file path")
    decode_parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Output directory for batch processing")
    decode_parser.add_argument("--format", choices=["json", "text", "binary", "auto"], default="auto",
                              help="Input format (default: auto-detect from input file)")
    decode_parser.add_argument("--special-tokens", nargs="*", help="Additional special tokens")
    
    return parser


def main() -> None:
    """Main entry point for tokenize CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "encode":
            encode_command(args)
        elif args.command == "decode":
            decode_command(args)
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
