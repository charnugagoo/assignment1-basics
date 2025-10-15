"""Entry point for running cs336_basics as a module."""

import argparse
import sys
from pathlib import Path

from .train_bpe_cli import main as train_main
from .tokenize_cli import main as tokenize_main


def main() -> None:
    """Main entry point with subcommands."""
    # Check if this is a tokenize command and handle it directly
    if len(sys.argv) > 1 and sys.argv[1] == "tokenize":
        # Pass through all arguments after "tokenize" to tokenize_cli
        tokenize_args = sys.argv[2:]  # Skip "python -m cs336_basics" and "tokenize"
        
        # Temporarily replace sys.argv for tokenize_cli
        original_argv = sys.argv[:]
        sys.argv = ["tokenize_cli"] + tokenize_args
        try:
            tokenize_main()
        finally:
            sys.argv = original_argv
        return
    
    # Handle train command and legacy usage
    parser = argparse.ArgumentParser(
        description="CS336 Basics: BPE tokenizer training and text tokenization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  train     Train a BPE tokenizer on text data
  tokenize  Tokenize text files using trained tokenizers

Examples:
  # Train a tokenizer
  %(prog)s train data/sample.txt --vocab-size 1000
  
  # Tokenize text files
  %(prog)s tokenize encode input.txt --vocab vocab.json --merges merges.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train subcommand (existing functionality)
    train_parser = subparsers.add_parser("train", help="Train a BPE tokenizer")
    train_parser.add_argument("input_file", type=Path, help="Input text file for training")
    train_parser.add_argument("--vocab-size", type=int, default=50000, help="Target vocabulary size")
    train_parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"], help="Special tokens")
    train_parser.add_argument("--output-dir", type=Path, default=Path.cwd(), help="Output directory")
    train_parser.add_argument("--vocab-file", type=str, default="vocab.json", help="Vocabulary filename")
    train_parser.add_argument("--merges-file", type=str, default="merges.txt", help="Merges filename")
    train_parser.add_argument("--progress-interval", type=int, default=1000, help="Progress interval")
    
    # Handle legacy usage (no subcommand = train)
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] not in ["train"]):
        # Legacy usage: treat first argument as input file for training
        if len(sys.argv) > 1:
            # Insert "train" as first argument
            sys.argv.insert(1, "train")
        else:
            # No arguments, show help
            parser.print_help()
            return
    
    args = parser.parse_args()
    
    if args.command == "train":
        # Set up arguments for train_bpe_cli
        train_args = [
            str(args.input_file),
            "--vocab-size", str(args.vocab_size),
            "--output-dir", str(args.output_dir),
            "--vocab-file", args.vocab_file,
            "--merges-file", args.merges_file,
            "--progress-interval", str(args.progress_interval),
        ]
        if args.special_tokens:
            train_args.extend(["--special-tokens"] + args.special_tokens)
        
        # Temporarily replace sys.argv for train_bpe_cli
        original_argv = sys.argv[:]
        sys.argv = ["train_bpe_cli"] + train_args
        try:
            train_main()
        finally:
            sys.argv = original_argv
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
