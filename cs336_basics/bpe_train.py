from __future__ import annotations

import os
from collections import Counter
from typing import Iterable

import regex as re


# GPT-2 pre-tokenization pattern
GPT2_PRETOKEN_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def _pretokenize(text: str) -> list[str]:
    """Run GPT-2 style pretokenization on input text.

    Splits the text into a sequence of strings according to the GPT-2
    regex pattern, preserving leading spaces on word-like tokens.

    Args:
        text: Input string to split.

    Returns:
        List of token-like substrings in order.
    """
    pattern = re.compile(GPT2_PRETOKEN_PATTERN)
    return [m.group(0) for m in pattern.finditer(text)]


def _split_on_specials(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on special tokens and remove them from training spans.

    This prevents merges from crossing the boundaries of special tokens.

    Args:
        text: Full input corpus chunk.
        special_tokens: List of special token strings to split on.

    Returns:
        List of substrings that exclude the special tokens themselves.
    """
    if not special_tokens:
        return [text]
    escaped = [re.escape(tok) for tok in special_tokens]
    splitter = re.compile("|".join(escaped))
    # Split and drop the delimiters; we do not include specials in training data
    return [seg for seg in splitter.split(text) if seg]


def _words_to_symbol_sequences(tokens: Iterable[str]) -> Counter[tuple[bytes, ...]]:
    """Convert pretokenized strings to counted byte-symbol sequences.

    Each string is encoded as UTF-8 and represented as a tuple of bytes objects,
    where initially each symbol is a single-byte bytes object.

    Args:
        tokens: Pretokenized string tokens.

    Returns:
        Counter mapping tuples of byte symbols to their frequency.
    """
    word_counter: Counter[tuple[bytes, ...]] = Counter()
    for tok in tokens:
        b = tok.encode("utf-8")
        if not b:
            continue
        seq = tuple(bytes([bb]) for bb in b)
        if seq:
            word_counter[seq] += 1
    return word_counter


def _get_pair_counts(words: Counter[tuple[bytes, ...]]) -> Counter[tuple[bytes, bytes]]:
    """Count adjacent symbol pairs across all sequences.

    Args:
        words: Counter of symbol sequences to their frequency.

    Returns:
        Counter of adjacent pair -> frequency across the multiset of sequences.
    """
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    for seq, freq in words.items():
        if len(seq) < 2:
            continue
        prev = seq[0]
        for cur in seq[1:]:
            pair_counts[(prev, cur)] += freq
            prev = cur
    return pair_counts


def _merge_pair_in_sequence(seq: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all non-overlapping occurrences of a given pair in a sequence.

    Args:
        seq: Original symbol sequence.
        pair: (a, b) pair to merge into a+b.

    Returns:
        New sequence with the pair merged greedily left-to-right.
    """
    a, b = pair
    merged: list[bytes] = []
    i = 0
    n = len(seq)
    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            merged.append(a + b)
            i += 2
        else:
            merged.append(seq[i])
            i += 1
    return tuple(merged)


def _apply_merge(words: Counter[tuple[bytes, ...]], pair: tuple[bytes, bytes]) -> Counter[tuple[bytes, ...]]:
    """Apply a merge pair to all sequences in the multiset.

    Args:
        words: Counter of sequences to frequencies.
        pair: Pair to merge.

    Returns:
        New counter with sequences after applying the merge.
    """
    new_words: Counter[tuple[bytes, ...]] = Counter()
    for seq, freq in words.items():
        new_seq = _merge_pair_in_sequence(seq, pair)
        new_words[new_seq] += freq
    return new_words


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    progress_interval: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return (vocab, merges).

    Processing details:
    - Pretokenizes with the GPT-2 regex pattern to produce initial word-like chunks.
    - Splits out any provided special tokens to avoid merging across their boundaries.
    - Initializes symbols at the byte level (256 single-byte tokens) plus any specials.
    - Iteratively merges the most frequent adjacent pair until `vocab_size` is reached.

    Args:
        input_path: Path to a UTF-8 text file used for training.
        vocab_size: Target total vocabulary size, including initial bytes and specials.
        special_tokens: Optional list of special token strings to include in the vocab.

    Returns:
        vocab: Mapping from token id to token bytes.
        merges: Ordered list of (token1_bytes, token2_bytes) pairs in merge order.
    """
    # TODO: Add chunking for large files (11GB+ data files) to avoid OOM errors.
    # Use find_chunk_boundaries from pretokenization_example.py for parallel processing.
    
    # Special tokens are always added to the vocab.
    special_tokens = special_tokens or []

    # Read the entire file into memory.
    print(f"Reading input file: {input_path}")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Remove special tokens from training spans by splitting on them
    print(f"Removing special tokens: {special_tokens}")
    segments = _split_on_specials(text, special_tokens)
    print(f"Pretokenizing head of segments: {segments[:5]}")
    pre_tokens: list[str] = []
    for seg in segments:
        pre_tokens.extend(_pretokenize(seg))

    # Build word multiset of byte symbol sequences
    print(f"Building word multiset of byte symbol sequences: {pre_tokens[:5]}")
    words = _words_to_symbol_sequences(pre_tokens)
    print(f"Head of word multiset: {list(words.keys())[:5]}")

    # Initial vocabulary: 256 single-byte symbols + specials
    print(f"Initial vocabulary: {special_tokens}")
    init_vocab_values: set[bytes] = set(bytes([i]) for i in range(256))
    for tok in special_tokens:
        init_vocab_values.add(tok.encode("utf-8"))

    print(f"Initial vocabulary preview: {list(init_vocab_values)[:5]}")
    merges: list[tuple[bytes, bytes]] = []

    # Target number of total tokens in vocab
    # We will add one new symbol per merge
    current_vocab_size = len(init_vocab_values)
    print(f"Current vocabulary size: {current_vocab_size}")
    max_merges = max(0, vocab_size - current_vocab_size)

    # Iteratively merge the most frequent adjacent pair until `vocab_size` is reached.
    # The merge is applied to all word sequences in the multiset.
    # TODO: Parallelize this loop using multiple processes.
    if progress_interval and progress_interval > 0:
        print(f"Starting BPE merges: planning to perform up to {max_merges} merges")
    for merge_index in range(1, max_merges + 1):
        pair_counts = _get_pair_counts(words)
        if not pair_counts:
            break
        # Deterministic tie-break: by count desc, then lexicographic bytes order
        best_pair, best_count = max(
            pair_counts.items(), key=lambda kv: (kv[1], kv[0][0] + b"\x00" + kv[0][1])
        )

        # Add the merge to the list of merges.
        merges.append(best_pair)
        # Apply merge to all word sequences.
        words = _apply_merge(words, best_pair)

        # Track newly formed symbol for vocab size accounting.
        init_vocab_values.add(best_pair[0] + best_pair[1])
        current_vocab_size = len(init_vocab_values)

        if progress_interval and progress_interval > 0:
            if merge_index % progress_interval == 0 or current_vocab_size >= vocab_size:
                print(
                    f"Merge {merge_index}/{max_merges}: formed {len(merges)} merges, "
                    f"current vocab size {current_vocab_size}, top pair freq {best_count}"
                )
        if current_vocab_size >= vocab_size:
            break

    # Build final vocab mapping: assign contiguous ids [0..N-1]
    vocab_list = sorted(init_vocab_values)
    vocab: dict[int, bytes] = {i: b for i, b in enumerate(vocab_list)}
    if progress_interval and progress_interval > 0:
        print(f"Completed BPE training with {len(merges)} merges and vocab size {len(vocab_list)}")
    return vocab, merges



