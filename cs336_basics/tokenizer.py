from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any
import json
import os
import regex as re

from .bpe_train import GPT2_PRETOKEN_PATTERN


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        """
        Initialize a Tokenizer from a vocabulary, merges, and special tokens.

        Args:
            vocab: A dictionary mapping token IDs to bytes.
            merges: A list of tuples representing BPE merges.
            special_tokens: A list of special tokens. We assume the mergest list to be the merge (training) order, earliest first.
        """
        # Build vocab maps
        self.id_to_bytes: dict[int, bytes] = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in self.id_to_bytes.items()}
        print(f"bytes_to_id size: {len(self.bytes_to_id)}")

        # Handle special tokens: ensure present in vocab
        self.special_tokens: list[str] = special_tokens or []
        print(f"special_tokens size: {len(self.special_tokens)}")
        self.special_tokens_bytes: list[bytes] = []
        self._special_bytes_to_str: dict[bytes, str] = {}
        for tok in self.special_tokens:
            tok_b = tok.encode("utf-8")
            self.special_tokens_bytes.append(tok_b)
            self._special_bytes_to_str[tok_b] = tok
            if tok_b not in self.bytes_to_id:
                new_id = len(self.id_to_bytes)
                self.id_to_bytes[new_id] = tok_b
                self.bytes_to_id[tok_b] = new_id
                print(f"special token {tok} {tok_b} mapping to id {new_id}")
        print(f"bytes_to_id size after special tokens: {len(self.bytes_to_id)}")
        

        # Merge ranking for BPE
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {p: i for i, p in enumerate(self.merges)}
        print(f"merge_rank size: {len(self.merge_rank)}")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        # Load vocab
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_raw = json.load(vf)

        id_to_bytes: dict[int, bytes] = {}
        byte_decoder = cls._gpt2_unicode_to_byte_decoder()
        # Handle common formats:
        # 1) {token(str): id(int)} e.g., GPT-2 style (we'll utf-8 encode token)
        # 2) {id(str|int): token(str|list[int])}
        if vocab_raw and isinstance(next(iter(vocab_raw.keys())), str) and not next(iter(vocab_raw.keys())).isdigit():
            # GPT-2 style vocab
            # token -> id
            for token_str, idx in vocab_raw.items():
                if isinstance(idx, str) and idx.isdigit():
                    idx = int(idx)
                # Decode GPT-2 unicode-space token into original bytes
                token_bytes = bytes([byte_decoder[ch] for ch in token_str])
                id_to_bytes[int(idx)] = token_bytes
        else:
            # bpe.vocab.json style vocab
            # id -> representation (e.g., bpe.vocab.json)
            for k, v in vocab_raw.items():
                idx = int(k) if not isinstance(k, int) else k
                if isinstance(v, list):
                    token_bytes = bytes(v)
                elif isinstance(v, str):
                    # Try GPT-2 decoding first; fall back to utf-8
                    try:
                        token_bytes = bytes([byte_decoder[ch] for ch in v])
                    except KeyError:
                        token_bytes = v.encode("utf-8")
                else:
                    raise ValueError("Unsupported vocab value type")
                id_to_bytes[int(idx)] = token_bytes

        # Normalize ids to contiguous 0..N-1 only if necessary.
        sorted_ids = sorted(id_to_bytes.keys())
        if sorted_ids == list(range(len(sorted_ids))):
            vocab = id_to_bytes  # already contiguous 0..N-1, preserve original ids
        else:
            remapped: dict[int, bytes] = {}
            for new_id, old_id in enumerate(sorted_ids):
                remapped[new_id] = id_to_bytes[old_id]
            vocab = remapped

        # Load merges
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as merge_file:
            for line in merge_file:
                line = line.strip()
                if not line or line.startswith("#"): # skip empty lines and comments
                    continue
                parts = line.split()
                if len(parts) != 2: # skip lines with not exactly two parts
                    continue
                a, b = parts # a and b are the two tokens to merge
                a_b = bytes([byte_decoder[ch] for ch in a])
                b_b = bytes([byte_decoder[ch] for ch in b])
                merges.append((a_b, b_b))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @staticmethod
    def _gpt2_unicode_to_byte_decoder() -> dict[str, int]:
        # Reconstruct GPT-2 byte -> unicode mapping, then invert
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
            range(ord("®"), ord("ÿ") + 1)
        )
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)
                n += 1
        characters = [chr(n) for n in cs]
        byte_to_unicode = dict(zip(bs, characters))
        # invert
        return {v: k for k, v in byte_to_unicode.items()}

    def encode(self, text: str) -> list[int]:
        if not text:
            return []

        # Build special token matcher (order by length desc to handle overlaps greedily)
        specials = list(self.special_tokens) # convert to list to allow sorting
        specials.sort(key=len, reverse=True) # sort by length descending to handle overlaps greedily
        segments: list[tuple[str, bool]] = []  # (substring, is_special)
        if specials:
            pattern = re.compile("|".join(re.escape(tok) for tok in specials))
            pos = 0
            for m in pattern.finditer(text):
                if m.start() > pos:
                    segments.append((text[pos:m.start()], False))
                segments.append((m.group(0), True))
                pos = m.end()
            if pos < len(text):
                segments.append((text[pos:], False))
        else:
            segments.append((text, False))

        gpt2_re = re.compile(GPT2_PRETOKEN_PATTERN)
        out_ids: list[int] = []

        for seg, is_special in segments:
            if not seg:
                continue
            if is_special:
                tok_b = seg.encode("utf-8")
                out_ids.append(self.bytes_to_id[tok_b])
                continue
            # Pretokenize, then BPE each token
            for m in gpt2_re.finditer(seg):
                piece = m.group(0)
                piece_b = piece.encode("utf-8")
                for token_bytes in self._bpe(piece_b):
                    out_ids.append(self.bytes_to_id[token_bytes])
        return out_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            # Process each chunk independently to keep memory bounded
            for _id in self.encode(chunk):
                yield _id

    def decode(self, ids: list[int]) -> str:
        # Minimal decode using UTF-8 roundtrip
        if not ids:
            return ""
        parts = []
        for _id in ids:
            parts.append(self.id_to_bytes[_id])
        data = b"".join(parts)
        try:
            return data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            # For single-token decode convenience in tests, preserve specials exactly
            if len(ids) == 1:
                single_b = parts[0]
                if single_b in self._special_bytes_to_str:
                    return self._special_bytes_to_str[single_b]
                # Otherwise, return a best-effort decode that won't raise
                return single_b.decode("utf-8", errors="ignore")
            raise

    # --- internal helpers ---
    def _bpe(self, byte_seq: bytes) -> list[bytes]:
        """
        Apply BPE to a byte sequence.

        Args:
            byte_seq: The byte sequence to apply BPE to.

        Returns:
            A list of bytes representing the BPE-ed sequence.
        """
        # Start with single-byte symbols
        symbols = [bytes([b]) for b in byte_seq]
        if len(symbols) <= 1:
            return symbols

        while True:
            # Find best-ranked adjacent pair
            best_pair = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            # Merge all non-overlapping occurrences of best_pair
            merged: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    merged.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            if len(symbols) <= 1:
                break
        return symbols


