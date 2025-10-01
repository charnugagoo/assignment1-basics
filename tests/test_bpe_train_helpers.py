from __future__ import annotations

import os

import pytest

from cs336_basics.bpe_train import (
    _pretokenize,
    _split_on_specials,
    _words_to_symbol_sequences,
    _get_pair_counts,
    _merge_pair_in_sequence,
    _apply_merge,
)


def test_pretokenize_basic():
    text = "Hello, how are you?"
    toks = _pretokenize(text)
    assert toks == ["Hello", ",", " how", " are", " you", "?"]


def test_split_on_specials_removes_and_preserves_boundaries():
    text = "A<|endoftext|>B<|endoftext|>C"
    segs = _split_on_specials(text, ["<|endoftext|>"])
    assert segs == ["A", "B", "C"]


def test_words_to_symbol_sequences_counts_and_byte_level():
    toks = ["A", " A", "🙃"]
    counts = _words_to_symbol_sequences(toks)
    # "A" is one byte 0x41, but " A" includes leading space byte 0x20
    seq_A = (bytes([0x41]),)
    seq_space_A = (bytes([0x20]), bytes([0x41]))
    assert counts[seq_A] == 1
    assert counts[seq_space_A] == 1
    # "🙃" is multi-byte in UTF-8
    assert any(len(sym) == 1 for sym in next(iter(counts)).__class__()) is not None


def test_get_pair_counts_simple():
    # sequences: [a, b, a] x2 and [b, a]
    a, b = b"a", b"b"
    words = {
        (a, b, a): 2,
        (b, a): 1,
    }
    counts = _get_pair_counts(words)
    assert counts[(a, b)] == 2  # from (a,b,a) twice
    assert counts[(b, a)] == 3  # (a,b,a) twice has (b,a) twice + (b,a) once


def test_merge_pair_in_sequence_non_overlapping():
    a, b, c = b"a", b"b", b"c"
    seq = (a, b, b, c)
    merged = _merge_pair_in_sequence(seq, (b, c))
    assert merged == (a, b, b + c)
    merged2 = _merge_pair_in_sequence(seq, (a, b))
    assert merged2 == (a + b, b, c)
    merged2 = _merge_pair_in_sequence((a + b, b, c), (a + b, b))
    assert merged2 == (a + b + b, c)
    merged3 = _merge_pair_in_sequence((a + b + b, c), (a + b + b, c))
    assert merged3 == (a + b + b + c,)
    merged4 = _merge_pair_in_sequence((a + b + b, c), (a + b + b, c))
    assert merged4 == (a + b + b + c,)
    merged5 = _merge_pair_in_sequence((a + b + b, c), (a + b + c, c))
    assert merged5 == (a + b + b, c)


def test_apply_merge_updates_multiset():
    a, b = b"a", b"b"
    words = {
        (a, b, a): 2,
        (b, a): 1,
    }
    merged_words = _apply_merge(words, (a, b))
    # (a,b,a)-> (ab,a), (b,a) unchanged
    assert merged_words[(a + b, a)] == 2
    assert merged_words[(b, a)] == 1
    assert merged_words.keys() == {(a + b, a), (b, a)}


