"""Build the closed vocabulary.

Pool: wordfreq English top-50k, filtered to lowercase a-z words that are also in
the ENABLE dictionary (removes names, abbreviations, foreign tokens).

Constraints, per word length L in [MIN_LEN, MAX_LEN]:
  1. every (letter, position) pair appears in at least M words, or in as many
     as the pool allows if the pool has fewer than M;
  2. at least MIN_PER_LEN words.
Then the vocabulary is filled up to VOCAB_SIZE with the most frequent unused words.

Selection is deterministic: greedy coverage, ties broken by frequency rank.
Output: data/vocab.txt, one word per line, sorted by length then frequency rank.
"""
import string
from collections import Counter
from pathlib import Path

from wordfreq import top_n_list

MIN_LEN, MAX_LEN = 3, 10
M = 2                 # min words per (length, letter, position)
MIN_PER_LEN = 100
VOCAB_SIZE = 2000
POOL_SIZE = 50_000

ROOT = Path(__file__).resolve().parent.parent
DICT_PATH = ROOT / "data" / "enable1.txt"
OUT_PATH = ROOT / "data" / "vocab.txt"


def load_pool():
    """Frequency-ranked candidate words (index = rank)."""
    dictionary = set(DICT_PATH.read_text().split())
    pool = []
    for w in top_n_list("en", POOL_SIZE):
        if w in dictionary and w.isascii() and w.isalpha() and MIN_LEN <= len(w) <= MAX_LEN:
            pool.append(w)
    return pool


def slots(word):
    return {(c, i) for i, c in enumerate(word)}


def targets_for_length(words):
    """Required count per (letter, position): M, capped by what the pool has."""
    available = Counter(s for w in words for s in slots(w))
    return {s: min(M, n) for s, n in available.items()}


def select_length(words):
    """Greedy cover of the (letter, position) targets for one length.

    `words` is frequency-ordered. Each step picks the word that fills the most
    still-unmet slots; max() keeps the first (most frequent) word on ties.
    """
    need = targets_for_length(words)
    chosen, remaining = [], list(words)
    while any(need.values()):
        best = max(remaining, key=lambda w: sum(need.get(s, 0) > 0 for s in slots(w)))
        chosen.append(best)
        remaining.remove(best)
        for s in slots(best):
            if need.get(s, 0) > 0:
                need[s] -= 1
    # top up to the per-length minimum with the most frequent unused words
    for w in remaining:
        if len(chosen) >= MIN_PER_LEN:
            break
        chosen.append(w)
    return chosen


def build_vocab():
    pool = load_pool()
    rank = {w: i for i, w in enumerate(pool)}
    vocab = []
    for L in range(MIN_LEN, MAX_LEN + 1):
        vocab += select_length([w for w in pool if len(w) == L])
    chosen = set(vocab)
    for w in pool:
        if len(vocab) >= VOCAB_SIZE:
            break
        if w not in chosen:
            vocab.append(w)
            chosen.add(w)
    assert len(vocab) == VOCAB_SIZE, f"pool too small: {len(vocab)}"
    vocab.sort(key=lambda w: (len(w), rank[w]))
    return vocab, pool


def verify(vocab, pool):
    """Check every constraint against the pool and print a summary per length."""
    assert len(vocab) == len(set(vocab)) == VOCAB_SIZE
    print(f"pool: {len(pool)} words | vocab: {len(vocab)}")
    print(" len  words  pairs met (=required)  pairs capped by pool  pairs absent from pool")
    for L in range(MIN_LEN, MAX_LEN + 1):
        vw = [w for w in vocab if len(w) == L]
        need = targets_for_length([w for w in pool if len(w) == L])
        have = Counter(s for w in vw for s in slots(w))
        unmet = [s for s, n in need.items() if have[s] < n]
        assert not unmet, f"length {L}: unmet {unmet}"
        assert len(vw) >= MIN_PER_LEN, f"length {L}: only {len(vw)} words"
        capped = sorted(f"{c}@{i + 1}" for (c, i), n in need.items() if n < M)
        absent = sorted(f"{c}@{i + 1}" for i in range(L) for c in string.ascii_lowercase
                        if (c, i) not in need)
        met = sum(have[s] >= n for s, n in need.items())
        print(f" {L:3d}  {len(vw):5d}  {met:4d} ({len(need)})              "
              f"{len(capped):3d}                   {len(absent):3d}")
        if capped or absent:
            print(f"      capped: {' '.join(capped)}\n      absent: {' '.join(absent)}")
    print("all constraints satisfied")


if __name__ == "__main__":
    vocab, pool = build_vocab()
    verify(vocab, pool)
    OUT_PATH.write_text("\n".join(vocab) + "\n")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
