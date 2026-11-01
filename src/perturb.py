"""Test-time letter-swap images (never used in training).

n=2 swap: two adjacent letters at positions (i, i+1) exchange places once.
  interior:     1 <= i and i+1 <= L-2  (first and last letters stay put)
  touches edge: i == 0 or i+1 == L-1
"Whole word" n=2 = interior + touches-edge swaps. Pairs of identical letters
(e.g. the 'll' in 'will') are skipped because the swap would change nothing.

Each test-split image (word, font, dx, dy) is re-rendered once per valid pair
with the swapped string at the same font and jitter, plus fresh Gaussian noise
from the same distribution as the dataset (seeded).

Saved to data/rendered/swap_n2.npz with:
  images, label, font, dx, dy, ink, cell   as in dataset.py (boxes are for the
                                           swapped string as drawn)
  pos        (N,) int   i, the first swapped position (0-indexed)
  interior   (N,) bool
  scrambled  (N,) str   the drawn string
  in_vocab   (N,) bool  the scrambled string is itself a vocab word
"""
import numpy as np
from tqdm import tqdm

from dataset import NOISE, OUT_DIR, VOCAB, load, pad_boxes
from render import FONT_NAMES, Canvas, load_font, render

SEED = 100


def adjacent_swaps(word):
    """All n=2 adjacent swaps that change the word: list of (i, scrambled, interior)."""
    out = []
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:
            continue
        s = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        out.append((i, s, 1 <= i and i + 1 <= len(word) - 2))
    return out


def build():
    fonts = [load_font(n) for n in FONT_NAMES]
    canvas = Canvas(dict(zip(FONT_NAMES, fonts)), VOCAB)
    vocab = set(VOCAB)
    test = load("test")
    rng = np.random.default_rng(SEED)
    rows = []
    for label, font_id, dx, dy in tqdm(zip(test["label"], test["font"], test["dx"], test["dy"]),
                                       total=len(test["label"]), desc="swap n=2"):
        word = VOCAB[label]
        for i, s, interior in adjacent_swaps(word):
            img, ink, cell = render(s, fonts[font_id], canvas, int(dx), int(dy))
            img = np.clip(img + rng.normal(0, NOISE, img.shape), 0, 1)
            rows.append((np.round(img * 255).astype(np.uint8), label, font_id, dx, dy,
                         pad_boxes(ink), pad_boxes(cell), i, interior, s, s in vocab))
    images, label, font, dx, dy, ink, cell, pos, interior, scrambled, in_vocab = zip(*rows)
    np.savez_compressed(OUT_DIR / "swap_n2.npz", images=np.stack(images), label=np.array(label),
                        font=np.array(font), dx=np.array(dx), dy=np.array(dy), ink=np.stack(ink),
                        cell=np.stack(cell), pos=np.array(pos), interior=np.array(interior),
                        scrambled=np.array(scrambled), in_vocab=np.array(in_vocab))
    interior = np.array(interior)
    print(f"swap_n2: {len(rows)} images ({interior.sum()} interior, {(~interior).sum()} touch an edge), "
          f"{np.sum(in_vocab)} scrambled strings are vocab words")


if __name__ == "__main__":
    build()
