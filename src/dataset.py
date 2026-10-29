"""Render the train/val/test splits once and save them.

Per word and per font: draw 8 distinct (dx, dy) jitter positions, assign 6 to
train, 1 to val, 1 to test. Distinct positions mean a test image is never a
train image of the same word and font with only the noise changed.
That gives 48/8/8 images per word over the 8 fonts.

Each image: render -> add Gaussian noise (sigma NOISE) -> clip to [0, 1] ->
quantize to uint8. Saved as data/rendered/{split}.npz with:
  images  (N, H, W) uint8, ink = 255
  label   (N,) int   index into data/vocab.txt
  font    (N,) int   index into render.FONT_NAMES
  dx, dy  (N,) int   jitter
  ink, cell  (N, MAX_LEN, 4) int16 boxes (x0, y0, x1, y1), rows past the
             word's length are -1
"""
import itertools
import json

import numpy as np
from PIL import __version__ as PIL_VERSION
from tqdm import tqdm

from render import (FONT_NAMES, FONT_PX, JITTER_X, JITTER_Y, ROOT, WEIGHT, Canvas,
                    load_font, render)

SEED = 0
NOISE = 0.05
SPLITS = {"train": 6, "val": 1, "test": 1}     # positions per (word, font)
MAX_LEN = 10

VOCAB = (ROOT / "data" / "vocab.txt").read_text().split()
OUT_DIR = ROOT / "data" / "rendered"
POSITIONS = list(itertools.product(range(-JITTER_X, JITTER_X + 1), range(-JITTER_Y, JITTER_Y + 1)))


def pad_boxes(boxes):
    out = np.full((MAX_LEN, 4), -1, np.int16)
    out[:len(boxes)] = boxes
    return out


def build():
    fonts = [load_font(n) for n in FONT_NAMES]
    canvas = Canvas(dict(zip(FONT_NAMES, fonts)), VOCAB)
    rng = np.random.default_rng(SEED)
    per_word = sum(SPLITS.values())
    rows = {s: [] for s in SPLITS}

    for label, word in enumerate(tqdm(VOCAB, desc="render")):
        for font_id, font in enumerate(fonts):
            picks = rng.choice(len(POSITIONS), per_word, replace=False)
            start = 0
            for split, n in SPLITS.items():
                for p in picks[start:start + n]:
                    dx, dy = POSITIONS[p]
                    img, ink, cell = render(word, font, canvas, dx, dy)
                    img = np.clip(img + rng.normal(0, NOISE, img.shape), 0, 1)
                    rows[split].append((np.round(img * 255).astype(np.uint8), label, font_id,
                                        dx, dy, pad_boxes(ink), pad_boxes(cell)))
                start += n

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, r in rows.items():
        images, label, font, dx, dy, ink, cell = zip(*r)
        np.savez_compressed(OUT_DIR / f"{split}.npz", images=np.stack(images),
                            label=np.array(label), font=np.array(font), dx=np.array(dx),
                            dy=np.array(dy), ink=np.stack(ink), cell=np.stack(cell))
        print(f"{split}: {len(r)} images")

    meta = dict(seed=SEED, noise=NOISE, splits=SPLITS, font_px=FONT_PX, weight=WEIGHT,
                jitter=[JITTER_X, JITTER_Y], fonts=FONT_NAMES, canvas=[canvas.height, canvas.width],
                vocab_size=len(VOCAB), pillow=PIL_VERSION)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))


def load(split):
    """Returns a dict of numpy arrays for one split."""
    with np.load(OUT_DIR / f"{split}.npz") as f:
        return {k: f[k] for k in f.files}


if __name__ == "__main__":
    build()
