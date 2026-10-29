"""Verification for the saved splits.

1. Balance: 48/8/8 images per word, 6/1/1 per (word, font).
2. No (word, font, dx, dy) appears twice, within or across splits.
3. Stored boxes equal boxes recomputed from (label, font, dx, dy), and the
   stored image minus a clean re-render is only noise.
4. Overlay of random stored samples with their stored boxes.
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from dataset import MAX_LEN, NOISE, SPLITS, VOCAB, load, pad_boxes
from render import FONT_NAMES, ROOT, Canvas, load_font, render

OUT = ROOT / "results" / "checks"


def main():
    data = {s: load(s) for s in SPLITS}
    fonts = [load_font(n) for n in FONT_NAMES]
    canvas = Canvas(dict(zip(FONT_NAMES, fonts)), VOCAB)

    for s, d in data.items():                                            # check 1
        per_word = set(Counter(d["label"]).values())
        per_word_font = set(Counter(zip(d["label"], d["font"])).values())
        print(f"[1] {s:5s} {d['images'].shape}  per word: {per_word}  per (word, font): {per_word_font}")
        assert per_word == {SPLITS[s] * len(FONT_NAMES)} and per_word_font == {SPLITS[s]}

    keys = [k for d in data.values() for k in zip(d["label"], d["font"], d["dx"], d["dy"])]
    print(f"[2] duplicate (word, font, dx, dy): {len(keys) - len(set(keys))}")   # check 2
    assert len(keys) == len(set(keys))

    rng = np.random.default_rng(1)                                       # check 3
    resid = []
    for s, d in data.items():
        for i in rng.choice(len(d["label"]), 300, replace=False):
            word, font = VOCAB[d["label"][i]], fonts[d["font"][i]]
            clean, ink, cell = render(word, font, canvas, int(d["dx"][i]), int(d["dy"][i]))
            assert (d["ink"][i] == pad_boxes(ink)).all() and (d["cell"][i] == pad_boxes(cell)).all()
            resid.append(d["images"][i] / 255.0 - clean)
    resid = np.stack(resid)
    print(f"[3] boxes match re-render on 900 samples; residual mean {resid.mean():+.4f}, "
          f"std {resid.std():.4f} (noise sigma {NOISE}, clipped)")

    d = data["test"]                                                     # check 4
    idx = rng.choice(len(d["label"]), 12, replace=False)
    fig, axes = plt.subplots(4, 3, figsize=(9, 3.2))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(d["images"][i], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for (x0, y0, x1, y1) in d["ink"][i][:len(VOCAB[d["label"][i]])]:
            ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                           fill=False, edgecolor="red", linewidth=0.6))
        ax.set_title(f"{VOCAB[d['label'][i]]} | {FONT_NAMES[d['font'][i]]} | "
                     f"dx={d['dx'][i]} dy={d['dy'][i]}", fontsize=6)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    path = OUT / "dataset_samples.png"
    fig.savefig(path, dpi=200)
    print(f"[4] wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
