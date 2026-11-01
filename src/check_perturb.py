"""Verification for perturb.py (swap_n2).

1. Strings: each scrambled string equals the word with letters pos, pos+1
   exchanged and nothing else changed; interior swaps keep first and last.
2. Coverage: each test image appears once per valid pair.
3. Rendering: for every row, a clean re-render of the scrambled string
   reproduces the stored boxes and has no ink on the canvas border.
4. Figure: original vs. swapped, swapped letters' ink boxes highlighted.
"""
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm

from dataset import OUT_DIR, VOCAB, load, pad_boxes
from perturb import adjacent_swaps
from render import FONT_NAMES, ROOT, Canvas, load_font, render

OUT = ROOT / "results" / "checks"


def main():
    with np.load(OUT_DIR / "swap_n2.npz") as f:
        d = {k: f[k] for k in f.files}
    test = load("test")
    fonts = [load_font(n) for n in FONT_NAMES]
    canvas = Canvas(dict(zip(FONT_NAMES, fonts)), VOCAB)

    for label, i, s, interior in zip(d["label"], d["pos"], d["scrambled"], d["interior"]):   # check 1
        w = VOCAB[label]
        diff = [k for k in range(len(w)) if w[k] != s[k]]
        assert diff == [i, i + 1] and s[i] == w[i + 1] and s[i + 1] == w[i], (w, s, i)
        assert interior == (i >= 1 and i + 1 <= len(w) - 2)
        if interior:
            assert s[0] == w[0] and s[-1] == w[-1]
    print("[1] all scrambled strings are exactly one adjacent swap; interior swaps keep first/last")

    expected = Counter()                                                             # check 2
    for label, font, dx, dy in zip(test["label"], test["font"], test["dx"], test["dy"]):
        expected[(label, font, dx, dy)] = len(adjacent_swaps(VOCAB[label]))
    got = Counter(zip(d["label"], d["font"], d["dx"], d["dy"]))
    assert got == +expected
    no_swap = [VOCAB[l] for (l, *_), n in expected.items() if n == 0]
    print(f"[2] one image per valid pair for every test image; "
          f"words with no valid swap: {sorted(set(no_swap))}")

    border = 0.0                                                                     # check 3
    for k in tqdm(range(len(d["label"])), desc="re-render", leave=False):
        clean, ink, cell = render(str(d["scrambled"][k]), fonts[d["font"][k]], canvas,
                                  int(d["dx"][k]), int(d["dy"][k]))
        assert (d["ink"][k] == pad_boxes(ink)).all() and (d["cell"][k] == pad_boxes(cell)).all()
        border = max(border, clean[0].max(), clean[-1].max(), clean[:, 0].max(), clean[:, -1].max())
    print(f"[3] stored boxes match re-render for all {len(d['label'])} rows; "
          f"max clean ink on border: {border:.3f}")

    vocab_hits = sorted({(VOCAB[l], str(s)) for l, s, v in zip(d["label"], d["scrambled"], d["in_vocab"]) if v})
    print(f"    scrambled strings that are vocab words: {vocab_hits}")

    OUT.mkdir(parents=True, exist_ok=True)                                           # check 4
    rng = np.random.default_rng(0)
    rows = rng.choice(len(d["label"]), 8, replace=False)
    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 1.0 * len(rows)))
    for r, k in enumerate(rows):
        word, i = VOCAB[d["label"][k]], int(d["pos"][k])
        font = fonts[d["font"][k]]
        orig, ink_o, _ = render(word, font, canvas, int(d["dx"][k]), int(d["dy"][k]))
        for ax, img, ink, title in [(axes[r, 0], orig, ink_o, word),
                                    (axes[r, 1], d["images"][k] / 255.0, d["ink"][k],
                                     f"{d['scrambled'][k]}  ({'interior' if d['interior'][k] else 'edge'}, pos {i})")]:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            for j in (i, i + 1):
                x0, y0, x1, y1 = ink[j]
                ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                               fill=False, edgecolor="red", linewidth=0.8))
            ax.set_title(title, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    path = OUT / "swap_n2_samples.png"
    fig.savefig(path, dpi=200)
    print(f"[4] wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
