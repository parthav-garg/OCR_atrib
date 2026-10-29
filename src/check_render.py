"""Verification for render.py.

1. Containment: each letter rendered alone at its computed origin has all its
   ink inside its ink box and inside its cell's x-range (up to overhang).
2. Reconstruction: the pixelwise max of the single-letter renders equals the
   whole-word render, i.e. the computed origins are where Pillow draws the
   letters. Differences can only appear where glyphs overlap.
3. Canvas: no ink on the canvas border for the widest words at extreme jitter.
4. Overlay figure: ink boxes (red) and cells (blue) on every font.
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from render import (FONT_NAMES, JITTER_X, JITTER_Y, ROOT, Canvas, draw, letter_boxes,
                    letter_origins, load_font, render)

VOCAB = (ROOT / "data" / "vocab.txt").read_text().split()
OUT = ROOT / "results" / "checks"
SHOW = ["fjord", "quizzes", "typography", "jukebox", "will", "affidavits"]


def solo_renders(word, font, canvas, x, y):
    origins = letter_origins(font, word)
    return [draw(font, c, canvas, x + o, y) for c, o in zip(word, origins)]


def box_mask(shape, box):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    m[max(y0, 0):y1, max(x0, 0):x1] = True
    return m


def main():
    fonts = {n: load_font(n) for n in FONT_NAMES}
    canvas = Canvas(fonts, VOCAB)
    print(f"canvas: {canvas.width} x {canvas.height} px")
    rng = np.random.default_rng(0)
    sample = list(rng.choice(VOCAB, 200, replace=False)) + SHOW

    worst_outside, worst_recon, recon_px = 0.0, 0.0, 0
    for (name, font), word in itertools.product(fonts.items(), sample):
        dx, dy = int(rng.integers(-JITTER_X, JITTER_X + 1)), int(rng.integers(-JITTER_Y, JITTER_Y + 1))
        img, ink, _ = render(word, font, canvas, dx, dy)
        x, y = canvas.pen_x + dx, canvas.baseline + dy
        solos = solo_renders(word, font, canvas, x, y)
        for s, box in zip(solos, ink):                       # check 1
            outside = s[~box_mask(s.shape, box)].sum() / max(s.sum(), 1e-9)
            worst_outside = max(worst_outside, outside)
        diff = np.abs(np.max(solos, axis=0) - img)           # check 2
        worst_recon = max(worst_recon, diff.max())
        recon_px += int((diff > 0.1).sum())
    print(f"[1] max fraction of a letter's ink outside its ink box: {worst_outside:.4f}")
    print(f"[2] max |word - max(letters)|: {worst_recon:.3f}; pixels differing > 0.1: "
          f"{recon_px} over {len(fonts) * len(sample)} renders")

    edge_ink = 0.0                                           # check 3
    for name, font in fonts.items():
        widest = sorted(VOCAB, key=font.getlength)[-20:]
        for word, dx, dy in itertools.product(widest, (-JITTER_X, JITTER_X), (-JITTER_Y, JITTER_Y)):
            img, _, _ = render(word, font, canvas, dx, dy)
            edge_ink = max(edge_ink, img[0].max(), img[-1].max(), img[:, 0].max(), img[:, -1].max())
    print(f"[3] max ink on canvas border: {edge_ink:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)                   # check 4
    fig, axes = plt.subplots(len(fonts), len(SHOW), figsize=(3 * len(SHOW), 0.9 * len(fonts)))
    for r, (name, font) in enumerate(fonts.items()):
        for c, word in enumerate(SHOW):
            img, ink, cell = render(word, font, canvas)
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", interpolation="nearest")
            for (x0, y0, x1, y1), color in [(b, "tab:blue") for b in cell] + [(b, "red") for b in ink]:
                ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                                               fill=False, edgecolor=color, linewidth=0.6))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8)
    fig.tight_layout()
    path = OUT / "render_boxes.png"
    fig.savefig(path, dpi=200)
    print(f"[4] wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
