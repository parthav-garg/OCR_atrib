"""Average LRP heatmaps per word length (clean, correct test images, true class).

    python src/plot_lrp_average.py --runs lenet_seed0 lenet_seed1 lenet_seed2

Each map is shifted by (-dx, -dy) to undo the render jitter, so all maps share
the pen origin (exact: jitter is whole pixels). Letters still sit at different
x positions in different fonts; ticks under each map mark the mean cell
boundaries for that length.

Per length and run: pixelwise median over images (primary, per the medians
decision) and pixelwise mean over stable images only (unstable = the
summarize_lrp.py criterion). Then averaged over runs.
Output: results/lrp/average_maps.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import VOCAB, load
from eval_swap import load_model
from lrp import relevance
from render import JITTER_X, JITTER_Y, ROOT
from summarize_lrp import unstable

OUT = ROOT / "results" / "lrp"


def unjitter(maps, dx, dy):
    """Shift each (H, W) map by (-dy, -dx); pixels shifted in from outside are 0."""
    n, h, w = maps.shape
    padded = np.zeros((n, h + 2 * JITTER_Y, w + 2 * JITTER_X), maps.dtype)
    padded[:, JITTER_Y:JITTER_Y + h, JITTER_X:JITTER_X + w] = maps
    out = np.empty_like(maps)
    for i in range(n):
        y0, x0 = JITTER_Y + dy[i], JITTER_X + dx[i]
        out[i] = padded[i, y0:y0 + h, x0:x0 + w]
    return out


def all_maps(model, images, labels, batch=1024):
    out = []
    for i in range(0, len(labels), batch):
        x = torch.from_numpy(images[i:i + batch]).unsqueeze(1).cuda().float() / 255.0
        y = torch.from_numpy(labels[i:i + batch]).long().cuda()
        out.append(relevance(model, x, y, len(VOCAB)).cpu().numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["lenet_seed0", "lenet_seed1", "lenet_seed2"])
    args = p.parse_args()

    test = load("test")
    length = np.array([len(VOCAB[l]) for l in test["label"]])
    lengths = range(3, 11)
    med = {L: [] for L in lengths}
    mean = {L: [] for L in lengths}
    n_used = {L: [] for L in lengths}
    for run in args.runs:
        res = np.load(OUT / f"{run}_clean.npz")
        ok = res["pred"] == res["label"]
        stable = ~unstable(res, "true")
        maps = unjitter(all_maps(load_model(run), test["images"], test["label"]), test["dx"], test["dy"])
        for L in lengths:
            m = ok & (length == L)
            med[L].append(np.median(maps[m], axis=0))
            mean[L].append(maps[m & stable].mean(axis=0))
            n_used[L].append((int(m.sum()), int((m & stable).sum())))

    # mean cell boundaries per length, in un-jittered coordinates
    cell_x = {}
    for L in lengths:
        c = test["cell"][length == L, :L].astype(float)
        c[..., [0, 2]] -= test["dx"][length == L, None, None]
        cell_x[L] = np.append(c[..., 0].mean(0), c[:, -1, 2].mean())

    fig, axes = plt.subplots(len(lengths), 2, figsize=(12, 1.25 * len(lengths)))
    for r, L in enumerate(lengths):
        for c, (stat, name) in enumerate(((med, "median over images"), (mean, "mean over stable images"))):
            img = np.mean(stat[L], axis=0)
            lim = np.abs(img).max() or 1
            ax = axes[r, c]
            ax.imshow(img, cmap="bwr", vmin=-lim, vmax=lim, interpolation="nearest")
            ax.set_xticks(cell_x[L] - 0.5)
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=4, color="black")
            ax.set_yticks([])
            n_all, n_stable = np.mean(n_used[L], axis=0).astype(int)
            ax.set_title(f"length {L}: {name} (n ≈ {n_all if c == 0 else n_stable} per run)", fontsize=8)
    fig.suptitle(f"LRP (ε-γ-box), true class, clean correct test images, jitter removed; "
                 f"average over {len(args.runs)} seeds; ticks = mean letter-cell boundaries; "
                 f"each panel scaled to its own max |value|", fontsize=9)
    fig.tight_layout()
    path = OUT / "average_maps.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
