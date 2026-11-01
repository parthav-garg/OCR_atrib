"""Cross-seed agreement of LRP maps on the same clean test images.

    python src/seed_agreement.py --runs lenet_seed0 lenet_seed1 lenet_seed2

Images: clean test images every run classifies correctly; maps for the true class.
Agreement per image and run pair, as Spearman rank correlation:
  pixel:  over pixels inside the word's region (bounding box of its letter
          cells), so empty background doesn't dominate the ranking
  letter: over the per-letter cell sums (words with >= 5 letters)
Reference: each run vs. a randomly initialised LeNet (should be ~0).

Outputs in results/lrp/:
  seed_agreement_examples.png  same images, maps from each seed side by side
  seed_agreement_hist.png      distribution of per-image correlations
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

from dataset import VOCAB, load
from eval_swap import load_model
from lrp import box_sums, relevance
from models import LeNet
from render import FONT_NAMES, ROOT

OUT = ROOT / "results" / "lrp"


def all_maps(model, images, labels, batch=1024):
    out = []
    for i in range(0, len(labels), batch):
        x = torch.from_numpy(images[i:i + batch]).unsqueeze(1).cuda().float() / 255.0
        y = torch.from_numpy(labels[i:i + batch]).long().cuda()
        out.append(relevance(model, x, y, len(VOCAB)).cpu().numpy())
    return np.concatenate(out)


def word_region(cells, n):
    x0, y0 = cells[:n, 0].min(), cells[:n, 1].min()
    x1, y1 = cells[:n, 2].max(), cells[:n, 3].max()
    return slice(max(y0, 0), y1), slice(max(x0, 0), x1)


def pixel_rho(a, b, region):
    ra, rb = rankdata(a[region].ravel()), rankdata(b[region].ravel())
    return np.corrcoef(ra, rb)[0, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["lenet_seed0", "lenet_seed1", "lenet_seed2"])
    args = p.parse_args()

    test = load("test")
    length = np.array([len(VOCAB[l]) for l in test["label"]])
    ok = np.ones(len(length), bool)
    for run in args.runs:
        ok &= np.load(OUT / f"{run}_clean.npz")["pred"] == test["label"]
    idx = np.flatnonzero(ok)
    imgs, labels = test["images"][idx], test["label"][idx]
    cells, L = test["cell"][idx], length[idx]

    maps = {run: all_maps(load_model(run), imgs, labels) for run in args.runs}
    torch.manual_seed(123)
    maps["random init"] = all_maps(LeNet(len(VOCAB)).cuda(), imgs, labels)
    letters = {k: box_sums(torch.from_numpy(m), torch.from_numpy(cells)).numpy() for k, m in maps.items()}
    regions = [word_region(c, n) for c, n in zip(cells, L)]

    pairs = list(itertools.combinations(args.runs, 2)) + [(r, "random init") for r in args.runs[:1]]
    rho_px, rho_let = {}, {}
    long_words = np.flatnonzero(L >= 5)
    for a, b in pairs:
        rho_px[(a, b)] = np.array([pixel_rho(maps[a][i], maps[b][i], regions[i]) for i in range(len(idx))])
        rho_let[(a, b)] = np.array([spearmanr(letters[a][i, :L[i]], letters[b][i, :L[i]])[0]
                                    for i in long_words])
    print(f"{len(idx)} clean images correct for all runs ({len(long_words)} with >= 5 letters)")
    print("median Spearman  [pixel, word region]  [letter cells, L>=5]")
    for a, b in pairs:
        print(f"  {a} vs {b}: {np.nanmedian(rho_px[(a, b)]):.3f}  {np.nanmedian(rho_let[(a, b)]):.3f}")

    # figure 1: same images, maps from each seed
    rng = np.random.default_rng(0)
    pick = [rng.choice(np.flatnonzero(L == n)) for n in (4, 5, 6, 7, 8, 10)]
    cols = args.runs + ["random init"]
    fig, axes = plt.subplots(len(pick), len(cols) + 2, figsize=(3.0 * (len(cols) + 2), 1.25 * len(pick)),
                             gridspec_kw=dict(width_ratios=[1] * (len(cols) + 1) + [0.7]))
    for r, i in enumerate(pick):
        word = VOCAB[labels[i]]
        ax = axes[r, 0]
        ax.imshow(imgs[i], cmap="gray", interpolation="nearest")
        ax.set_title(f"{word} ({FONT_NAMES[test['font'][idx[i]]]})", fontsize=7)
        for c, run in enumerate(cols, start=1):
            m = maps[run][i]
            lim = np.abs(m).max() or 1
            ax = axes[r, c]
            ax.imshow(m, cmap="bwr", vmin=-lim, vmax=lim, interpolation="nearest")
            for x0, y0, x1, y1 in cells[i, :L[i]]:
                ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                                               edgecolor="black", linewidth=0.3))
            title = run
            if run != args.runs[0]:
                title += f"  ρ vs {args.runs[0]}: {pixel_rho(maps[args.runs[0]][i], m, regions[i]):.2f}"
            ax.set_title(title, fontsize=7)
        ax = axes[r, -1]
        for k, run in enumerate(args.runs):
            ax.bar(np.arange(L[i]) + (k - 1) * 0.27, letters[run][i, :L[i]], width=0.27, label=run)
        ax.set_xticks(range(L[i]))
        ax.set_xticklabels(list(word), fontsize=7)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title("per-letter (cells)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
    axes[0, -1].legend(fontsize=5)
    for ax in axes[:, :-1].flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("LRP (ε-γ-box), true class, same clean images; each map scaled to its own max |value|; "
                 "ρ = Spearman over the word region", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "seed_agreement_examples.png", dpi=200)

    # figure 2: distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, rho, name in ((axes[0], rho_px, "pixel (word region)"), (axes[1], rho_let, "letter cells (L ≥ 5)")):
        for (a, b), v in rho.items():
            ax.hist(v[~np.isnan(v)], bins=40, range=(-1, 1), histtype="step", lw=1.2,
                    label=f"{a} vs {b} (median {np.nanmedian(v):.2f})",
                    ls="--" if b == "random init" else "-")
        ax.set_title(f"per-image Spearman, {name}", fontsize=9)
        ax.set_xlabel("Spearman ρ")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "seed_agreement_hist.png", dpi=200)
    print("wrote results/lrp/seed_agreement_examples.png, seed_agreement_hist.png")


if __name__ == "__main__":
    main()
