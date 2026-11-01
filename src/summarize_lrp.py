"""Per-position LRP relevance profile on clean, correctly classified test images.

For each image (true class): share of letter j = relevance in letter j's box.
LRP starts from relevance 1 at the target logit and the image total stays
close to 1, so the raw sum already is a share of the unit relevance.

Decision: the composite stays as fixed (dense-layer epsilon = 1e-6). A small
fraction of images get exploding, cancelling relevance under it, so profiles
use the median over images (mean +- std of that median over runs), and the
fraction of unstable images is reported alongside.
Unstable image: |total relevance - 1| > 0.5, or any letter with |relevance| > 10.

Also reports how much relevance lands outside all cells (background).
Output: results/lrp/clean_profile.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import VOCAB
from render import ROOT

OUT = ROOT / "results" / "lrp"
TOTAL_TOL, LETTER_MAX = 0.5, 10.0


def unstable(d, target):
    total, cell = d[f"{target}_total"], d[f"{target}_cell"]
    return (np.abs(total - 1) > TOTAL_TOL) | (np.nanmax(np.abs(cell), axis=1) > LETTER_MAX)


def profiles(run, box):
    """Median share per position for each word length, clean correct images."""
    d = np.load(OUT / f"{run}_clean.npz")
    ok = d["pred"] == d["label"]
    length = np.array([len(VOCAB[l]) for l in d["label"]])
    share = d[f"true_{box}"]
    return {L: np.nanmedian(share[ok & (length == L), :L], axis=0) for L in range(3, 11)}


def report_unstable(runs):
    print(f"unstable images (|total - 1| > {TOTAL_TOL} or any |letter| > {LETTER_MAX}):")
    for run in runs:
        parts = []
        for name in ("clean", "swap_n2"):
            d = np.load(OUT / f"{run}_{name}.npz")
            wrong = d["pred"] != d["label"]
            parts.append(f"{name} true {unstable(d, 'true').mean():.2%}, "
                         f"pred (errors only) {unstable(d, 'pred')[wrong].mean():.2%}")
        print(f"  {run}: " + " | ".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["lenet_seed0", "lenet_seed1", "lenet_seed2"])
    args = p.parse_args()

    report_unstable(args.runs)
    outside = []
    for run in args.runs:
        d = np.load(OUT / f"{run}_clean.npz")
        ok = d["pred"] == d["label"]
        outside.append(np.median((1 - np.nansum(d["true_cell"], axis=1) / d["true_total"])[ok]))
    print("median share of relevance outside all cells (background): "
          + ", ".join(f"{r}: {o:.3f}" for r, o in zip(args.runs, outside)))

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharey=True)
    for box, color in (("cell", "tab:blue"), ("ink", "tab:red")):
        res = [profiles(run, box) for run in args.runs]
        print(f"\n{box}: median share per position (mean over runs)")
        for ax, L in zip(axes.flat, range(3, 11)):
            meds = np.stack([prof[L] for prof in res])
            med, std = meds.mean(0), meds.std(0)
            print(f"  L={L:2d}: " + " ".join(f"{m:.2f}" for m in med))
            ax.errorbar(np.arange(1, L + 1), med, yerr=std, color=color, marker="o", ms=3,
                        capsize=2, label=f"{box} boxes")
            ax.axhline(1 / L, color="gray", ls="--", lw=0.8)
            ax.set_title(f"length {L}", fontsize=9)
            ax.set_xticks(range(1, L + 1))
    axes[0, 0].legend(fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of letter relevance")
    for ax in axes[1]:
        ax.set_xlabel("letter position")
    fig.suptitle(f"LRP (ε-γ-box), true class, clean correct test images; median over images, "
                 f"mean ± std of that median over {len(args.runs)} seeds; dashed = 1/L", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "clean_profile.png", dpi=200)
    print(f"\nwrote {(OUT / 'clean_profile.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
