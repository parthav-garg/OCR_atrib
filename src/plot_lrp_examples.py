"""LRP heatmaps for example words, clean vs. swapped.

    python src/plot_lrp_examples.py --run lenet_seed0

One row per word length (3-10): a random clean test image the model gets
right, and one random swap of that same image (interior if the word has
one, else edge). Columns:
  clean image | LRP true class | swapped image | LRP true class | LRP predicted class
Red = positive relevance, blue = negative; each map is scaled to its own
max |value|. Black boxes = letter cells.
Output: results/lrp/examples_{run}.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

from dataset import OUT_DIR, VOCAB, load
from eval_swap import load_model
from lrp import relevance
from render import FONT_NAMES, ROOT

OUT = ROOT / "results" / "lrp"


def lrp_map(model, img, target):
    x = torch.from_numpy(img[None, None]).cuda().float() / 255.0
    return relevance(model, x, torch.tensor([target]).cuda(), len(VOCAB))[0].cpu().numpy()


def show(ax, img, cells, n, title, heat):
    if heat:
        lim = np.abs(img).max() or 1
        ax.imshow(img, cmap="bwr", vmin=-lim, vmax=lim, interpolation="nearest")
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    for x0, y0, x1, y1 in cells[:n]:
        ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="black" if heat else "yellow", linewidth=0.4))
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="lenet_seed0")
    p.add_argument("--seed", type=int, default=0, help="which example images to pick")
    args = p.parse_args()

    model = load_model(args.run)
    test = load("test")
    with np.load(OUT_DIR / "swap_n2.npz") as f:
        swap = {k: f[k] for k in ("images", "label", "font", "dx", "dy", "cell", "scrambled", "interior", "pos")}
    clean_res = np.load(OUT / f"{args.run}_clean.npz")
    swap_res = np.load(OUT / f"{args.run}_swap_n2.npz")
    length = np.array([len(VOCAB[l]) for l in test["label"]])
    rng = np.random.default_rng(args.seed)

    fig, axes = plt.subplots(8, 5, figsize=(14, 9))
    for row, L in enumerate(range(3, 11)):
        i = rng.choice(np.flatnonzero((length == L) & (clean_res["pred"] == test["label"])))
        word = VOCAB[test["label"][i]]
        same = ((swap["label"] == test["label"][i]) & (swap["font"] == test["font"][i])
                & (swap["dx"] == test["dx"][i]) & (swap["dy"] == test["dy"][i]))
        pool = np.flatnonzero(same & swap["interior"]) if (same & swap["interior"]).any() else np.flatnonzero(same)
        k = rng.choice(pool)
        pred = int(swap_res["pred"][k])
        kind = "interior" if swap["interior"][k] else "edge"

        ax = axes[row]
        show(ax[0], test["images"][i], test["cell"][i], L, f"{word} ({FONT_NAMES[test['font'][i]]})", False)
        show(ax[1], lrp_map(model, test["images"][i], test["label"][i]), test["cell"][i], L,
             f"clean, true: {word}", True)
        show(ax[2], swap["images"][k], swap["cell"][k], L,
             f"{swap['scrambled'][k]} ({kind}, pos {swap['pos'][k] + 1}-{swap['pos'][k] + 2})", False)
        show(ax[3], lrp_map(model, swap["images"][k], swap["label"][k]), swap["cell"][k], L,
             f"swapped, true: {word}", True)
        show(ax[4], lrp_map(model, swap["images"][k], pred), swap["cell"][k], L,
             f"swapped, pred: {VOCAB[pred]} ({'correct' if pred == swap['label'][k] else 'wrong'})", True)
    fig.suptitle(f"LRP (ε-γ-box), {args.run}; red = positive, blue = negative relevance; "
                 f"each map scaled to its own max |value|", fontsize=10)
    fig.tight_layout()
    path = OUT / f"examples_{args.run}.png"
    fig.savefig(path, dpi=200)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
