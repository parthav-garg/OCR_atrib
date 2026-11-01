"""Verification for lrp.py.

1. Known answer: with the epsilon rule (eps ~ 0) on every layer and all biases
   set to zero, LRP of a ReLU/max-pool network equals gradient x input exactly.
   This checks the zennit wiring on LeNet, including max pooling.
2. box_sums matches a brute-force loop over random maps and boxes.
3. Conservation for the real composite on clean test images: share of the
   unit start relevance that reaches the input.
4. Model randomization: maps from the trained model vs. the same architecture
   with re-initialized weights should be dissimilar (Spearman rank correlation).
5. Figure: clean and swapped images with true/predicted-class maps.
"""
import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from scipy.stats import spearmanr
from zennit.attribution import Gradient
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Pass
from zennit.types import Activation, Convolution

from dataset import OUT_DIR, VOCAB, load
from eval_swap import load_model
from lrp import box_sums, relevance
from models import LeNet
from render import ROOT

OUT = ROOT / "results" / "checks"
RUN = "lenet_seed0"


def main():
    torch.manual_seed(0)
    test = load("test")
    x = torch.from_numpy(test["images"][:256]).unsqueeze(1).cuda().float() / 255.0
    y = torch.from_numpy(test["label"][:256]).long().cuda()
    model = load_model(RUN)

    m0 = copy.deepcopy(model)                                                     # check 1
    with torch.no_grad():
        for mod in m0.modules():
            if getattr(mod, "bias", None) is not None:
                mod.bias.zero_()
    lrp0 = LayerMapComposite([(Activation, Pass()), (Convolution, Epsilon(1e-12)),
                              (torch.nn.Linear, Epsilon(1e-12))])
    onehot = torch.eye(len(VOCAB), device="cuda")[y]
    with Gradient(m0, lrp0) as attributor:
        _, r_lrp = attributor(x, onehot)
    xg = x.clone().requires_grad_(True)
    logit = (m0(xg) * onehot).sum(1)
    logit.sum().backward()
    # LRP starts from relevance 1 at the target; gradient x input sums to the logit
    gxi = xg.grad * xg / logit.detach()[:, None, None, None]
    err = (r_lrp - gxi).abs().max().item() / gxi.abs().max().item()
    print(f"[1] LRP-0 (zero bias) vs gradient x input / logit: max rel. error {err:.2e}; "
          f"relevance sums in [{r_lrp.sum((1, 2, 3)).min():.4f}, {r_lrp.sum((1, 2, 3)).max():.4f}]")
    assert err < 1e-4

    maps = torch.randn(64, 26, 120, device="cuda")                                # check 2
    boxes = torch.from_numpy(test["ink"][:64]).cuda()
    fast = box_sums(maps, boxes).cpu().numpy()
    slow = np.full(fast.shape, np.nan)
    for n in range(64):
        for j, (x0, y0, x1, y1) in enumerate(test["ink"][n]):
            if x0 >= 0:
                slow[n, j] = maps[n, max(y0, 0):y1, max(x0, 0):x1].sum().item()
    print(f"[2] box_sums vs brute force: max abs diff {np.nanmax(np.abs(fast - slow)):.2e}")
    assert np.allclose(fast, slow, atol=1e-3, equal_nan=True)

    r = relevance(model, x, y, len(VOCAB))                                        # check 3
    total = r.sum((1, 2)).cpu().numpy()
    print(f"[3] share of start relevance reaching the input (true class, 256 clean images): "
          f"median {np.median(total):.3f}, 5-95% [{np.percentile(total, 5):.3f}, {np.percentile(total, 95):.3f}]")

    rand = LeNet(len(VOCAB)).cuda()                                               # check 4
    r_rand = relevance(rand, x, y, len(VOCAB))
    rho_rand = [spearmanr(a.ravel(), b.ravel())[0] for a, b in zip(r.cpu().numpy(), r_rand.cpu().numpy())]
    r_seed1 = relevance(load_model("lenet_seed1"), x, y, len(VOCAB))
    rho_seed = [spearmanr(a.ravel(), b.ravel())[0] for a, b in zip(r.cpu().numpy(), r_seed1.cpu().numpy())]
    print(f"[4] Spearman(map, map): trained seed0 vs random init median {np.median(rho_rand):.3f}; "
          f"seed0 vs seed1 median {np.median(rho_seed):.3f}")

    with np.load(OUT_DIR / "swap_n2.npz") as f:                                   # check 5
        swap = {k: f[k][:4000] for k in ("images", "label", "ink", "scrambled", "interior", "pos")}
    xs = torch.from_numpy(swap["images"]).unsqueeze(1).cuda().float() / 255.0
    with torch.no_grad():
        pred = model(xs).argmax(1).cpu().numpy()
    wrong = np.flatnonzero(pred != swap["label"])[:3]
    right = np.flatnonzero(pred == swap["label"])[:2]
    rows = list(right) + list(wrong)
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 1.1 * len(rows)))
    for r_i, k in enumerate(rows):
        xk = xs[k:k + 1]
        r_true = relevance(model, xk, torch.tensor([swap["label"][k]]).cuda(), len(VOCAB))[0].cpu().numpy()
        r_pred = relevance(model, xk, torch.tensor([pred[k]]).cuda(), len(VOCAB))[0].cpu().numpy()
        panels = [(swap["images"][k] / 255.0, "gray", f"{swap['scrambled'][k]} (pos {swap['pos'][k]})"),
                  (r_true, "bwr", f"true: {VOCAB[swap['label'][k]]}"),
                  (r_pred, "bwr", f"pred: {VOCAB[pred[k]]}")]
        for ax, (img, cmap, title) in zip(axes[r_i], panels):
            lim = np.abs(img).max() if cmap == "bwr" else 1
            ax.imshow(img, cmap=cmap, vmin=-lim if cmap == "bwr" else 0, vmax=lim, interpolation="nearest")
            for (x0, y0, x1, y1) in swap["ink"][k][:len(swap["scrambled"][k])]:
                ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                                               edgecolor="black" if cmap == "bwr" else "yellow",
                                               linewidth=0.4))
            ax.set_title(title, fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    path = OUT / "lrp_examples.png"
    fig.savefig(path, dpi=200)
    print(f"[5] wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
