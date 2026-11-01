"""Per-letter LRP relevance on the clean test set and the swap n=2 set.

    python src/run_lrp.py --runs lenet_seed0 lenet_seed1 lenet_seed2

For every image: maps for the true class and the predicted class (computed
once when they are the same). Saved to results/lrp/{run}_{set}.npz:
  label, pred          (N,)
  {true,pred}_ink      (N, MAX_LEN) relevance summed in each letter's ink box
  {true,pred}_cell     (N, MAX_LEN) relevance summed in each letter's cell
  {true,pred}_total    (N,) relevance summed over the whole image
Full maps are not stored; they can be recomputed from the checkpoint.
"""
import argparse

import numpy as np
import torch
from tqdm import tqdm

from dataset import OUT_DIR, VOCAB, load
from eval_swap import load_model
from lrp import box_sums, relevance
from render import ROOT

OUT = ROOT / "results" / "lrp"
BATCH = 512


def letter_relevance(model, images, label, ink, cell):
    out = {k: [] for k in ("pred", "true_ink", "true_cell", "true_total", "pred_ink", "pred_cell", "pred_total")}
    for i in tqdm(range(0, len(label), BATCH), leave=False):
        x = torch.from_numpy(images[i:i + BATCH]).unsqueeze(1).cuda().float() / 255.0
        y = torch.from_numpy(label[i:i + BATCH]).long().cuda()
        b_ink, b_cell = torch.from_numpy(ink[i:i + BATCH]).cuda(), torch.from_numpy(cell[i:i + BATCH]).cuda()
        with torch.no_grad():
            pred = model(x).argmax(1)
        r_true = relevance(model, x, y, len(VOCAB))
        wrong = pred != y
        r_pred = r_true.clone()
        if wrong.any():
            r_pred[wrong] = relevance(model, x[wrong], pred[wrong], len(VOCAB))
        out["pred"].append(pred.cpu().numpy())
        for name, r in (("true", r_true), ("pred", r_pred)):
            out[f"{name}_ink"].append(box_sums(r, b_ink).cpu().numpy())
            out[f"{name}_cell"].append(box_sums(r, b_cell).cpu().numpy())
            out[f"{name}_total"].append(r.sum((1, 2)).cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["lenet_seed0", "lenet_seed1", "lenet_seed2"])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"clean": load("test")}
    with np.load(OUT_DIR / "swap_n2.npz") as f:
        sets["swap_n2"] = {k: f[k] for k in ("images", "label", "ink", "cell")}
    for run in args.runs:
        model = load_model(run)
        for name, d in sets.items():
            res = letter_relevance(model, d["images"], d["label"], d["ink"], d["cell"])
            np.savez_compressed(OUT / f"{run}_{name}.npz", label=d["label"], **res)
            print(f"{run} {name}: {len(d['label'])} images, acc {np.mean(res['pred'] == d['label']):.4f}")


if __name__ == "__main__":
    main()
