"""Evaluate trained models on the n=2 adjacent-swap test set.

    python src/eval_swap.py --runs lenet_seed0 lenet_seed1 lenet_seed2
    python src/eval_swap.py --runs dsc_avg1_seed0 dsc_avg1_seed1 dsc_avg1_seed2 --out swap_n2/dsc_avg1

A prediction is correct if it is the original word. Swaps whose scrambled
string is itself a vocab word (form/from, ...) are excluded from accuracy
and reported separately, since either answer could be called right.

Outputs in results/swap_n2/ (or results/{--out}/):
  predictions.csv  one row per (run, swap image)
  summary.csv      accuracy per run and condition (+ by word length)
  error_pairs.csv  (condition, true word, predicted word) -> count over runs
  error_pairs.png  most frequent error pairs per condition
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import OUT_DIR, VOCAB, load
from models import MODELS
from render import FONT_NAMES, ROOT

CONDITIONS = {                      # name -> row mask over the swap set
    "interior": lambda d: d["interior"],
    "edge only": lambda d: ~d["interior"],
    "whole word": lambda d: np.ones(len(d["interior"]), bool),
}


@torch.no_grad()
def predict(model, images, batch_size=2048):
    model.eval()
    preds = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(images[i:i + batch_size]).unsqueeze(1).cuda().float() / 255.0
        preds.append(model(x).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def load_model(run):
    arch = json.loads((ROOT / "runs" / run / "metrics.json").read_text())["arch"]
    model = MODELS[arch](len(VOCAB)).cuda()
    model.load_state_dict(torch.load(ROOT / "runs" / run / "best.pt"))
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["lenet_seed0", "lenet_seed1", "lenet_seed2"])
    p.add_argument("--out", default="swap_n2", help="output folder under results/")
    args = p.parse_args()
    OUT = ROOT / "results" / args.out

    with np.load(OUT_DIR / "swap_n2.npz") as f:
        swap = {k: f[k] for k in f.files}
    test = load("test")
    length = np.array([len(VOCAB[l]) for l in swap["label"]])
    # index of the clean test image each swap image came from
    test_index = {k: i for i, k in enumerate(zip(test["label"], test["font"], test["dx"], test["dy"]))}
    src = np.array([test_index[k] for k in zip(swap["label"], swap["font"], swap["dx"], swap["dy"])])

    OUT.mkdir(parents=True, exist_ok=True)
    pred_rows, summary = [], []
    for run in args.runs:
        model = load_model(run)
        clean_ok = predict(model, test["images"]) == test["label"]
        pred = predict(model, swap["images"])
        ok = pred == swap["label"]
        summary.append(dict(run=run, condition="clean test", length="all", n=len(clean_ok),
                            acc=clean_ok.mean(), acc_if_clean_correct=1.0))
        for name, mask_fn in CONDITIONS.items():
            mask = mask_fn(swap) & ~swap["in_vocab"]
            for L in ["all"] + sorted(set(length[mask])):
                m = mask if L == "all" else mask & (length == L)
                summary.append(dict(run=run, condition=name, length=L, n=int(m.sum()), acc=ok[m].mean(),
                                    acc_if_clean_correct=ok[m & clean_ok[src]].mean()))
        pred_rows.append(pd.DataFrame(dict(
            run=run, word=[VOCAB[l] for l in swap["label"]], scrambled=swap["scrambled"],
            pred=[VOCAB[q] for q in pred], correct=ok, clean_correct=clean_ok[src],
            interior=swap["interior"], pos=swap["pos"], length=length, in_vocab=swap["in_vocab"],
            font=[FONT_NAMES[f] for f in swap["font"]])))

    preds = pd.concat(pred_rows)
    preds.to_csv(OUT / "predictions.csv", index=False)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)

    # accuracy: mean +- std over runs
    agg = (summary.groupby(["condition", "length"], sort=False)
           .agg(n=("n", "first"), acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                acc_if_clean_correct=("acc_if_clean_correct", "mean")).reset_index())
    print(agg[agg.length == "all"].to_string(index=False, float_format="%.4f"))
    print("\nby word length (mean over runs):")
    print(agg[agg.length != "all"].pivot(index="length", columns="condition", values="acc_mean")
          .to_string(float_format="%.3f"))

    # swaps that form another vocab word: which word does the model pick?
    v = preds[preds.in_vocab]
    print(f"\nscrambled = another vocab word ({len(v)} predictions): "
          f"original {np.mean(v.pred == v.word):.1%}, scrambled word {np.mean(v.pred == v.scrambled):.1%}, "
          f"other {np.mean((v.pred != v.word) & (v.pred != v.scrambled)):.1%}")

    # error pairs: (true word -> predicted word) counted over runs, fonts and positions
    errors = preds[~preds.correct & ~preds.in_vocab].copy()
    errors["condition"] = np.where(errors.interior, "interior", "edge only")
    pairs = (errors.groupby(["condition", "word", "pred"]).size().rename("count").reset_index()
             .sort_values(["condition", "count"], ascending=[True, False]))
    pairs.to_csv(OUT / "error_pairs.csv", index=False)

    top = 25
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.28 * top + 1.2))
    for ax, cond in zip(axes, ["interior", "edge only"]):
        t = pairs[pairs.condition == cond].head(top)[::-1]
        ax.barh([f"{w} → {q}" for w, q in zip(t.word, t.pred)], t["count"], color="tab:blue")
        ax.set_title(f"{cond}: top {top} error pairs\n(count over {len(args.runs)} runs × fonts × positions)",
                     fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("count")
    fig.tight_layout()
    fig.savefig(OUT / "error_pairs.png", dpi=200)
    print(f"\nwrote {OUT.relative_to(ROOT)}/: predictions.csv, summary.csv, error_pairs.csv, error_pairs.png")


if __name__ == "__main__":
    main()
