"""Train one word classifier on the rendered splits.

    python src/train.py --arch lenet --seed 0

Adam, ReduceLROnPlateau on val loss, early stopping on val loss; the
checkpoint with the lowest val loss is kept and evaluated on test.
Outputs in runs/{arch}_seed{seed}/: best.pt, history.csv, metrics.json.
"""
import argparse
import csv
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # needed for deterministic cuBLAS

import numpy as np
import torch
import torch.nn.functional as F

from dataset import VOCAB, load
from models import MODELS
from render import ROOT


def to_device(split, device, n_words=None):
    """Whole split on the GPU as uint8 images + int64 labels (~300 MB for train).

    n_words keeps only labels < n_words, so every split covers the same words.
    """
    d = load(split)
    keep = d["label"] < (n_words or len(VOCAB))
    x = torch.from_numpy(d["images"][keep]).unsqueeze(1).to(device)
    y = torch.from_numpy(d["label"][keep]).long().to(device)
    return x, y


def batches(x, y, batch_size, generator=None, channels_last=False):
    order = (torch.randperm(len(y), device=y.device, generator=generator)
             if generator is not None else torch.arange(len(y), device=y.device))
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    for i in range(0, len(y), batch_size):
        idx = order[i:i + batch_size]
        yield (x[idx].float() / 255.0).contiguous(memory_format=fmt), y[idx]


@torch.no_grad()
def evaluate(model, x, y, batch_size=1024, bf16=False):
    model.eval()
    loss, correct = 0.0, 0
    for xb, yb in batches(x, y, batch_size, channels_last=bf16):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(xb).float()
        loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return loss / len(y), correct / len(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="lenet", choices=MODELS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--plateau-patience", type=int, default=3)   # epochs before LR x0.5
    p.add_argument("--stop-patience", type=int, default=8)      # epochs before stopping
    p.add_argument("--n-words", type=int, default=None, help="use only the first N words (smoke test)")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast + channels-last for training and per-epoch val (~2x for DSC); "
                        "weights stay fp32 and the final val/test evaluation runs in fp32")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    out = ROOT / "runs" / (args.out or f"{args.arch}_seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    x_tr, y_tr = to_device("train", device, args.n_words)
    x_va, y_va = to_device("val", device, args.n_words)

    model = MODELS[args.arch](len(VOCAB)).to(device)
    if args.bf16:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=args.plateau_patience)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    print(f"{out.name}: {n_params:,} params, {len(y_tr)} train / {len(y_va)} val")

    history, best_loss, best_epoch = [], float("inf"), 0
    for epoch in range(1, args.max_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for xb, yb in batches(x_tr, y_tr, args.batch_size, gen, channels_last=args.bf16):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                logits = model(xb).float()
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
        va_loss, va_acc = evaluate(model, x_va, y_va, bf16=args.bf16)
        sched.step(va_loss)
        row = dict(epoch=epoch, lr=opt.param_groups[0]["lr"], train_loss=tr_loss / len(y_tr),
                   train_acc=tr_correct / len(y_tr), val_loss=va_loss, val_acc=va_acc)
        history.append(row)
        print(f"ep {epoch:3d}  lr {row['lr']:.1e}  train {row['train_loss']:.3f}/{row['train_acc']:.3f}"
              f"  val {va_loss:.3f}/{va_acc:.3f}  {time.time() - t0:.1f}s")
        if va_loss < best_loss:
            best_loss, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), out / "best.pt")
        elif epoch - best_epoch >= args.stop_patience:
            break

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    model = MODELS[args.arch](len(VOCAB)).to(device)          # fresh fp32, default memory format
    model.load_state_dict(torch.load(out / "best.pt"))
    _, va_acc = evaluate(model, x_va, y_va)
    del x_tr, y_tr
    test_acc = None
    if args.n_words is None:
        x_te, y_te = to_device("test", device)
        _, test_acc = evaluate(model, x_te, y_te)
    metrics = dict(vars(args), params=n_params, best_epoch=best_epoch, epochs_run=len(history),
                   val_loss=best_loss, val_acc=va_acc, test_acc=test_acc, torch=torch.__version__)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"best epoch {best_epoch}: val acc {va_acc:.4f}, test acc {test_acc}")


if __name__ == "__main__":
    main()
