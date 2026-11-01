"""LRP relevance maps and per-letter aggregation.

Composite (fixed before looking at results): zennit EpsilonGammaBox
  first conv: z^B box rule with bounds [0, 1] (the pixel range)
  other convs: gamma = 0.25
  linear layers: epsilon = 1e-6
  ReLU: pass-through; MaxPool: gradient (winner takes all)
LeNet has no BatchNorm, so no canonizer is needed.

Starting relevance is 1 at the target logit (one-hot), so a map's pixel sum
is the share of that unit relevance that reaches the input (biases and the
stabilizers absorb the rest).

Per-letter relevance: sum of the map inside each letter's ink box, and
inside each letter's cell. Ink boxes can overlap, so a pixel in two ink
boxes counts for both letters.
"""
import numpy as np
import torch
from zennit.attribution import Gradient
from zennit.composites import EpsilonGammaBox

COMPOSITE = dict(low=0.0, high=1.0, epsilon=1e-6, gamma=0.25)


def relevance(model, x, target, n_classes):
    """x: (N, 1, H, W) float on the model's device; target: (N,) long. Returns (N, H, W)."""
    composite = EpsilonGammaBox(**COMPOSITE)
    model.eval()
    with Gradient(model, composite) as attributor:
        _, rel = attributor(x, torch.eye(n_classes, device=x.device)[target])
    return rel[:, 0]


def box_sums(maps, boxes):
    """maps: (N, H, W) tensor; boxes: (N, L, 4) int tensor, rows of -1 are padding.

    Returns (N, L) sums of each map over each box (NaN for padding rows),
    computed with a 2-D cumulative sum so each box costs four lookups.
    """
    n, h, w = maps.shape
    s = torch.zeros(n, h + 1, w + 1, dtype=torch.float64, device=maps.device)
    s[:, 1:, 1:] = maps.double().cumsum(1).cumsum(2)
    x0, y0, x1, y1 = [boxes[..., k].long().clamp(0) for k in range(4)]
    x0, x1 = x0.clamp(max=w), x1.clamp(max=w)
    y0, y1 = y0.clamp(max=h), y1.clamp(max=h)
    i = torch.arange(n, device=maps.device)[:, None]
    out = s[i, y1, x1] - s[i, y0, x1] - s[i, y1, x0] + s[i, y0, x0]
    return torch.where(boxes[..., 0] >= 0, out, torch.nan).float()
