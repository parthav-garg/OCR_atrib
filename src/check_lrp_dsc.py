"""LRP wiring check for the DSC variants (random init, or trained weights with --trained).

Known answer: with all biases zeroed and epsilon / sum-pool stabilizers ~0, LRP
of a ReLU network with linear, max-pool and average-pool layers equals
gradient x input / target logit. Run in float64 so that tiny activations
(random init, zero biases) don't turn rounding into a false failure.
Also prints shapes per block and parameter counts.
"""
import argparse
import copy

import torch
from zennit.attribution import Gradient
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Norm, Pass
from zennit.types import Activation, AvgPool, Convolution

from dataset import load
from eval_swap import load_model
from models import MODELS


def lrp0_error(model, x, target):
    m = copy.deepcopy(model).double().eval()
    with torch.no_grad():
        for mod in m.modules():
            if getattr(mod, "bias", None) is not None:
                mod.bias.zero_()
    onehot = torch.eye(2000, device=x.device, dtype=torch.float64)[target]
    comp = LayerMapComposite([(Activation, Pass()), (AvgPool, Norm(stabilizer=1e-30)),
                              (Convolution, Epsilon(1e-30)), (torch.nn.Linear, Epsilon(1e-30))])
    with Gradient(m, comp) as att:
        _, r = att(x, onehot)
    xg = x.clone().requires_grad_(True)
    logit = (m(xg) * onehot).sum(1)
    logit.sum().backward()
    gxi = xg.grad * xg / logit.detach()[:, None, None, None]
    return ((r - gxi).abs().max() / gxi.abs().max()).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trained", action="store_true", help="use runs/{arch}_seed0 weights")
    args = p.parse_args()
    test = load("test")
    x = torch.from_numpy(test["images"][:16]).unsqueeze(1).cuda().double() / 255.0
    target = torch.from_numpy(test["label"][:16]).long().cuda()
    torch.manual_seed(0)
    for name in [k for k in MODELS if k.startswith("dsc")]:
        model = load_model(f"{name}_seed0") if args.trained else MODELS[name](2000).cuda()
        h, shapes = model.stem(x.float()), []
        for i, b in enumerate(model.blocks):
            h = b(h)
            shapes.append(tuple(h.shape[1:]))
            if str(i) in model.pools:
                h = model.pools[str(i)](h)
        n = sum(q.numel() for q in model.parameters())
        err = lrp0_error(model, x, target)
        print(f"{name}: blocks {shapes}, {n:,} params, LRP-0 vs grad x input / logit: {err:.1e}")
        assert err < 1e-9


if __name__ == "__main__":
    main()
