"""Verification for gradcam.py.

1. Known answer (CAM equivalence): for conv -> ReLU -> global average pool ->
   linear, Grad-CAM at the conv output equals ReLU(sum_k w_ck A_k) / (h*w),
   the class activation map of Zhou et al. (2016).
2. Placement: for every layer of LeNet and every DSC variant, an interior
   feature cell is placed centred on its receptive-field centre. The centre is
   measured from the gradient of that cell w.r.t. the input in a copy of the
   model with uniform positive weights and max pools swapped for average pools
   of the same size (same geometry, symmetric receptive field).
3. Every model returns one full-size map per layer.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from gradcam import cam_layers, gradcam, place
from models import MODELS

H, W = 26, 120


def centroid(m):
    ys, xs = torch.meshgrid(torch.arange(m.shape[0], dtype=m.dtype), torch.arange(m.shape[1], dtype=m.dtype),
                            indexing="ij")
    return ((m * ys).sum() / m.sum()).item(), ((m * xs).sum() / m.sum()).item()


def symmetric_copy(model):
    m = copy.deepcopy(model).double()
    for name, mod in list(m.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.MaxPool2d):
                setattr(mod, child_name, nn.AvgPool2d(child.kernel_size, child.stride))
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                mod.weight.fill_(1.0 / mod.weight[0].numel())
                mod.bias.fill_(0.1)
    return m


def main():
    torch.manual_seed(0)
    conv = nn.Conv2d(1, 8, 3, padding=1)                                           # check 1
    fc = nn.Linear(8, 5)
    model = nn.Sequential(conv, nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), fc).double()
    x = torch.rand(4, 1, H, W, dtype=torch.float64)
    target = torch.tensor([0, 1, 2, 4])
    cam = gradcam(model, x, target, layers={"conv": (model[1], 1)})["conv"]
    with torch.no_grad():
        a = model[1](model[0](x))
        expect = F.relu((fc.weight.double()[target][:, :, None, None] * a).sum(1)) / (H * W)
    print(f"[1] Grad-CAM vs CAM: max abs diff {(cam - expect).abs().max():.1e}")
    assert torch.allclose(cam, expect, atol=1e-12)

    worst = 0.0                                                                    # check 2
    for arch in ("lenet", "dsc_avg1"):          # all DSC variants share the trunk
        m = symmetric_copy(MODELS[arch](10))
        for name, (layer, stride) in cam_layers(m).items():
            acts = {}
            hook = layer.register_forward_hook(lambda mod, inp, out: acts.__setitem__("a", out))
            xin = torch.ones(1, 1, H, W, dtype=torch.float64, requires_grad=True)
            m(xin)
            h, w = acts["a"].shape[2:]
            for i, j in ((h // 2, w // 2), (h // 3, w // 4)):
                m(xin)                                  # fresh graph for each cell
                g, = torch.autograd.grad(acts["a"][0, 0, i, j], xin)
                rf = centroid(g[0, 0])
                hot = torch.zeros(1, 1, h, w, dtype=torch.float64)
                hot[0, 0, i, j] = 1
                placed = centroid(place(hot, stride, (H, W))[0])
                err = max(abs(rf[0] - placed[0]), abs(rf[1] - placed[1]))
                worst = max(worst, err)
            hook.remove()
            print(f"[2] {arch} {name}: map {h}x{w}, stride {stride}, placement error {err:.2e} px")
    assert worst < 1e-6

    x = torch.rand(2, 1, H, W, device="cuda")                                      # check 3
    for name, ctor in MODELS.items():
        maps = gradcam(ctor(2000).cuda(), x, torch.tensor([3, 7], device="cuda"))
        assert all(mp.shape == (2, H, W) for mp in maps.values())
    print(f"[3] all {len(MODELS)} architectures return full-size maps for every layer")


if __name__ == "__main__":
    main()
