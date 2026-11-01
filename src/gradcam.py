"""Grad-CAM at chosen layers, placed on the input pixel grid.

For layer output A (N, C, h, w) and target logit y_c:
  alpha_k = mean over (h, w) of dy_c / dA_k
  cam     = ReLU(sum_k alpha_k * A_k)                   (N, h, w)

Placement on the (H, W) input: bilinear upsampling by the layer's stride s
(align_corners=False puts cell i's centre at pixel (i + 0.5)s - 0.5), then
equal zero padding on each side up to (H, W). That centre-offset is exact for
both layer types used here:
  DSCNet: 'same'-padded convs, so h*s == H and the padding is 0.
  LeNet:  unpadded 5x5 convs; conv1 (s=1) cells centre at i + 2,
          conv2 (s=2) at 2i + 6.5, i.e. pads of 2 and 6 px.
Verified against receptive-field centres in check_gradcam.py.

Layers: DSCNet -> the output of each DSC block (before any pooling);
        LeNet  -> the output of each conv's ReLU.
"""
import torch
import torch.nn.functional as F

from models import DSCNet, LeNet


def cam_layers(model):
    """{name: (module, stride relative to the input)}"""
    if isinstance(model, DSCNet):
        stride, out = 1, {}
        for i, b in enumerate(model.blocks):
            out[f"block{i + 1}"] = (b, stride)
            if str(i) in model.pools:
                stride *= 2
        return out
    if isinstance(model, LeNet):
        return {"conv1": (model.features[1], 1), "conv2": (model.features[4], 2)}
    raise ValueError(type(model))


def place(cam, stride, size):
    """(N, 1, h, w) -> (N, H, W): upsample by stride, then centre with zero padding."""
    H, W = size
    up = F.interpolate(cam, scale_factor=stride, mode="bilinear", align_corners=False) if stride > 1 else cam
    py, px = H - up.shape[2], W - up.shape[3]
    assert py >= 0 and px >= 0 and py % 2 == 0 and px % 2 == 0, (up.shape, size)
    return F.pad(up, (px // 2, px // 2, py // 2, py // 2))[:, 0]


def gradcam(model, x, target, layers=None):
    """x: (N, 1, H, W); target: (N,) long. Returns {layer name: (N, H, W) map}."""
    layers = layers or cam_layers(model)
    acts = {}
    hooks = [m.register_forward_hook(lambda mod, inp, out, n=n: acts.__setitem__(n, out))
             for n, (m, _) in layers.items()]
    model.eval()
    try:
        logits = model(x)
        score = logits.gather(1, target[:, None]).sum()
        grads = torch.autograd.grad(score, [acts[n] for n in layers])
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for (name, (_, stride)), g in zip(layers.items(), grads):
        a = acts[name]
        alpha = g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((alpha * a).sum(1, keepdim=True)).detach()
        out[name] = place(cam, stride, x.shape[2:])
    return out
