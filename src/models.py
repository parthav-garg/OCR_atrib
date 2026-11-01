"""Word classifiers. Input: (N, 1, 26, 120) float in [0, 1], ink = 1."""
from functools import partial

import torch.nn as nn

INPUT_HW = (26, 120)


class LeNet(nn.Module):
    """LeNet-5 layout with ReLU and max pooling (original: tanh, average pooling).

    26x120 -conv5-> 22x116 -pool-> 11x58 -conv5-> 7x54 -pool-> 3x27
    -> flatten 16*3*27 = 1296 -> 120 -> 84 -> n_classes
    """

    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 27, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DSCBlock(nn.Module):
    """Depthwise-separable conv: depthwise 3x3 -> ReLU -> pointwise 1x1 -> ReLU (no BatchNorm)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in)
        self.pointwise = nn.Conv2d(c_in, c_out, 1)
        self.act1, self.act2 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.act2(self.pointwise(self.act1(self.depthwise(x))))


class DSCNet(nn.Module):
    """Stem conv + 4 DSC blocks, 2x2 max pool after blocks 2 and 4, adaptive pool head.

    26x120 -stem,b1,b2-> 26x120 -pool-> 13x60 -b3,b4-> 13x60 -pool-> 6x30
    -> adaptive {avg, max} pool to grain x grain -> flatten (64*grain^2) -> n_classes

    `blocks` holds the 4 DSC blocks; their outputs are the Grad-CAM layers.
    """

    STEM = 16
    WIDTHS = (16, 32, 64, 64)
    POOL_AFTER = (1, 3)        # block indices followed by a 2x2 max pool

    def __init__(self, n_classes, head="avg", grain=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, self.STEM, 3, padding=1), nn.ReLU())
        widths = (self.STEM,) + self.WIDTHS
        self.blocks = nn.ModuleList(DSCBlock(a, b) for a, b in zip(widths[:-1], widths[1:]))
        self.pools = nn.ModuleDict({str(i): nn.MaxPool2d(2) for i in self.POOL_AFTER})
        pool = nn.AdaptiveAvgPool2d if head == "avg" else nn.AdaptiveMaxPool2d
        self.head = nn.Sequential(pool(grain), nn.Flatten(),
                                  nn.Linear(self.WIDTHS[-1] * grain * grain, n_classes))
        # Without BatchNorm, PyTorch's default init gives logits that barely depend on
        # the input (spread across images ~1e-6) and training stalls at chance.
        # He init with zero biases keeps the input signal through the ReLU stack.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if str(i) in self.pools:
                x = self.pools[str(i)](x)
        return self.head(x)


MODELS = {"lenet": LeNet}
for _head in ("avg", "max"):
    for _grain in (1, 2, 4):
        MODELS[f"dsc_{_head}{_grain}"] = partial(DSCNet, head=_head, grain=_grain)
