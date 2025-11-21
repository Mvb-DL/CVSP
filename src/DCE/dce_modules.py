"""
DCE-YOLOv8 Custom Modules
Implementation based on: "DCE-YOLOv8: Lightweight and Accurate Object Detection for Drone Vision"
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import Bottleneck

# --- Divided Context Extraction (DCE) ---
# Paper Reference: Section III.C / Eq. (2)
# Trennt Features in "Processing" (Low-Level) und "Identity"
class DCE(nn.Module):
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e) 
        self.split_ratio = 0.75  # 3/4 Processing, 1/4 Identity (Paper logic)
        c_process = int(c1 * self.split_ratio)
        c_identity = c1 - c_process

        # Partial Conv Logic
        self.cv1 = Conv(c_process, self.c, 1, 1)
        self.m = nn.ModuleList([Conv(self.c, self.c, 3, 1, g=g) for _ in range(n)])
        self.cv2 = Conv(self.c + c_identity, c2, 1, 1)

    def forward(self, x):
        split_idx = int(x.size(1) * self.split_ratio)
        x_process = x[:, :split_idx, :, :]
        x_identity = x[:, split_idx:, :, :]

        y = self.cv1(x_process)
        for conv in self.m:
            y = conv(y)

        # Feature Fusion (F)
        return self.cv2(torch.cat([y, x_identity], dim=1))


# --- Efficient Residual Bottleneck (ERB) ---
# Paper Reference: Section III.B / Eq. (1)
# Leichter als C2f, nutzt Addition statt Concatenation
class ERB(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 1, 1)
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0) for _ in range(n)
        ])
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv1(x)
        for m in self.m:
            y = m(y)
        y = self.cv2(y)
        return x + y if self.add else y


# --- Spatial-Channel Decoupled Downsampling (SCDown) ---
# Paper Reference: Table 2
class SCDown(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = DWConv(c2, c2, k, s, act=False) # Depthwise

    def forward(self, x):
        return self.cv2(self.cv1(x))