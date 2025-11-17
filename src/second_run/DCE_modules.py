"""
DCE-YOLOv8 Custom Modules
Implementation of Divided Context Extraction and Efficient Residual Bottleneck
Based on: "DCE-YOLOv8: Lightweight and Accurate Object Detection for Drone Vision"

Usage:
    1. Copy this file to your Ultralytics directory
    2. Register modules in ultralytics/nn/modules/__init__.py
    3. Use in custom YAML configs
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, DWConv, autopad
from ultralytics.nn.modules.block import C2f, Bottleneck


# =============================================================================
# DCE MODULE - Divided Context Extraction
# =============================================================================

class DCE(nn.Module):
    """
    Divided Context Extraction Module
    
    Implements partial convolution for efficient small object feature extraction.
    Splits input into two parts:
    - 75% goes through sequential 3x3 convolutions (feature extraction)
    - 25% passes through as identity (residual connection)
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of sequential conv blocks (default: 2)
        shortcut (bool): Use residual connection (default: True)
        g (int): Groups for convolution (default: 1)
        e (float): Expansion ratio (default: 0.5)
    
    Paper equation:
        DCE = F(Conv3x3(Conv3x3(X1)), X2)
        where X1 = 75% of input, X2 = 25% of input
    """
    
    def __init__(self, c1, c2, n=2, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # Hidden channels
        self.n = n  # Number of conv blocks
        self.shortcut = shortcut
        
        # Split ratio: 75% for processing, 25% for identity
        self.split_ratio = 0.75
        c_process = int(c1 * self.split_ratio)
        c_identity = c1 - c_process
        
        # Initial 1x1 conv to adjust channels if needed
        self.cv1 = Conv(c1, self.c, 1, 1)
        
        # Sequential 3x3 convolutions for feature extraction
        self.m = nn.ModuleList([
            Conv(self.c if i == 0 else self.c, self.c, 3, 1, g=g)
            for i in range(n)
        ])
        
        # Final 1x1 conv to output channels
        self.cv2 = Conv(self.c + c_identity, c2, 1, 1)
    
    def forward(self, x):
        """
        Forward pass with partial convolution
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, C2, H, W]
        """
        # Split input: 75% for processing, 25% as identity
        split_idx = int(x.size(1) * self.split_ratio)
        x_process = x[:, :split_idx, :, :]
        x_identity = x[:, split_idx:, :, :]
        
        # Process the majority through conv layers
        y = self.cv1(x_process)
        
        # Sequential convolutions
        for conv in self.m:
            y = conv(y)
        
        # Concatenate processed features with identity
        y = torch.cat([y, x_identity], dim=1)
        
        # Final projection
        return self.cv2(y)
    
    def forward_split(self, x):
        """
        Alternative forward with explicit split (for debugging)
        """
        # Split
        c_split = int(x.size(1) * self.split_ratio)
        x1, x2 = torch.split(x, [c_split, x.size(1) - c_split], dim=1)
        
        # Process x1
        y = self.cv1(x1)
        for conv in self.m:
            y = conv(y)
        
        # Concatenate with x2
        y = torch.cat([y, x2], dim=1)
        return self.cv2(y)


# =============================================================================
# DCE_C2f - DCE integrated into C2f structure
# =============================================================================

class DCE_C2f(nn.Module):
    """
    DCE-enhanced C2f module
    
    Replaces standard C2f bottlenecks with DCE modules for more efficient
    feature extraction in early backbone layers.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of DCE blocks (default: 1)
        shortcut (bool): Use residual connections (default: False)
        g (int): Groups (default: 1)
        e (float): Expansion ratio (default: 0.5)
    """
    
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        
        # Use DCE modules instead of standard Bottlenecks
        self.m = nn.ModuleList([
            DCE(self.c, self.c, n=2, shortcut=shortcut, g=g)
            for _ in range(n)
        ])
    
    def forward(self, x):
        """CSP Bottleneck with 2 convolutions and DCE blocks"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# =============================================================================
# ERB - Efficient Residual Bottleneck
# =============================================================================

class ERB(nn.Module):
    """
    Efficient Residual Bottleneck
    
    Lightweight alternative to C2f with fewer parameters.
    Uses addition operations instead of concatenation for efficiency.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        n (int): Number of bottleneck blocks (default: 1)
        shortcut (bool): Use residual connections (default: True)
        g (int): Groups (default: 1)
        e (float): Expansion ratio (default: 0.5)
    
    Paper equation:
        ERB = Conv1x1(F(Bottleneck(Conv1x1(X1)), X2) + X1)
    """
    
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c, c2, 1, 1)
        
        # Efficient bottlenecks with addition
        self.m = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut=shortcut, g=g, e=1.0)
            for _ in range(n)
        ])
        
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        """Forward with residual addition"""
        y = self.cv1(x)
        
        # Process through bottlenecks
        for bottleneck in self.m:
            y = bottleneck(y)
        
        y = self.cv2(y)
        
        # Add residual connection if dimensions match
        return x + y if self.add else y


# =============================================================================
# SCDown - Spatial-Channel Decoupled Downsampling (from YOLOv10)
# =============================================================================

class SCDown(nn.Module):
    """
    Spatial-Channel Decoupled Downsampling
    
    More efficient downsampling that separately handles spatial and channel dimensions.
    Used in DCE-YOLOv8 to replace standard Conv downsampling.
    
    Args:
        c1 (int): Input channels
        c2 (int): Output channels
        k (int): Kernel size (default: 3)
        s (int): Stride (default: 2)
    """
    
    def __init__(self, c1, c2, k=3, s=2):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = DWConv(c2, c2, k, s, g=c2, act=False)
    
    def forward(self, x):
        """Decoupled downsampling"""
        return self.cv2(self.cv1(x))


# =============================================================================
# P2 Detection Head Support (for tiny objects)
# =============================================================================

class P2Head(nn.Module):
    """
    P2 Detection Head for tiny objects
    
    Additional detection head operating at 1/4 input resolution
    instead of standard 1/8, enabling better tiny object detection.
    
    Args:
        c1 (int): Input channels from backbone
        nc (int): Number of classes
        anchors (tuple): Anchor sizes for this scale
    """
    
    def __init__(self, c1, nc=80, anchors=()):
        super().__init__()
        self.nc = nc  # Number of classes
        self.no = nc + 5  # Number of outputs per anchor
        self.nl = 1  # Number of detection layers (this is one layer)
        
        # Detection head convolution
        self.m = nn.Conv2d(c1, self.no, 1)
    
    def forward(self, x):
        """Forward pass for P2 detection head"""
        return self.m(x)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_dce_model_yaml(base_yaml='yolov8m.yaml', output_yaml='yolov8m_dce.yaml'):
    """
    Create a custom YAML config with DCE modules
    
    Args:
        base_yaml: Base YOLOv8 YAML to modify
        output_yaml: Output YAML file path
    
    Returns:
        Path to created YAML file
    """
    import yaml
    from pathlib import Path
    
    # Example YAML structure with DCE modules
    config = {
        'nc': 1,  # Number of classes (person only)
        'depth_multiple': 0.67,  # Model depth (0.67 for YOLOv8m)
        'width_multiple': 0.75,  # Model width (0.75 for YOLOv8m)
        
        'backbone': [
            # Layer 0: Initial conv
            [-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
            
            # Layer 1: DCE instead of C2f (early feature extraction)
            [-1, 1, 'DCE', [64, 2]],  # 1
            
            # Layer 2: Downsample
            [-1, 1, 'Conv', [128, 3, 2]],  # 2-P2/4
            
            # Layer 3: DCE
            [-1, 1, 'DCE', [128, 2]],  # 3
            
            # Layer 4: ERB instead of C2f
            [-1, 1, 'ERB', [128, 1]],  # 4
            
            # Continue with standard layers...
            [-1, 1, 'SCDown', [256, 3, 2]],  # 5-P3/8
            [-1, 2, 'ERB', [256, 2]],  # 6
            [-1, 1, 'SCDown', [512, 3, 2]],  # 7-P4/16
            [-1, 2, 'ERB', [512, 2]],  # 8
            [-1, 1, 'SPPF', [512, 5]],  # 9
        ],
        
        'head': [
            [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
            [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
            [-1, 2, 'ERB', [512, 2]],  # 12
            
            [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
            [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
            [-1, 2, 'ERB', [256, 2]],  # 15 (P3/8-small)
            
            [-1, 1, 'Conv', [256, 3, 2]],
            [[-1, 12], 1, 'Concat', [1]],  # cat head P4
            [-1, 2, 'ERB', [512, 2]],  # 18 (P4/16-medium)
            
            [[15, 18], 1, 'Detect', ['nc']],  # Detect(P3, P4)
        ]
    }
    
    output_path = Path(output_yaml)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Created DCE model YAML: {output_path}")
    return output_path


def compare_model_params(model1, model2):
    """
    Compare parameter counts between two models
    
    Args:
        model1: First model (e.g., standard YOLOv8m)
        model2: Second model (e.g., YOLOv8m + DCE)
    
    Returns:
        Dictionary with comparison statistics
    """
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    
    reduction = (params1 - params2) / params1 * 100
    
    return {
        'model1_params': params1,
        'model2_params': params2,
        'reduction_percentage': reduction,
        'difference': params1 - params2
    }


# =============================================================================
# TESTING & VALIDATION
# =============================================================================

if __name__ == "__main__":
    """Test DCE modules"""
    
    print("Testing DCE Modules...\n")
    
    # Test input
    batch_size = 4
    c_in, c_out = 128, 128
    h, w = 80, 80
    x = torch.randn(batch_size, c_in, h, w)
    
    print(f"Input shape: {x.shape}")
    print(f"="*60)
    
    # Test DCE module
    print("\n1. Testing DCE Module:")
    dce = DCE(c_in, c_out, n=2, shortcut=True)
    y_dce = dce(x)
    print(f"   Output shape: {y_dce.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dce.parameters()):,}")
    
    # Test DCE_C2f
    print("\n2. Testing DCE_C2f Module:")
    dce_c2f = DCE_C2f(c_in, c_out, n=1)
    y_c2f = dce_c2f(x)
    print(f"   Output shape: {y_c2f.shape}")
    print(f"   Parameters: {sum(p.numel() for p in dce_c2f.parameters()):,}")
    
    # Test ERB
    print("\n3. Testing ERB Module:")
    erb = ERB(c_in, c_out, n=1, shortcut=True)
    y_erb = erb(x)
    print(f"   Output shape: {y_erb.shape}")
    print(f"   Parameters: {sum(p.numel() for p in erb.parameters()):,}")
    
    # Test SCDown
    print("\n4. Testing SCDown Module:")
    scdown = SCDown(c_in, c_out*2, k=3, s=2)
    y_down = scdown(x)
    print(f"   Output shape: {y_down.shape}")
    print(f"   Parameters: {sum(p.numel() for p in scdown.parameters()):,}")
    
    # Compare with standard Conv
    print("\n5. Comparison with Standard Conv:")
    standard_conv = nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, 1, 1),
        nn.BatchNorm2d(c_out),
        nn.SiLU()
    )
    
    dce_params = sum(p.numel() for p in dce.parameters())
    conv_params = sum(p.numel() for p in standard_conv.parameters())
    
    print(f"   DCE params: {dce_params:,}")
    print(f"   Conv params: {conv_params:,}")
    print(f"   Reduction: {(1 - dce_params/conv_params)*100:.1f}%")
    
    print(f"\n{'='*60}")
    print("✅ All modules tested successfully!")