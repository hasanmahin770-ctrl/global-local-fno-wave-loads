"""Baseline models: RAO, U-Net, CNN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAOBaseline(nn.Module):
    """Linear Frequency-Domain RAO Baseline."""
    
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Learnable RAO kernel
        self.rao_kernel = nn.Parameter(torch.ones(1, 1, 64, 64) * 0.5)
    
    def forward(self, x):
        # Simple multiplicative RAO
        out = x * self.rao_kernel
        return out


class UNetBaseline(nn.Module):
    """Simple U-Net baseline."""
    
    def __init__(self, input_channels=1, output_channels=1, features=32):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encoder
        self.enc1 = self._conv_block(input_channels, features)
        self.enc2 = self._conv_block(features, features * 2)
        self.enc3 = self._conv_block(features * 2, features * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(features * 4, features * 8)
        
        # Decoder
        self.dec3 = self._conv_block(features * 8 + features * 4, features * 4)
        self.dec2 = self._conv_block(features * 4 + features * 2, features * 2)
        self.dec1 = self._conv_block(features * 2 + features, features)
        
        # Output
        self.out = nn.Conv2d(features, output_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))
        
        # Decoder
        d3 = self.dec3(torch.cat([F.interpolate(b, e3.shape[-2:]), e3], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, e2.shape[-2:]), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, e1.shape[-2:]), e1], 1))
        
        out = self.out(d1)
        return out


class CNNBaseline(nn.Module):
    """Simple CNN baseline."""
    
    def __init__(self, input_channels=1, output_channels=1, features=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, output_channels, 1),
        )
    
    def forward(self, x):
        return self.features(x)
