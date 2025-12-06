"""Base Fourier Neural Operator implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class FNOBlock(nn.Module):
    """Single FNO layer: Fourier spectral convolution + linear lift."""
    
    def __init__(self, modes, width, activation='gelu'):
        """
        Parameters
        ----------
        modes : tuple
            (modes_x, modes_y) = number of low-frequency modes to keep
        width : int
            Channel width
        activation : str
            Activation function
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.activation = getattr(F, activation)
        
        # Learnable Fourier coefficients (complex-valued)
        self.scale = nn.Parameter(
            torch.zeros(modes[0], modes[1], width, width, dtype=torch.complex64)
        )
        nn.init.uniform_(self.scale.real, -1.0, 1.0)
        nn.init.uniform_(self.scale.imag, -1.0, 1.0)
        
        # Skip connection weight
        self.linear = nn.Linear(width, width)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, width, n_freq, n_dir)
        
        Returns
        -------
        out : torch.Tensor
            Same shape as x
        """
        batch_size = x.shape[0]
        
        # Apply Fourier transform
        x_ft = torch.fft.rfft2d(x, dim=(-2, -1))
        
        # Multiply by low-frequency modes
        m1, m2 = self.modes
        x_ft_filtered = x_ft[:, :, :m1, :m2]
        
        # Spectral convolution
        out_ft = torch.einsum('bwij,wij mn->bmnij', x_ft_filtered, self.scale)
        
        # Zero-pad back to original size
        out_ft_padded = torch.zeros_like(x_ft)
        out_ft_padded[:, :, :m1, :m2] = out_ft
        
        # Inverse FFT
        out = torch.fft.irfft2d(out_ft_padded, s=x.shape[-2:], dim=(-2, -1))
        
        # Skip connection
        skip = self.linear(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = out + skip
        out = self.activation(out)
        
        return out


class FNO(nn.Module):
    """Fourier Neural Operator for 2D spectral inputs."""
    
    def __init__(
        self,
        modes=(16, 16),
        width=96,
        depth=4,
        input_channels=1,
        output_channels=1,
        activation='gelu',
    ):
        """
        Parameters
        ----------
        modes : tuple
            (modes_freq, modes_dir)
        width : int
            Channel width
        depth : int
            Number of FNO blocks
        input_channels : int
        output_channels : int
        activation : str
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        
        # Input lift: (Nf, Ndir) -> (width, Nf, Ndir)
        self.fc_in = nn.Linear(input_channels, width)
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(modes, width, activation)
            for _ in range(depth)
        ])
        
        # Output projection: (width, Nf, Ndir) -> (output_channels, Nf, Ndir)
        self.fc_out1 = nn.Linear(width, 128)
        self.fc_out2 = nn.Linear(128, output_channels)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, freq_bins, dir_bins) or (batch, channels, freq_bins, dir_bins)
        
        Returns
        -------
        out : torch.Tensor
            Shape (batch, output_channels, freq_bins, dir_bins)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim
        
        batch_size, _, nf, nd = x.shape
        
        # Reshape for linear layer
        x = x.permute(0, 2, 3, 1)  # (B, Nf, Ndir, C)
        x = self.fc_in(x)           # (B, Nf, Ndir, width)
        x = x.permute(0, 3, 1, 2)   # (B, width, Nf, Ndir)
        
        # Apply FNO blocks
        for fno_layer in self.fno_layers:
            x = fno_layer(x)
        
        # Project to output
        x = x.permute(0, 2, 3, 1)   # (B, Nf, Ndir, width)
        x = self.fc_out1(x)
        x = F.gelu(x)
        x = self.fc_out2(x)          # (B, Nf, Ndir, output_channels)
        x = x.permute(0, 3, 1, 2)   # (B, output_channels, Nf, Ndir)
        
        return x
