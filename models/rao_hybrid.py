"""Hybrid: Linear RAO + FNO Residual decomposition."""

import torch
import torch.nn as nn
import numpy as np
from .fno_dpe_modal import FNOWithDPEAndAttention


class RAOHybridFNO(nn.Module):
    """Decompose prediction into linear RAO + learned nonlinear residual."""
    
    def __init__(
        self,
        modes=(16, 16),
        width=96,
        depth=4,
        input_channels=1,
        output_channels=1,
        rao_stiffness=1.0,
        rao_damping=0.1,
    ):
        """
        Parameters
        ----------
        modes : tuple
        width : int
        depth : int
        input_channels : int
        output_channels : int
        rao_stiffness : float
            Linear system stiffness (tunable)
        rao_damping : float
            Linear system damping ratio
        """
        super().__init__()
        self.modes = modes
        self.rao_stiffness = rao_stiffness
        self.rao_damping = rao_damping
        self.output_channels = output_channels
        
        # Linear RAO kernel (parametric)
        self.rao_kernel = self._make_rao_kernel(modes, rao_stiffness, rao_damping)
        
        # FNO residual operator
        self.fno_residual = FNOWithDPEAndAttention(
            modes=modes,
            width=width,
            depth=depth,
            input_channels=input_channels,
            output_channels=output_channels,
            use_modal_attention=True,
        )
    
    def _make_rao_kernel(self, modes, k, c):
        """Create parametric RAO kernel (simplified)."""
        mf, md = modes
        freq = torch.linspace(0.1, 2.0, mf)  # Hz
        direction = torch.linspace(0, 2*np.pi, md)
        
        # Single DOF response amplitude operator (simplified)
        omega = 2 * np.pi * freq  # rad/s
        omega_n = np.sqrt(k)       # natural frequency
        H_mag = 1.0 / (torch.sqrt((omega_n**2 - omega**2)**2 + (2*c*omega_n*omega)**2))
        
        # Apply directional spreading
        H = H_mag.unsqueeze(1) * torch.cos(direction.unsqueeze(0))**2
        
        return H.unsqueeze(0).unsqueeze(0)  # (1, 1, mf, md)
    
    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, channels, freq_bins, dir_bins)
        
        Returns
        -------
        out : torch.Tensor
            Linear RAO + learned residual
        """
        # Linear part: convolution with RAO kernel
        rao_kernel = self.rao_kernel.to(x.device)
        linear_response = x * rao_kernel
        
        # Nonlinear residual
        residual = self.fno_residual(x)
        
        # Combine
        out = linear_response + residual
        
        return out
