"""Modal Attention mechanism for learning per-mode importance in FNO."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalAttention(nn.Module):
    """Learn scalar attention weights for each Fourier mode."""
    
    def __init__(self, modes, width, num_heads=4):
        """
        Parameters
        ----------
        modes : tuple
            (modes_freq, modes_dir) - number of kept Fourier modes
        width : int
            Channel width
        num_heads : int
            Number of attention heads
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_heads = num_heads
        
        # Query, Key, Value projections for each mode
        self.mode_embed_dim = 16
        self.q_proj = nn.Linear(4, self.mode_embed_dim)  # [||k||^2, k_x, k_y, freq_norm]
        self.k_proj = nn.Linear(4, self.mode_embed_dim)
        self.v_proj = nn.Linear(width, width)
        
        # Output projection
        self.out_proj = nn.Linear(width, width)
        
    def forward(self, x_ft, x_spatial):
        """
        Apply modal attention to Fourier coefficients.
        
        Parameters
        ----------
        x_ft : torch.Tensor
            Fourier coefficients (batch, width, modes_freq, modes_dir)
        x_spatial : torch.Tensor
            Spatial domain input for context (batch, width, nf, nd)
        
        Returns
        -------
        out : torch.Tensor
            Attention-weighted Fourier coefficients
        """
        batch, width, mf, md = x_ft.shape
        
        # Create mode descriptors: [||k||, k_freq, k_dir, normalized_energy]
        k_freq = torch.arange(mf, dtype=torch.float32, device=x_ft.device) / max(mf, 1)
        k_dir = torch.arange(md, dtype=torch.float32, device=x_ft.device) / max(md, 1)
        kk_freq, kk_dir = torch.meshgrid(k_freq, k_dir, indexing='ij')
        
        k_norm = torch.sqrt(kk_freq**2 + kk_dir**2 + 1e-6)
        
        # Mode features: (mf*md, 4)
        mode_features = torch.stack([
            k_norm.flatten(),
            kk_freq.flatten(),
            kk_dir.flatten(),
            x_ft.abs().mean(dim=(0, 1)).flatten()  # Energy per mode
        ], dim=1)
        
        # Compute queries and keys
        q = self.q_proj(mode_features)  # (mf*md, embedding_dim)
        k = self.k_proj(mode_features)
        
        # Attention scores
        scores = torch.matmul(q, k.t()) / (self.mode_embed_dim ** 0.5)  # (mf*md, mf*md)
        attn_weights = F.softmax(scores.mean(dim=0), dim=-1)  # (mf*md,)
        attn_weights = attn_weights.view(1, 1, mf, md)
        
        # Apply attention to coefficients
        out = x_ft * attn_weights
        
        return out, attn_weights
