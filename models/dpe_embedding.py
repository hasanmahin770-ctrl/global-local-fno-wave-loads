"""Directional-Phase Embedding (DPE) for spectral input encoding."""

import torch
import torch.nn as nn
import numpy as np


class DirectionalPhaseEmbedding(nn.Module):
    """Embed directional phase information e^{i*theta} into input."""
    
    def __init__(self, n_dir_bins, embedding_dim=8):
        """
        Parameters
        ----------
        n_dir_bins : int
            Number of directional bins
        embedding_dim : int
            Embedding dimension for phase encoding
        """
        super().__init__()
        self.n_dir_bins = n_dir_bins
        self.embedding_dim = embedding_dim
        
        # Create directional angles
        self.register_buffer(
            'theta',
            torch.linspace(0, 2*np.pi, n_dir_bins, dtype=torch.float32)
        )
        
        # Learnable phase embedding weights
        self.phase_proj = nn.Linear(2, embedding_dim)  # cos(theta), sin(theta) -> embedding
    
    def forward(self, x):
        """
        Append directional phase embedding to input spectrum.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, channels, n_freq, n_dir)
        
        Returns
        -------
        x_augmented : torch.Tensor
            Shape (batch, channels + embedding_dim, n_freq, n_dir)
        """
        batch_size, channels, n_freq, n_dir = x.shape
        
        # Create phase embeddings
        cos_theta = torch.cos(self.theta).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, n_dir)
        sin_theta = torch.sin(self.theta).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Project phase components
        phase_input = torch.cat([cos_theta, sin_theta], dim=1)  # (1, 2, 1, n_dir)
        phase_input = phase_input.expand(batch_size, 2, n_freq, n_dir)
        
        # Apply learnable projection
        phase_input = phase_input.permute(0, 2, 3, 1)  # (B, Nf, Ndir, 2)
        phase_embed = self.phase_proj(phase_input)      # (B, Nf, Ndir, embedding_dim)
        phase_embed = phase_embed.permute(0, 3, 1, 2)   # (B, embedding_dim, Nf, Ndir)
        
        # Concatenate with original spectrum
        x_augmented = torch.cat([x, phase_embed], dim=1)
        
        return x_augmented
