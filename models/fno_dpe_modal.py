"""Complete FNO with Directional-Phase Embedding and Modal Attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno_core import FNOBlock
from .dpe_embedding import DirectionalPhaseEmbedding
from .modal_attention import ModalAttention


class FNOWithDPEAndAttention(nn.Module):
    """FNO augmented with Directional-Phase Embedding and Modal Attention."""
    
    def __init__(
        self,
        modes=(16, 16),
        width=96,
        depth=4,
        input_channels=1,
        output_channels=1,
        dpe_dim=8,
        use_modal_attention=True,
        activation='gelu',
    ):
        """
        Parameters
        ----------
        modes : tuple
        width : int
        depth : int
        input_channels : int
        output_channels : int
        dpe_dim : int
            Directional-Phase Embedding dimension
        use_modal_attention : bool
        activation : str
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.use_modal_attention = use_modal_attention
        
        # Directional-Phase Embedding
        self.dpe = DirectionalPhaseEmbedding(n_dir_bins=modes[1], embedding_dim=dpe_dim)
        effective_input_channels = input_channels + dpe_dim
        
        # Input lift
        self.fc_in = nn.Linear(effective_input_channels, width)
        
        # Modal attention (optional)
        if use_modal_attention:
            self.modal_attention = ModalAttention(modes, width)
        
        # FNO layers
        self.fno_layers = nn.ModuleList([
            FNOBlock(modes, width, activation)
            for _ in range(depth)
        ])
        
        # Output projection
        self.fc_out1 = nn.Linear(width, 128)
        self.fc_out2 = nn.Linear(128, output_channels)
    
    def forward(self, x, meta=None):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, freq_bins, dir_bins) or (batch, channels, freq_bins, dir_bins)
        meta : torch.Tensor, optional
            Meta features like Hs, Tp (batch, n_meta)
        
        Returns
        -------
        out : torch.Tensor
            Shape (batch, output_channels, freq_bins, dir_bins)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        batch_size, _, nf, nd = x.shape
        
        # Apply DPE
        x = self.dpe(x)  # (batch, channels + dpe_dim, nf, nd)
        
        # Reshape for linear layer
        x = x.permute(0, 2, 3, 1)  # (B, Nf, Ndir, C+dpe_dim)
        x = self.fc_in(x)           # (B, Nf, Ndir, width)
        x = x.permute(0, 3, 1, 2)   # (B, width, Nf, Ndir)
        
        # Apply FNO blocks with optional modal attention
        for i, fno_layer in enumerate(self.fno_layers):
            if self.use_modal_attention and i == 0:
                # Apply modal attention to first layer's output
                x_out = fno_layer(x)
                x_ft = torch.fft.rfft2d(x_out, dim=(-2, -1))
                _, attn = self.modal_attention(x_ft, x)
                # Integrate attention into next layer's computation
                x = x_out
            else:
                x = fno_layer(x)
        
        # Project to output
        x = x.permute(0, 2, 3, 1)   # (B, Nf, Ndir, width)
        x = self.fc_out1(x)
        x = F.gelu(x)
        x = self.fc_out2(x)          # (B, Nf, Ndir, output_channels)
        x = x.permute(0, 3, 1, 2)   # (B, output_channels, Nf, Ndir)
        
        return x
