"""Uncertainty quantification: ensembles and heteroscedastic outputs."""

import torch
import torch.nn as nn
from .fno_dpe_modal import FNOWithDPEAndAttention


class EnsembleFNO(nn.Module):
    """Ensemble of FNO models for uncertainty estimation."""
    
    def __init__(self, n_models=5, **fno_kwargs):
        """
        Parameters
        ----------
        n_models : int
            Number of models in ensemble
        **fno_kwargs
            Arguments for FNOWithDPEAndAttention
        """
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([
            FNOWithDPEAndAttention(**fno_kwargs)
            for _ in range(n_models)
        ])
    
    def forward(self, x, return_ensemble=False):
        """
        Parameters
        ----------
        x : torch.Tensor
        return_ensemble : bool
            If True, return all predictions
        
        Returns
        -------
        mean : torch.Tensor
        std : torch.Tensor
        (ensemble : list[Tensor], optional)
        """
        preds = [model(x) for model in self.models]
        preds = torch.stack(preds)  # (n_models, batch, out_channels, ...)
        
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        
        if return_ensemble:
            return mean, std, preds
        return mean, std


class HeteroscedasticFNO(nn.Module):
    """FNO with heteroscedastic output (mean + uncertainty per prediction)."""
    
    def __init__(self, output_channels=1, **fno_kwargs):
        """
        Parameters
        ----------
        output_channels : int
        **fno_kwargs
            FNO arguments
        """
        super().__init__()
        # Output 2x channels: mean + log_std
        self.fno = FNOWithDPEAndAttention(
            output_channels=output_channels * 2,
            **fno_kwargs
        )
        self.output_channels = output_channels
    
    def forward(self, x):
        """
        Returns
        -------
        mean : torch.Tensor
            Predicted mean
        std : torch.Tensor
            Predicted std (exp of learned log_std for stability)
        """
        out = self.fno(x)
        
        # Split into mean and log_std
        mean, log_std = out.chunk(2, dim=1)
        std = torch.exp(log_std.clamp(min=-10, max=2))  # Clamp for stability
        
        return mean, std
