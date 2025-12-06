"""Physics-regularized loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DataLoss(nn.Module):
    """Supervised MSE loss."""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.mse(pred, target)


class EnergyResidualLoss(nn.Module):
    """Energy conservation loss."""
    
    def forward(self, pred_spectrum, target_spectrum):
        # Integral (energy) should be conserved
        pred_energy = pred_spectrum.sum(dim=(-2, -1))
        target_energy = target_spectrum.sum(dim=(-2, -1))
        return F.mse_loss(pred_energy, target_energy)


class MomentumResidualLoss(nn.Module):
    """Momentum balance loss."""
    
    def forward(self, pred_loads, target_loads):
        # Mean loads should match
        pred_mean = pred_loads.mean(dim=1)
        target_mean = target_loads.mean(dim=1)
        return F.mse_loss(pred_mean, target_mean)


class TailWeightedLoss(nn.Module):
    """Extreme-event-aware loss with tail weighting."""
    
    def __init__(self, alpha=1.5, percentile=90):
        super().__init__()
        self.alpha = alpha
        self.percentile = percentile
    
    def forward(self, pred, target, hs_values):
        # Compute per-sample weights
        threshold = torch.quantile(hs_values, self.percentile / 100.0)
        weights = torch.where(
            hs_values >= threshold,
            torch.ones_like(hs_values) * self.alpha,
            torch.ones_like(hs_values)
        )
        
        # Weighted MSE
        mse = F.mse_loss(pred, target, reduction='none')
        weighted_mse = (mse * weights[:, np.newaxis]).mean()
        
        return weighted_mse


class PhysicsRegularizedLoss(nn.Module):
    """Combined physics-regularized loss."""
    
    def __init__(
        self,
        lambda_energy=0.1,
        lambda_momentum=0.05,
        lambda_reg=1e-4,
        alpha_tail=1.5,
        percentile_tail=90,
    ):
        super().__init__()
        self.data_loss = DataLoss()
        self.energy_loss = EnergyResidualLoss()
        self.momentum_loss = MomentumResidualLoss()
        self.tail_loss = TailWeightedLoss(alpha=alpha_tail, percentile=percentile_tail)
        
        self.lambda_energy = lambda_energy
        self.lambda_momentum = lambda_momentum
        self.lambda_reg = lambda_reg
    
    def forward(
        self,
        pred,
        target,
        hs_values=None,
        model=None,
    ):
        """
        Parameters
        ----------
        pred : Tensor
            Predictions
        target : Tensor
            Targets
        hs_values : Tensor, optional
            Hs for tail weighting
        model : nn.Module, optional
            Model for weight regularization
        """
        
        # Data loss with tail weighting
        if hs_values is not None:
            data_loss = self.tail_loss(pred, target, hs_values)
        else:
            data_loss = self.data_loss(pred, target)
        
        # Energy and momentum residuals
        energy_loss = self.energy_loss(pred, target)
        momentum_loss = self.momentum_loss(pred, target)
        
        # Weight regularization
        reg_loss = 0.0
        if model is not None:
            for param in model.parameters():
                reg_loss = reg_loss + param.pow(2).sum()
            reg_loss = self.lambda_reg * reg_loss
        
        # Combined loss
        total_loss = (
            data_loss +
            self.lambda_energy * energy_loss +
            self.lambda_momentum * momentum_loss +
            reg_loss
        )
        
        return total_loss
