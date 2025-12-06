"""Evaluation metrics for load predictions."""

import numpy as np
import torch
from scipy import stats

def rmse(pred, target):
    """Root Mean Squared Error."""
    return np.sqrt(np.mean((pred - target)**2))

def nrmse(pred, target):
    """Normalized RMSE (by target std)."""
    return rmse(pred, target) / (np.std(target) + 1e-6)

def mae(pred, target):
    """Mean Absolute Error."""
    return np.mean(np.abs(pred - target))

def peak_error_quantile(pred, target, q=0.95):
    """Error in peak predictions at quantile q."""
    threshold = np.quantile(target, q)
    mask = target >= threshold
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(pred[mask] - target[mask]))

def crps(pred_mean, pred_std, target):
    """Continuous Ranked Probability Score (Gaussian assumption)."""
    # Assuming Gaussian predictive distribution
    z = (target - pred_mean) / (pred_std + 1e-6)
    crps_vals = pred_std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps_vals)

def energy_residual(pred_spectrum, target_spectrum):
    """Energy conservation check."""
    pred_energy = pred_spectrum.sum()
    target_energy = target_spectrum.sum()
    return np.abs(pred_energy - target_energy) / (target_energy + 1e-6)

def momentum_residual(pred_loads, target_loads):
    """Momentum balance check."""
    pred_mean = pred_loads.mean()
    target_mean = target_loads.mean()
    return np.abs(pred_mean - target_mean) / (np.abs(target_mean) + 1e-6)

class MetricsCalculator:
    """Compute all metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_all(
        self,
        pred_mean,
        pred_std=None,
        target=None,
        pred_spectrum=None,
        target_spectrum=None,
    ):
        """Compute all metrics."""
        result = {
            'rmse': rmse(pred_mean, target),
            'nrmse': nrmse(pred_mean, target),
            'mae': mae(pred_mean, target),
            'peak_error_95': peak_error_quantile(pred_mean, target, q=0.95),
        }
        
        if pred_std is not None:
            result['crps'] = crps(pred_mean, pred_std, target)
        
        if pred_spectrum is not None and target_spectrum is not None:
            result['energy_residual'] = energy_residual(pred_spectrum, target_spectrum)
            result['momentum_residual'] = momentum_residual(pred_mean, target)
        
        return result
