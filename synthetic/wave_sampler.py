"""Random wave spectrum sampler for training data."""

import numpy as np
from scipy.stats import gamma, lognorm, uniform

class WaveSampler:
    """Sample random wave spectra from realistic distributions."""
    
    def __init__(self, n_freq=64, n_dir=64, seed=None):
        self.n_freq = n_freq
        self.n_dir = n_dir
        if seed is not None:
            np.random.seed(seed)
    
    def sample_spectrum(self):
        """
        Generate random directional wave spectrum.
        
        Returns
        -------
        spectrum : ndarray (n_freq, n_dir)
        hs, tp, dp, spreading : float
            Wave parameters
        """
        
        # Sample Hs (significant wave height) - lognormal distribution
        hs = lognorm.rvs(s=0.3, loc=0.5, scale=2.0)  # m
        hs = np.clip(hs, 0.2, 8.0)
        
        # Sample Tp (peak period) - gamma distribution
        tp = gamma.rvs(a=2, loc=2, scale=2)  # s
        tp = np.clip(tp, 3.0, 18.0)
        
        # Sample Dp (dominant direction) - uniform
        dp = uniform.rvs(0, 2*np.pi)
        
        # Create 2D spectrum
        freq = np.linspace(0.04, 1.0, self.n_freq)
        dir_bins = np.linspace(0, 2*np.pi, self.n_dir)
        
        # Jonswap spectrum in frequency
        fp = 1.0 / tp
        alpha = 0.0081  # JONSWAP alpha
        gamma_js = 3.3  # JONSWAP gamma
        
        spec_f = np.zeros(self.n_freq)
        for i, f in enumerate(freq):
            r = f / fp if f <= fp else (f / fp)
            sigma = 0.07 if f <= fp else 0.09
            spec_f[i] = (
                alpha * 9.81**2 / (2*np.pi)**4 / f**5 *
                np.exp(-1.25 * (fp/f)**4) *
                gamma_js ** np.exp(-(r-1)**2 / (2*sigma**2))
            )
        
        spec_f = spec_f / spec_f.max()  # Normalize
        
        # Directional spreading (cos^2 centered at dp)
        spreading = 0.1 + 0.2 * np.random.rand()  # spreading parameter
        dir_spread = (np.cos(dir_bins - dp)**2 + spreading * np.sin(dir_bins - dp)**2)
        dir_spread = dir_spread / dir_spread.max()
        
        # 2D spectrum
        spectrum = (hs**2 * spec_f[:, np.newaxis] * dir_spread[np.newaxis, :]) / hs**2
        
        return spectrum, hs, tp, dp
