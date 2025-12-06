"""Extract meta features from wave spectra."""

import numpy as np

class MetaFeatureBuilder:
    """Compute integrated wave parameters from directional spectrum."""
    
    def __init__(self, freq_edges=None, dir_edges=None):
        """Initialize with frequency and direction bins."""
        self.freq_edges = freq_edges or np.linspace(0.04, 1.0, 64)
        self.dir_edges = dir_edges or np.linspace(0, 2*np.pi, 64)
    
    def compute_hs(self, spectrum):
        """Significant wave height."""
        # Hs = 4 * sqrt(m0) where m0 = integral of spectrum
        m0 = spectrum.sum()
        return 4.0 * np.sqrt(m0 * self._bandwidth())
    
    def compute_tp(self, spectrum):
        """Peak period."""
        # Frequency of maximum spectral density
        freq_peak_idx = spectrum.sum(axis=1).argmax()
        return 1.0 / self.freq_edges[freq_peak_idx] if freq_peak_idx > 0 else 0.0
    
    def compute_dp(self, spectrum):
        """Mean wave direction."""
        # Directional moment
        S1 = (spectrum * np.sin(self.dir_edges[:, np.newaxis])).sum()
        S2 = (spectrum * np.cos(self.dir_edges[:, np.newaxis])).sum()
        return np.arctan2(S1, S2)
    
    def compute_spreading(self, spectrum):
        """Directional spreading."""
        # sqrt(1 - sqrt(S1^2 + S2^2) / M0)
        m0 = spectrum.sum()
        S1 = (spectrum * np.sin(self.dir_edges[:, np.newaxis])).sum()
        S2 = (spectrum * np.cos(self.dir_edges[:, np.newaxis])).sum()
        if m0 > 0:
            return np.sqrt(np.maximum(1 - np.sqrt(S1**2 + S2**2) / m0, 0))
        return 0.0
    
    def compute_all(self, spectrum):
        """Compute all meta features."""
        return {
            'Hs': self.compute_hs(spectrum),
            'Tp': self.compute_tp(spectrum),
            'Dp': self.compute_dp(spectrum),
            'spreading': self.compute_spreading(spectrum),
        }
    
    def _bandwidth(self):
        """Frequency bandwidth."""
        return (self.freq_edges[1] - self.freq_edges[0])
