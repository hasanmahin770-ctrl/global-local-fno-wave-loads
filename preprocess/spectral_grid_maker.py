"""Resample directional wave spectra onto fixed grids."""

import numpy as np
import xarray as xr
from pathlib import Path

class SpectralGridMaker:
    """Convert variable spectral grids to fixed Nf x Ndir arrays."""
    
    def __init__(self, n_freq=64, n_dir=64):
        self.n_freq = n_freq
        self.n_dir = n_dir
    
    def process_netcdf(self, filepath, output_filepath):
        """Load netCDF spectrum, resample, normalize."""
        ds = xr.open_dataset(filepath)
        
        # Extract spectral density (assume Copernicus format)
        spec_var = [v for v in ds.data_vars if 'spectrum' in v.lower()][0]
        spec = ds[spec_var].values
        
        # Resample to fixed grid
        spec_grid = self._resample_to_grid(spec)
        
        # Normalize
        spec_norm = self._normalize(spec_grid)
        
        # Save as netCDF
        ds_out = xr.Dataset(
            {'spectrum': (('freq', 'dir'), spec_norm)},
            coords={'freq': np.arange(self.n_freq), 'dir': np.arange(self.n_dir)}
        )
        ds_out.to_netcdf(output_filepath)
        
        return spec_norm
    
    def _resample_to_grid(self, spec):
        """Resample to fixed frequency-direction grid."""
        # Simple bilinear interpolation
        from scipy.interpolate import interp2d
        
        # Assume input is (time, freq, dir) or (freq, dir)
        if spec.ndim == 3:
            spec = spec[0]  # Take first time step
        
        # Create interpolator
        freq_old = np.arange(spec.shape[0])
        dir_old = np.arange(spec.shape[1])
        
        freq_new = np.linspace(0, spec.shape[0]-1, self.n_freq)
        dir_new = np.linspace(0, spec.shape[1]-1, self.n_dir)
        
        f = interp2d(dir_old, freq_old, spec)
        spec_resampled = f(dir_new, freq_new)
        
        return spec_resampled
    
    def _normalize(self, spec):
        """Normalize spectrum to [0, 1]."""
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max > spec_min:
            return (spec - spec_min) / (spec_max - spec_min)
        return spec
