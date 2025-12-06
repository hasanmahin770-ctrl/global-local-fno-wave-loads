"""Align spectral data with load time-series."""

import numpy as np
import pandas as pd

class TimeAligner:
    """Time-align spectra with device loads."""
    
    def __init__(self, tolerance_hours=1):
        self.tolerance_hours = tolerance_hours
    
    def align(
        self,
        spec_times,  # datetime index for spectra
        spec_data,   # spectrum array (time, freq, dir)
        load_times,  # datetime index for loads
        load_data,   # load array (time, load_components)
    ):
        """
        Match spectrum to nearest load timestamp.
        """
        spec_times = pd.to_datetime(spec_times)
        load_times = pd.to_datetime(load_times)
        
        aligned_spec = []
        aligned_load = []
        valid_idx = []
        
        for i, st in enumerate(spec_times):
            # Find nearest load time
            dt = (load_times - st).abs()
            nearest_idx = dt.argmin()
            
            # Check if within tolerance
            if dt.iloc[nearest_idx].total_seconds() / 3600 <= self.tolerance_hours:
                aligned_spec.append(spec_data[i])
                aligned_load.append(load_data[nearest_idx])
                valid_idx.append(i)
        
        return (
            np.array(aligned_spec),
            np.array(aligned_load),
            np.array(valid_idx)
        )
