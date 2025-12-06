"""Generate synthetic RAO (Response Amplitude Operator) kernels."""

import numpy as np
from scipy import signal

class RAOGenerator:
    """Frequency-domain RAO generator for different device geometries."""
    
    def __init__(self, freq_min=0.04, freq_max=1.0, n_freq=64, n_dir=64):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.n_freq = n_freq
        self.n_dir = n_dir
        
        self.freq = np.linspace(freq_min, freq_max, n_freq)
        self.dir = np.linspace(0, 2*np.pi, n_dir)
    
    def generate_rao(
        self,
        device_type='cylinder',
        draft=5.0,  # meters
        diameter=4.0,  # meters
        mass=1000.0,  # kg
        stiffness=1e5,  # N/m
        damping=0.05,  # damping ratio
    ):
        """
        Generate RAO for a device.
        
        Parameters
        ----------
        device_type : str
            'cylinder', 'box', 'sphere'
        draft : float
        diameter : float
        mass : float
        stiffness : float
        damping : float
        
        Returns
        -------
        rao : ndarray
            (n_freq, n_dir) response amplitude operator
        """
        
        # Natural frequency
        omega_n = np.sqrt(stiffness / mass)
        
        # Frequency response (single DOF)
        omega = 2 * np.pi * self.freq
        H_mag = np.zeros_like(omega)
        
        for i, w in enumerate(omega):
            denominator = np.sqrt((omega_n**2 - w**2)**2 + (2*damping*omega_n*w)**2)
            H_mag[i] = 1.0 / max(denominator, 1e-6)
        
        # Apply directional spreading (cosine^2 weighting)
        rao = H_mag[:, np.newaxis] * np.cos(self.dir[np.newaxis, :])**2
        
        return rao
    
    def spectrum_to_loads(
        self,
        spectrum,  # (n_freq, n_dir)
        rao,
        time_window=1800,  # seconds
        fs=2.0,  # Hz
    ):
        """
        Apply RAO to wave spectrum to generate loads.
        
        Parameters
        ----------
        spectrum : ndarray (n_freq, n_dir)
        rao : ndarray (n_freq, n_dir)
        time_window : float
            Duration of load time-series
        fs : float
            Sampling frequency
        
        Returns
        -------
        loads : ndarray (n_samples,)
            Time-series loads
        """
        
        # Spectral moment
        m0 = (spectrum * rao**2).sum()
        
        # Generate Gaussian random process with spectral shape
        t = np.arange(0, time_window, 1/fs)
        loads = np.random.normal(0, np.sqrt(m0), len(t))
        
        return loads
