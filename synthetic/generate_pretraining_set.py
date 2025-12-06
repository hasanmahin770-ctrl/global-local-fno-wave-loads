#!/usr/bin/env python3
"""Generate 50k synthetic spectrum->load training pairs."""

import argparse
import numpy as np
from pathlib import Path
from rao_generator import RAOGenerator
from wave_sampler import WaveSampler

def generate_pretraining_set(n_samples=50000, seed=0, output_dir="data/synthetic/pretraining/"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(seed)
    
    sampler = WaveSampler(n_freq=64, n_dir=64, seed=seed)
    rao_gen = RAOGenerator(n_freq=64, n_dir=64)
    
    # Generate multiple device types
    device_configs = [
        {'type': 'cylinder', 'draft': 5, 'diameter': 4, 'mass': 500, 'stiffness': 1e5},
        {'type': 'box', 'draft': 4, 'diameter': 5, 'mass': 800, 'stiffness': 2e5},
        {'type': 'sphere', 'draft': 3, 'diameter': 6, 'mass': 600, 'stiffness': 1.5e5},
    ]
    
    spectra = []
    loads = []
    params = []
    
    print(f"Generating {n_samples} synthetic training samples...")
    
    for i in range(n_samples):
        if (i+1) % 10000 == 0:
            print(f"  {i+1}/{n_samples}")
        
        # Sample wave spectrum
        spectrum, hs, tp, dp = sampler.sample_spectrum()
        
        # Sample device configuration
        config = device_configs[i % len(device_configs)]
        
        # Generate RAO
        rao = rao_gen.generate_rao(
            device_type=config['type'],
            draft=config['draft'],
            diameter=config['diameter'],
            mass=config['mass'],
            stiffness=config['stiffness'],
        )
        
        # Generate loads
        load_ts = rao_gen.spectrum_to_loads(spectrum, rao)
        
        spectra.append(spectrum)
        loads.append(load_ts)
        params.append({'Hs': hs, 'Tp': tp, 'Dp': dp, 'device': config['type']})
    
    # Stack and save
    spectra = np.array(spectra)
    loads = np.array(loads)
    
    np.savez(
        output_dir / "pretraining_data.npz",
        spectra=spectra,
        loads=loads,
        params=np.array(params, dtype=object)
    )
    
    print(f"\nSaved to {output_dir / 'pretraining_data.npz'}")
    print(f"  Spectra shape: {spectra.shape}")
    print(f"  Loads shape: {loads.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="data/synthetic/pretraining/")
    
    args = parser.parse_args()
    generate_pretraining_set(**vars(args))
