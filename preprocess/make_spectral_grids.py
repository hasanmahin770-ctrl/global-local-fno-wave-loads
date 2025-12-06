#!/usr/bin/env python3
"""Preprocess pipeline: convert raw spectra to fixed grids."""

import argparse
from pathlib import Path
from spectral_grid_maker import SpectralGridMaker

def main(emec_buoy_dir, copernicus_dir, output_dir, grid_size=(64, 64), normalize=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    maker = SpectralGridMaker(n_freq=grid_size[0], n_dir=grid_size[1])
    
    # Process EMEC
    emec_dir = Path(emec_buoy_dir)
    if emec_dir.exists():
        for nc_file in emec_dir.glob('*.nc'):
            print(f"Processing {nc_file}...")
            maker.process_netcdf(str(nc_file), str(output_dir / f"{nc_file.stem}_gridded.nc"))
    
    # Process Copernicus
    cop_dir = Path(copernicus_dir)
    if cop_dir.exists():
        for nc_file in cop_dir.glob('*.nc'):
            print(f"Processing {nc_file}...")
            maker.process_netcdf(str(nc_file), str(output_dir / f"{nc_file.stem}_gridded.nc"))
    
    print(f"\nPreprocessing complete. Output: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emec_buoy_dir", default="data/raw/emec/")
    parser.add_argument("--copernicus_dir", default="data/raw/copernicus/")
    parser.add_argument("--output_dir", default="data/processed/")
    parser.add_argument("--grid_size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--normalize", action="store_true")
    
    args = parser.parse_args()
    main(**vars(args))
