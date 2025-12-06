#!/usr/bin/env python3
"""
Download directional wave spectra from Copernicus Marine Service.
Requires free registration and API credentials.
"""

import os
import argparse
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
import sys

try:
    from copernicusmarine import Copernicusmarine
except ImportError:
    print("ERROR: copernicusmarine not installed. Install with: pip install copernicusmarine")
    sys.exit(1)


def download_copernicus_spectra(
    dataset,
    lon_min, lon_max,
    lat_min, lat_max,
    start_date, end_date,
    output_dir,
    verbose=True
):
    """
    Download directional wave spectra from Copernicus.
    
    Parameters
    ----------
    dataset : str
        Dataset name, e.g. 'GLOBAL_REANALYSIS_WAV_001_032'
    lon_min, lon_max, lat_min, lat_max : float
        Bounding box for region
    start_date, end_date : str
        ISO date format 'YYYY-MM-DD'
    output_dir : str
        Where to save netCDF files
    verbose : bool
        Print progress messages
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Downloading {dataset}")
        print(f"  Region: [{lon_min}, {lon_max}] x [{lat_min}, {lat_max}]")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Output: {output_dir}")
    
    try:
        cml = Copernicusmarine()
        
        ds = cml.subset(
            dataset_id=dataset,
            variables=[
                "sea_surface_wave_directional_variance_spectral_density",
                "sea_surface_wave_significant_height",
                "sea_surface_wave_peak_period",
                "sea_surface_wave_mean_direction",
            ],
            minimum_longitude=lon_min,
            maximum_longitude=lon_max,
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=start_date,
            end_datetime=end_date,
            force_download=False,
        )
        
        output_file = output_dir / f"{dataset}_{start_date}_{end_date}.nc"
        ds.to_netcdf(output_file)
        
        if verbose:
            print(f"\nSuccess! Saved to {output_file}")
            print(f"  Shape: {dict(ds.dims)}")
            print(f"  Size: {output_file.stat().st_size / 1e6:.1f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"\nERROR downloading from Copernicus:")
        print(f"  {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Register at https://data.marine.copernicus.eu/")
        print(f"  2. Save credentials to ~/.copernicusmarine_config")
        print(f"  3. Check username/password are correct")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download directional wave spectra from Copernicus Marine"
    )
    parser.add_argument(
        "--dataset",
        default="GLOBAL_REANALYSIS_WAV_001_032",
        help="Dataset ID (default: GLOBAL_REANALYSIS_WAV_001_032)"
    )
    parser.add_argument("--lon_min", type=float, default=-3.5)
    parser.add_argument("--lon_max", type=float, default=-2.8)
    parser.add_argument("--lat_min", type=float, default=59.0)
    parser.add_argument("--lat_max", type=float, default=59.5)
    parser.add_argument(
        "--start_date",
        default="2022-06-01",
        help="Start date (ISO format YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        default="2022-09-01",
        help="End date (ISO format YYYY-MM-DD)"
    )
    parser.add_argument("--output_dir", default="data/raw/copernicus/")
    
    args = parser.parse_args()
    
    download_copernicus_spectra(
        dataset=args.dataset,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
