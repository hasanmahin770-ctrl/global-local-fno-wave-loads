#!/usr/bin/env python3
"""
Download EMEC Billia Croo buoy data (public spectra).
Device loads may require direct contact with EMEC.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

try:
    import requests
except ImportError:
    print("ERROR: requests not installed")
    sys.exit(1)


def download_emec_buoy(region="billia_croo", start_date="2022-06-01", end_date="2022-09-01", output_dir="data/raw/emec/"):
    """
    Download EMEC Billia Croo buoy public data.
    
    NOTE: Direct API access may vary. This is a template.
    For guaranteed data access, visit https://www.emec.org.uk/
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading EMEC {region} buoy data")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Output: {output_dir}")
    print()
    print("NOTE: EMEC data download may require manual steps.")
    print("Visit https://www.emec.org.uk/ for details.")
    print()
    
    # For now, create a placeholder with metadata
    metadata = {
        "site": region,
        "location": "Orkney, Scotland",
        "coordinates": {"lat": 59.35, "lon": -2.96},
        "start_date": start_date,
        "end_date": end_date,
        "note": "Public buoy spectra. Device loads require EMEC permission.",
    }
    
    metadata_file = output_dir / "metadata.txt"
    with open(metadata_file, "w") as f:
        for key, val in metadata.items():
            f.write(f"{key}: {val}\n")
    
    print(f"Created metadata file: {metadata_file}")
    print()
    print("TO ACCESS FULL DATA:")
    print("  1. Visit https://www.emec.org.uk/data-guide/")
    print("  2. Request access to Billia Croo buoy data")
    print("  3. For device loads: contact support@emec.org.uk")
    print()
    print("ALTERNATIVELY: Use synthetic RAO labels")
    print("  python synthetic/generate_pretraining_set.py --n_samples 50000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download EMEC Billia Croo buoy data")
    parser.add_argument("--region", default="billia_croo")
    parser.add_argument("--start_date", default="2022-06-01")
    parser.add_argument("--end_date", default="2022-09-01")
    parser.add_argument("--output_dir", default="data/raw/emec/")
    
    args = parser.parse_args()
    download_emec_buoy(
        region=args.region,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
