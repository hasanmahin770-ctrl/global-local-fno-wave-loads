#!/bin/bash
# One-command reproduction of Figure 1: Data Coverage & Pipeline

set -e

echo "========================================"
echo "Reproducing Figure 1 (Data Coverage)"
echo "========================================"

# Check Copernicus credentials
if [ ! -f ~/.copernicusmarine_config ]; then
    echo "ERROR: Copernicus credentials not found at ~/.copernicusmarine_config"
    echo "Please register at https://data.marine.copernicus.eu/ and save credentials"
    exit 1
fi

# Download data (small sample)
echo "\nDownloading Copernicus sample data..."
python data/download_copernicus_spectra.py \
    --dataset GLOBAL_REANALYSIS_WAV_001_032 \
    --lon_min -3.5 --lon_max -2.8 \
    --lat_min 59.0 --lat_max 59.5 \
    --start_date 2022-06-01 \
    --end_date 2022-06-15 \
    --output_dir data/raw/copernicus/

echo "\nDownloading EMEC buoy metadata..."
python data/download_emec_buoy.py \
    --region billia_croo \
    --start_date 2022-06-01 \
    --end_date 2022-06-15

echo "\nPreprocessing spectral grids..."
python preprocess/make_spectral_grids.py \
    --emec_buoy_dir data/raw/emec/ \
    --copernicus_dir data/raw/copernicus/ \
    --output_dir data/processed/ \
    --grid_size 64 64

echo "\nGenerating Figure 1..."
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_figures.ipynb \
    --output /tmp/reproduce_figures_run.ipynb

echo "\n========================================"
echo "✅ Figure 1 saved to: notebooks/outputs/figure_01_data_coverage.pdf"
echo "========================================"
