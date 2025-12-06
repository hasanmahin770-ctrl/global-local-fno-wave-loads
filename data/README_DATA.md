# Data Download & Licensing

## Copernicus Marine Service (Global Reanalysis)

**Product**: GLOBAL_REANALYSIS_WAV_001_032
- **Temporal coverage**: 1993--present (daily)
- **Spatial resolution**: 0.5 degrees
- **Variables**: E(f, theta), Hs, Tp, Dp, etc.
- **License**: Creative Commons Attribution 4.0 (CC-BY-4.0)
- **Citation**: Copernicus Marine Service (2024)

### Access

1. Register free account: https://data.marine.copernicus.eu/
2. Create username/password
3. Download API credentials from account settings
4. Save to ~/.copernicusmarine_config:
   ```
   username=YOUR_EMAIL
   password=YOUR_PASSWORD
   ```

### Download Script

```bash
python data/download_copernicus_spectra.py \
  --dataset GLOBAL_REANALYSIS_WAV_001_032 \
  --lon_min -3.5 --lon_max -2.8 \
  --lat_min 59.0 --lat_max 59.5 \
  --start_date 2022-06-01 \
  --end_date 2022-09-01 \
  --output_dir data/raw/copernicus/
```

---

## EMEC Billia Croo Buoy

**Location**: Orkney, Scotland (59.35°N, -2.96°W)

### Public Data (FREE)
- Directional wave spectra (1-hourly)
- Buoy motion data
- Environmental parameters

**License**: Depends on specific data source; check EMEC website

### Download Script

```bash
python data/download_emec_buoy.py \
  --region billia_croo \
  --start_date 2022-06-01 \
  --end_date 2022-09-01 \
  --output_dir data/raw/emec/
```

### Restricted Data (Device Loads)
- Contact EMEC directly: support@emec.org.uk
- Request Billia Croo device load time-series
- Provide: research affiliation, intended use, timeline

**Alternative**: If unavailable, use synthetic RAO-generated labels (documented separately)

**Citation**: EMEC (2024). Billia Croo Buoy Data. European Marine Energy Centre.

---

## NOAA WaveWatch III (NWW3)

**Product**: Global wave hindcast via ERDDAP
- **Temporal**: 1980s--present
- **Resolution**: 0.5 degrees
- **License**: Public domain (NOAA)

### Download Script

```bash
python data/download_noaa_ww3.py \
  --lon_min -3.5 --lon_max -2.8 \
  --lat_min 59.0 --lat_max 59.5 \
  --start_date 2022-06-01 \
  --end_date 2022-09-01 \
  --output_dir data/raw/noaa/
```

**Citation**: NOAA (2024). WaveWatch III Hindcast. National Centers for Environmental Prediction.

---

## Synthetic RAO Labels (for reproducibility)

If EMEC device loads unavailable, we provide a synthetic pretraining dataset:
- 50,000 spectrum → load pairs
- Generated via linear frequency-domain RAO convolution
- Multiple canonical device geometries
- Clearly marked as synthetic in all results

**Usage**: `python synthetic/generate_pretraining_set.py --n_samples 50000`

---

## Summary of Licenses & Attribution

| Data Source | License | Attribution |
|-------------|---------|-------------|
| Copernicus CMEMS | CC-BY-4.0 | Cite Copernicus Marine (2024) |
| EMEC Billia Croo | Data provider-specific | Contact EMEC; see EMEC website |
| NOAA WW3 | Public domain | Cite NOAA (2024) |
| Synthetic RAO | Synthetic (this repo) | MIT License |

---

## Download Commands (All at Once)

```bash
# Set up credentials first
mkdir -p ~/.copernicusmarine
echo "username=YOUR_EMAIL" > ~/.copernicusmarine_config
echo "password=YOUR_PASSWORD" >> ~/.copernicusmarine_config

# Download all data
python data/download_copernicus_spectra.py --lon_min -3.5 --lon_max -2.8 --lat_min 59 --lat_max 59.5
python data/download_emec_buoy.py --region billia_croo
python data/download_noaa_ww3.py --lon_min -3.5 --lon_max -2.8 --lat_min 59 --lat_max 59.5

# Or use fallback synthetic data
python synthetic/generate_pretraining_set.py --n_samples 50000
```

**Expected size**: ~500 MB--2 GB depending on time window.

---

## Troubleshooting

### Copernicus Credentials Error
- Verify account is activated: https://data.marine.copernicus.eu/
- Check username/password are correct
- Try regenerating API token in account settings
- Ensure ~/.copernicusmarine_config file exists and is readable

### EMEC Data Missing
- Public buoy spectra should always download
- Device loads require EMEC permission (see EMEC website)
- Use synthetic RAO data as fallback (same format)

### NOAA Connection Issues
- Check internet connection
- Try again (ERDDAP server may have short downtime)
- Use cached data if available: `data/raw/noaa_cache/`

