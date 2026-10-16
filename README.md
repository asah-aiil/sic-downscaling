# SIC downscaling and navigation code (curated snapshot)

This is a **source-code snapshot** curated from `local-path` on 2026-09-29. The code was authored by the project author with AI tools used as coding assistants. It is not a data release. The original research scripts are kept close to their working versions; machine-specific path roots have been replaced with `local-path` placeholders. No raw NetCDF data, memmap caches, checkpoints, results, logs, or fonts are included.

## What is included

| Directory | Purpose |
| --- | --- |
| `environment_guided_residual_unet_train/full_10var/` | Main environment-guided residual U-Net, ten-channel single-day configuration, training, validation and evaluation scripts. |
| `code/` | PR+TOS+SIC+mask baseline training, CMCC preprocessing, cache construction and prediction example. |
| `environment_guided_residual_unet_train/group_O_tos/` | Ocean-temperature physical-group ablation example. |
| `no_mask_ablation_train/full10_nomask/` | No-mask ablation example. |
| `tenvar_leave_one_out_train/drop_tos/` | Leave-one-variable-out ablation example. |
| `physical_group_revised_ablation_train/group_HT_tos_t2m/` | Revised thermodynamic-group ablation example. |
| `cmcc_astar_daily_passability/` | Daily A* route passability, window visualization, and a study-region map example. |

The ablation directories are **representative runnable variants**, not every experiment performed on the server. See their `config.py` files for channel order, splits, hyperparameters and data paths. The main ten-channel model uses 2014–2023 data with training years 2014, 2016, 2017, 2019–2021; validation 2018 and 2023; test 2015 and 2022. Its configured maximum is 40 epochs.

## Data and setup

The training inputs are yearly 25 km NetCDF files named `input_25km_{year}_25km.nc`; the targets are Bremen 6.25 km SIC NetCDF files named `Bremen_{year}.nc`. The model also uses a prepared static mask and a float16 temporal memmap cache. CMCC examples expect CMCC daily `pr`, `tos` and `siconc` files plus matching grid templates. **These inputs are not distributed here**; download them from the links below.

### Public data download links

| Data | Public download portal |
| --- | --- |
| CMIP6 / CMCC-CM2-VHR4 | [ESGF CMIP6](https://esgf-node.llnl.gov/projects/cmip6/) |
| ERA5 single-level reanalysis | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) |
| Bremen sea-ice concentration | [University of Bremen Data Archive](https://seaice.uni-bremen.de/data-archive/) |

This repository provides code and download links. The datasets are not included.

Use a Python environment with PyTorch, NumPy, xarray, netCDF4, SciPy, pyproj, matplotlib and tqdm. CUDA is optional but training was designed for a GPU. See `requirements.txt` for the import-level dependency list; it is not a lockfile. Before execution, edit `DATA_ROOT`, `TEMPORAL_CACHE_DIR` and `PROJECT_ROOT` in the relevant local `config.py`, and supply paths for the command-line scripts. A configuration from one variant must not be substituted for another: the feature/channel order is model-specific.

Typical server-side sequence after downloading and preparing the data:

```bash
# From the full_10var directory, after editing its config.py:
python build_temporal_memmap_cache.py --help
python main.py

# CMCC projection example; inspect --help and specify your own paths:
python code/preprocess_cmcc_yearly.py --help
python code/build_cmcc_prediction_input_cache.py --help
python code/predict_cmcc_pr_tos_sic_mask_from_cache.py --help

# A* daily route passability example:
python cmcc_astar_daily_passability/evaluate_cmcc_astar_daily.py --help
```

The `code/` directory contains the four-channel baseline required by the supplied CMCC prediction script. The environment-guided full-ten-variable model and this four-channel CMCC projection are **different experiments**. Do not label a four-channel prediction as a full-ten-variable prediction.

## Verification and release notes

The package was checked for Python syntax and for accidental inclusion of data/weight files. End-to-end training or route evaluation cannot be verified without the original datasets and checkpoints. Replace all `local-path` placeholders with your own directories before running the scripts.
