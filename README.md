# SIC downscaling and navigation code (curated snapshot)

This is a **source-code snapshot** curated from `local-path` on 2026-09-29. The code was authored by the project author with AI tools used as coding assistants. It is not a data release. The original research scripts are kept close to their working versions; machine-specific path roots have been replaced with `local-path` placeholders. No raw NetCDF data, memmap caches, checkpoints, results, logs, or fonts are included.

## What is included

| Directory | Purpose |
| --- | --- |
| `environment_guided_residual_unet_train/full_10var/` | Main environment-guided residual U-Net, ten-channel single-day configuration, training, validation and evaluation scripts. |
| `code/` | PR+TOS+SIC+mask baseline training, CMCC preprocessing, cache construction and prediction example. |
| `physical_group_revised_ablation_train/group_HT_tos_t2m/` | Revised thermodynamic-group ablation example. |
| `cmcc_astar_daily_passability/` | Daily A* route passability, window visualization, and a study-region map example. |

The thermodynamic ablation directory is a **single representative runnable example**, not every experiment performed on the server. See their `config.py` files for channel order, splits, hyperparameters and data paths. The main ten-channel model uses 2014–2023 data with training years 2014, 2016, 2017, 2019–2021; validation 2018 and 2023; test 2015 and 2022. Its configured maximum is 40 epochs.

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

## Updated independent-period navigation evaluation

The following scripts are located in `environment_guided_residual_unet_train/full_10var/` and use the adjacent `plot_full10_nep_path_buffer.py`, `config.py`, and `model.py`:

- `evaluate_cmcc4_vs_full10_navigation.py`: paired four-channel/ten-channel connectivity evaluation with a bilinear baseline.
- `evaluate_cmcc4_vs_full10_navigation_aggregate_input_trial.py`: aggregate-input baseline without bilinear interpolation and an audited in-memory repair of 24 dateline mask pixels.
- `consolidate_aggregate_navigation_20261016.py`: consolidate existing five-threshold evaluation outputs; requires both aggregate-input and previous seam-trial result directories. These outputs are not included.

Configure the `local-path` placeholders before running. The connectivity evaluators use four-neighbour components for reachability and spot-check against A* with no diagonal corner cutting. The seam repair is specific to the audited mask and is guarded by coordinate and pixel checks.

## Shared training modules and early stopping

The thermodynamic ablation example retains its own configuration and training/check entry points, and shares unchanged modules from `environment_guided_residual_unet_train/full_10var/`. Run it in a separate Python process from its own directory. Keep the repository structure intact.

All retained training configurations (four-channel CMCC configuration, full ten-channel model, and thermodynamic ablation example) now enable early stopping on validation loss with `EARLY_STOPPING_PATIENCE = 5` and `EARLY_STOPPING_MIN_DELTA = 0.0`. Training stops after five consecutive epochs without improvement, up to the configured maximum of 40 epochs, and saves the best checkpoint. Set patience to zero to disable early stopping. This release changes the stopping policy for configurations that previously trained for a fixed maximum; it does not modify previously trained checkpoints or reported paper results.

## Random seed

The training runs reported in the paper used **seed 42**. Set the ordinary Python variable `seed` in the selected experiment's `config.py` before running:

```python
seed = 42
```

The distributed configuration uses `seed = None` as a placeholder and assigns `Config.SEED = seed`. Replace `None` with your chosen integer. No environment variable is required. Training and random sample splitting use the same seed; fixed year splits remain unchanged. A fixed seed alone does not guarantee identical results across different hardware and software environments.
