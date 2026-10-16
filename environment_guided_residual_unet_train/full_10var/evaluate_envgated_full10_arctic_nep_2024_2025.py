#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one environment-guided full-10-channel model over Arctic and NEP.

The script performs no training.  It runs each sample once and writes one tidy
CSV containing the model and its bilinear coarse-SIC baseline on identical
pixels.  Metrics are reported for raw output and output zeroed below 0.05.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset

from config import Config
from model import UNetWithResidual


DEFAULT_CHECKPOINT = (
    "/home/ubuntu/sic_dsc/code/checkpoints/"
    "20260910_015743_envgated_t2m_u10_v10_msl_pr_ssr_tcc_tos_sic_mask_"
    "year_fixed_sic_residual_2014_2023_T1_d_rmse_fixed/best_model.pth"
)
DEFAULT_CACHE = (
    "/home/ubuntu/sic_dsc/data/"
    "cache_temporal_T3_float16_2024_2025_rebuilt"
)
DEFAULT_MASK_FILE = (
    "/home/ubuntu/sic_dsc/data/dataset/mask/"
    "eurasia_shipping_masks_6p25km_with_arctic.nc"
)
LOGICAL_VARS = ["t2m", "u10", "v10", "msl", "pr", "ssr", "tcc", "tos", "sic"]
CHANNEL_ALIASES = {
    "t2m": ["t2m_d"],
    "u10": ["u10_d"],
    "v10": ["v10_d"],
    "msl": ["msl_d"],
    "pr": ["pr_d", "tp_d"],
    "ssr": ["ssr_d"],
    "tcc": ["tcc_d"],
    "tos": ["tos_d", "sst_d"],
    "sic": ["sic_d", "siconc_d"],
    "mask": ["mask_d", "sic_mask_d"],
}
METHODS = ("bilinear_sic", "envgated_full10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="环境引导全10通道模型：2024--2025全北极与东北航道统一评价"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--mask-file", default=DEFAULT_MASK_FILE)
    parser.add_argument("--arctic-mask-var", default="mask_arctic_ocean")
    parser.add_argument("--nep-mask-var", default="mask_all_shipping_seas")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--zero-below", type=float, default=0.05)
    parser.add_argument("--ice-threshold", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="0=全部样本；正数仅用于冒烟测试")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--output-csv",
        default="/home/ubuntu/sic_dsc/envgated_full10_arctic_nep_2024_2025.csv",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0:
        parser.error("batch-size必须>0，num-workers/max-samples必须>=0")
    if not 0.0 < args.zero_below < 1.0:
        parser.error("zero-below必须位于(0,1)")
    if not 0.0 < args.ice_threshold < 1.0:
        parser.error("ice-threshold必须位于(0,1)")
    return args


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise RuntimeError("checkpoint中没有可识别的模型权重")


def load_meta(cache: Path) -> Dict[str, Any]:
    path = cache / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_records(cache: Path, n: int, years: Sequence[int]) -> List[Dict[str, Any]]:
    path = cache / "sample_index.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    wanted = set(int(year) for year in years)
    records: List[Dict[str, Any]] = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"sample_id", "year"}.issubset(fields):
            raise RuntimeError(f"{path}缺少sample_id/year列: {reader.fieldnames}")
        for row in reader:
            sample_id = int(row["sample_id"])
            year = int(row["year"])
            if not 0 <= sample_id < n:
                raise RuntimeError(f"sample_id越界: {sample_id}, N={n}")
            if sample_id in seen:
                raise RuntimeError(f"sample_id重复: {sample_id}")
            seen.add(sample_id)
            if year not in wanted:
                continue
            date_key = str(row.get("date", "")).strip()
            if date_key:
                parsed = datetime.fromisoformat(date_key[:10])
                month = parsed.month
            else:
                time_idx = int(row.get("time_idx", 0))
                date_key = f"{year}:{time_idx:03d}"
                month = 0
            records.append({
                "sample_id": sample_id, "year": year,
                "month": month, "date_key": date_key,
            })
    records.sort(key=lambda row: row["date_key"])
    return records


def channel_index(names: Sequence[str], logical_name: str) -> int:
    for candidate in CHANNEL_ALIASES[logical_name]:
        if candidate in names:
            return list(names).index(candidate)
    raise RuntimeError(
        f"缓存缺少{logical_name}通道，候选={CHANNEL_ALIASES[logical_name]}，实际={list(names)}"
    )


def load_masks(
    path: Path, arctic_var: str, nep_var: str, expected_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    errors: List[str] = []
    dataset = None
    for engine in ("h5netcdf", "netcdf4", "scipy", None):
        try:
            kwargs = {} if engine is None else {"engine": engine}
            dataset = xr.open_dataset(path, decode_times=False, **kwargs)
            break
        except Exception as exc:
            errors.append(f"{engine or 'auto'}: {exc}")
    if dataset is None:
        raise RuntimeError("无法读取掩膜: " + " | ".join(errors))
    try:
        missing = [name for name in (arctic_var, nep_var) if name not in dataset]
        if missing:
            raise KeyError(f"掩膜缺少{missing}；可用变量={list(dataset.data_vars)}")

        def read_one(name: str) -> np.ndarray:
            array = dataset[name].squeeze(drop=True)
            if "y" in array.dims and "x" in array.dims:
                array = array.transpose("y", "x")
            values = np.asarray(array.values, dtype=np.float32)
            if values.ndim != 2 or tuple(values.shape) != tuple(expected_shape):
                raise ValueError(
                    f"{name}形状{values.shape}与目标网格{expected_shape}不一致，禁止自动插值"
                )
            result = np.isfinite(values) & (values > 0.5)
            if not result.any():
                raise ValueError(f"{name}没有有效像元")
            return result

        return read_one(arctic_var), read_one(nep_var)
    finally:
        dataset.close()


class CacheDataset(Dataset):
    def __init__(
        self, cache: Path, meta: Dict[str, Any], records: Sequence[Dict[str, Any]],
        channels: Sequence[int],
    ):
        self.x_path = cache / meta.get("x_file", "X.npy")
        self.y_path = cache / meta.get("y_file", "Y.npy")
        self.records = list(records)
        self.channels = list(channels)
        self._x = None
        self._y = None

    def _open(self) -> None:
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r")
        if self._y is None:
            self._y = np.load(self.y_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int):
        self._open()
        row = self.records[item]
        sample_id = int(row["sample_id"])
        return (
            torch.from_numpy(np.array(self._x[sample_id, self.channels], copy=True)),
            torch.from_numpy(np.array(self._y[sample_id], copy=True)),
            int(row["year"]),
            int(row["month"]),
        )


class Stats:
    def __init__(self, ice_threshold: float):
        self.ice_threshold = float(ice_threshold)
        self.n = 0
        self.sum_sq = 0.0
        self.sum_abs = 0.0
        self.sum_error = 0.0
        self.tp = self.fp = self.fn = 0
        self.ssim_sum = 0.0
        self.ssim_n = 0

    def update_pixels(self, prediction: np.ndarray, target: np.ndarray, region: np.ndarray) -> None:
        valid = region & np.isfinite(prediction) & np.isfinite(target)
        error = (prediction[valid] - target[valid]).astype(np.float64)
        if error.size == 0:
            return
        self.n += int(error.size)
        self.sum_sq += float(np.dot(error, error))
        self.sum_abs += float(np.abs(error).sum())
        self.sum_error += float(error.sum())
        pred_ice = prediction[valid] >= self.ice_threshold
        true_ice = target[valid] >= self.ice_threshold
        self.tp += int(np.count_nonzero(pred_ice & true_ice))
        self.fp += int(np.count_nonzero(pred_ice & ~true_ice))
        self.fn += int(np.count_nonzero(~pred_ice & true_ice))

    def update_ssim(self, ssim_map: np.ndarray, centers: np.ndarray) -> None:
        valid = centers & np.isfinite(ssim_map)
        values = ssim_map[valid].astype(np.float64)
        self.ssim_sum += float(values.sum())
        self.ssim_n += int(values.size)

    def metrics(self) -> Dict[str, float]:
        if self.n == 0:
            basic = {name: float("nan") for name in ("rmse", "mae", "bias")}
        else:
            basic = {
                "rmse": (self.sum_sq / self.n) ** 0.5,
                "mae": self.sum_abs / self.n,
                "bias": self.sum_error / self.n,
            }
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        iou = self.tp / (self.tp + self.fp + self.fn) if self.tp + self.fp + self.fn else 0.0
        return {
            **basic,
            "ssim": self.ssim_sum / self.ssim_n if self.ssim_n else float("nan"),
            "ice_precision": precision,
            "ice_recall": recall,
            "ice_f1": f1,
            "ice_iou": iou,
        }


def gaussian_kernel(device: torch.device) -> torch.Tensor:
    coordinates = torch.arange(11, device=device, dtype=torch.float32) - 5
    vector = torch.exp(-(coordinates ** 2) / (2.0 * 1.5 ** 2))
    vector /= vector.sum()
    return (vector[:, None] * vector[None, :])[None, None]


def spatial_ssim(prediction: torch.Tensor, target: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """11x11 Gaussian SSIM map, data range 1, valid convolution."""
    prediction = prediction.float()
    target = target.float()
    mean_p = F.conv2d(prediction, kernel)
    mean_t = F.conv2d(target, kernel)
    var_p = (F.conv2d(prediction * prediction, kernel) - mean_p * mean_p).clamp_min(0)
    var_t = (F.conv2d(target * target, kernel) - mean_t * mean_t).clamp_min(0)
    covariance = F.conv2d(prediction * target, kernel) - mean_p * mean_t
    return (
        (2 * mean_p * mean_t + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)
        / ((mean_p ** 2 + mean_t ** 2 + 0.01 ** 2) * (var_p + var_t + 0.03 ** 2))
    )


def valid_window_centers(mask: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(mask, device=device, dtype=torch.float32)[None, None]
    centers = F.avg_pool2d(tensor, 11, stride=1) >= (1.0 - 1e-7)
    return centers.cpu().numpy()[0, 0]


def postprocess(value: torch.Tensor, zero_below: float) -> torch.Tensor:
    result = value.float().clamp(0.0, 1.0)
    if zero_below > 0:
        result = torch.where(result < zero_below, torch.zeros_like(result), result)
    return result


def metric_direction(metric: str) -> str:
    if metric in {"rmse", "mae"}:
        return "lower"
    if metric == "bias":
        return "abs_lower"
    return "higher"


def improvement(model: float, baseline: float, metric: str) -> float:
    if metric == "bias":
        return abs(baseline) - abs(model)
    if metric_direction(metric) == "lower":
        return baseline - model
    return model - baseline


def configure_and_build_model(checkpoint: Any, device: torch.device) -> Tuple[UNetWithResidual, Dict[str, Any]]:
    cfg = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    expected_vars = LOGICAL_VARS
    actual_vars = [str(value).lower() for value in cfg.get("INPUT_VARS", [])]
    checks = {
        "ARCHITECTURE_NAME": "environment_guided_residual_unet",
        "MODEL_OUTPUT_MODE": "sic_residual",
    }
    for key, expected in checks.items():
        actual = str(cfg.get(key, "")).lower()
        if actual != expected:
            raise RuntimeError(f"checkpoint {key}={actual!r}，预期{expected!r}")
    if actual_vars != expected_vars:
        raise RuntimeError(f"checkpoint INPUT_VARS={actual_vars}，预期={expected_vars}")
    if [int(v) for v in cfg.get("TIME_OFFSETS", [])] != [0]:
        raise RuntimeError(f"checkpoint不是T1模型: {cfg.get('TIME_OFFSETS')}")
    if int(cfg.get("INPUT_CHANNELS", -1)) != 10:
        raise RuntimeError(f"checkpoint INPUT_CHANNELS={cfg.get('INPUT_CHANNELS')}，预期10")

    keys = (
        "MODEL_OUTPUT_MODE", "SIC_CHANNEL_INDEX", "MASK_CHANNEL_INDEX",
        "ENV_CHANNEL_INDICES", "SIC_RESIDUAL_SCALE", "SIC_BASELINE_INTERPOLATION",
        "ENV_GATE_INITIAL_BIAS", "ENV_GATE_SCALE", "SIC_RESIDUAL_ZERO_INIT",
    )
    previous = {key: getattr(Config, key, None) for key in keys}
    existed = {key: hasattr(Config, key) for key in keys}
    try:
        for key in keys:
            if key in cfg:
                setattr(Config, key, cfg[key])
        model = UNetWithResidual(in_channels=10, out_channels=1)
    finally:
        for key in keys:
            if existed[key]:
                setattr(Config, key, previous[key])
            elif hasattr(Config, key):
                delattr(Config, key)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    return model.to(device).eval(), cfg


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    cache = Path(args.cache).expanduser().resolve()
    mask_path = Path(args.mask_file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    meta = load_meta(cache)
    names = list(meta.get("channel_names", []))
    if not names:
        raise RuntimeError("meta.json缺少channel_names")
    channels = [channel_index(names, name) for name in LOGICAL_VARS]
    channels.append(channel_index(names, "mask"))
    selected_names = [names[index] for index in channels]

    x_path = cache / meta.get("x_file", "X.npy")
    y_path = cache / meta.get("y_file", "Y.npy")
    x_shape = np.load(x_path, mmap_mode="r").shape
    y_shape = np.load(y_path, mmap_mode="r").shape
    n = int(meta.get("N", x_shape[0]))
    if x_shape[0] != n or y_shape[0] != n or len(y_shape) != 4 or y_shape[1] != 1:
        raise RuntimeError(f"缓存形状异常: N={n}, X={x_shape}, Y={y_shape}")

    records = read_records(cache, n, args.years)
    if args.max_samples:
        records = records[:args.max_samples]
    if not records:
        raise RuntimeError(f"缓存中没有请求年份: {args.years}")
    year_counts = Counter(int(row["year"]) for row in records)

    arctic_mask, nep_mask = load_masks(
        mask_path, args.arctic_mask_var, args.nep_mask_var, tuple(y_shape[-2:])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = safe_torch_load(checkpoint_path)
    model, checkpoint_cfg = configure_and_build_model(checkpoint, device)
    use_amp = device.type == "cuda" and not args.no_amp
    kernel = gaussian_kernel(device)
    static_masks = {"arctic_ocean": arctic_mask, "nep_all": nep_mask}
    complete_windows = {
        scope: valid_window_centers(mask, device) for scope, mask in static_masks.items()
    }

    dataset = CacheDataset(cache, meta, records, channels)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    stats: Dict[Tuple[str, str, str, str, str], Stats] = defaultdict(
        lambda: Stats(args.ice_threshold)
    )
    sample_counts: Counter = Counter()
    zero_name = f"zero_below_{args.zero_below:g}".replace(".", "p")
    bands = [
        ("target_0p00_0p05", 0.00, 0.05),
        ("target_0p05_0p15", 0.05, 0.15),
        ("target_0p15_0p30", 0.15, 0.30),
        ("target_0p30_0p80", 0.30, 0.80),
        ("target_0p80_1p00", 0.80, 1.000001),
    ]
    print("=== Environment-guided full10 frozen evaluation ===", flush=True)
    print(f"checkpoint = {checkpoint_path}", flush=True)
    print(f"best epoch = {checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}", flush=True)
    print(f"best val   = {checkpoint.get('val_loss', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}", flush=True)
    print(f"cache      = {cache}", flush=True)
    print(f"channels   = {selected_names} @ {channels}", flush=True)
    print(f"samples    = {len(records)}, years={dict(year_counts)}", flush=True)
    print(f"device     = {device}, AMP={use_amp}", flush=True)
    print(f"arctic pixels={int(arctic_mask.sum())}, NEP pixels={int(nep_mask.sum())}", flush=True)
    print("training/fine-tune = disabled", flush=True)

    with torch.inference_mode():
        for batch_index, (inputs, target, years, months) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()
            if not use_amp:
                inputs = inputs.float()
            if not torch.isfinite(inputs).all() or not torch.isfinite(target).all():
                raise RuntimeError(f"第{batch_index}批输入或真值包含非有限值")
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                model_raw = model(inputs)
            if not torch.isfinite(model_raw).all():
                raise RuntimeError(f"第{batch_index}批模型输出包含非有限值")
            baseline_raw = F.interpolate(
                inputs[:, 8:9].float(), size=target.shape[-2:],
                mode="bilinear", align_corners=False,
            )
            raw_outputs = {"bilinear_sic": baseline_raw, "envgated_full10": model_raw}

            target_np = target.cpu().numpy()[:, 0]
            baseline_np = baseline_raw.float().clamp(0, 1).cpu().numpy()[:, 0]
            for process, threshold in (("raw", 0.0), (zero_name, args.zero_below)):
                for method, output in raw_outputs.items():
                    prediction = postprocess(output, threshold)
                    prediction_np = prediction.cpu().numpy()[:, 0]
                    ssim_np = spatial_ssim(prediction, target, kernel).cpu().numpy()[:, 0]
                    for item in range(len(target_np)):
                        truth = target_np[item]
                        ice_zone = (truth >= args.ice_threshold) | (
                            baseline_np[item] >= args.ice_threshold
                        )
                        groupings = [("overall", "all"), ("year", str(int(years[item])))]
                        month = int(months[item])
                        if month:
                            groupings.append(("season", "melt_may_sep" if 5 <= month <= 9 else "freeze_oct_apr"))
                        scopes = {
                            "arctic_ocean": arctic_mask,
                            "arctic_ice_zone": arctic_mask & ice_zone,
                            "nep_all": nep_mask,
                            "nep_ice_zone": nep_mask & ice_zone,
                        }
                        for group_type, group_name in groupings:
                            sample_counts[(group_type, group_name)] += 1
                            for scope, region in scopes.items():
                                key = (process, group_type, group_name, scope, method)
                                stats[key].update_pixels(prediction_np[item], truth, region)
                                static_scope = "arctic_ocean" if scope.startswith("arctic") else "nep_all"
                                centers = complete_windows[static_scope]
                                if scope.endswith("ice_zone"):
                                    centers = centers & ice_zone[5:-5, 5:-5]
                                stats[key].update_ssim(ssim_np[item], centers)
                        for base_scope, base_mask in static_masks.items():
                            for band_name, lower, upper in bands:
                                region = base_mask & (truth >= lower) & (truth < upper)
                                key = (process, "target_sic_band", f"{base_scope}:{band_name}", base_scope, method)
                                stats[key].update_pixels(prediction_np[item], truth, region)
                                centers = complete_windows[base_scope] & region[5:-5, 5:-5]
                                stats[key].update_ssim(ssim_np[item], centers)

            if batch_index == 0 or (batch_index + 1) % args.progress_every == 0:
                print(f"{batch_index + 1}/{len(loader)} batches", flush=True)

    rows: List[Dict[str, Any]] = []
    keys = sorted({key[:4] for key in stats})
    for process, group_type, group_name, scope in keys:
        computed = {
            method: stats[(process, group_type, group_name, scope, method)].metrics()
            for method in METHODS
        }
        baseline_stats = stats[(process, group_type, group_name, scope, "bilinear_sic")]
        for metric in computed["bilinear_sic"]:
            baseline_value = computed["bilinear_sic"][metric]
            model_value = computed["envgated_full10"][metric]
            gain = improvement(model_value, baseline_value, metric)
            rows.append({
                "postprocess": process,
                "stratum_type": group_type,
                "stratum": group_name,
                "scope": scope,
                "metric": metric,
                "direction": metric_direction(metric),
                "bilinear_sic": baseline_value,
                "envgated_full10": model_value,
                "model_minus_bilinear": model_value - baseline_value,
                "model_gain": gain,
                "model_gain_percent": (
                    100.0 * gain / abs(baseline_value)
                    if np.isfinite(baseline_value) and baseline_value != 0 else float("nan")
                ),
                "sample_count": (
                    len(records) if group_type == "target_sic_band"
                    else sample_counts[(group_type, group_name)] // (len(METHODS) * 2)
                ),
                "pixel_count": baseline_stats.n,
                "ssim_window_count": baseline_stats.ssim_n,
                "ssim_definition": (
                    "11x11 Gaussian sigma1.5; data_range=1; valid windows wholly inside "
                    "static Arctic/NEP mask; dynamic zone/band selected by center pixel"
                ),
                "ice_threshold": args.ice_threshold,
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch", "") if isinstance(checkpoint, dict) else "",
                "checkpoint_val_loss": checkpoint.get("val_loss", "") if isinstance(checkpoint, dict) else "",
                "cache": str(cache),
                "date_start": records[0]["date_key"],
                "date_end": records[-1]["date_key"],
                "arctic_mask_var": args.arctic_mask_var,
                "nep_mask_var": args.nep_mask_var,
                "mask_file": str(mask_path),
                "smoke_test": bool(args.max_samples),
            })

    output = Path(args.output_csv).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== Evaluation complete ===", flush=True)
    print(f"rows={len(rows)}", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
