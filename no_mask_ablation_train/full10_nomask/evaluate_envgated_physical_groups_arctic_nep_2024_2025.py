#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common Arctic/NEP evaluation for environment-gated full and physical groups.

Place this file beside the full_10var config.py/model.py and also keep
evaluate_envgated_full10_arctic_nep_2024_2025.py in the same directory.
No training or fine-tuning is performed.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import Config
from model import UNetWithResidual
from evaluate_envgated_full10_arctic_nep_2024_2025 import (
    CHANNEL_ALIASES,
    DEFAULT_CACHE,
    DEFAULT_CHECKPOINT,
    DEFAULT_MASK_FILE,
    CacheDataset,
    Stats,
    channel_index,
    extract_state_dict,
    gaussian_kernel,
    improvement,
    load_masks,
    load_meta,
    metric_direction,
    postprocess,
    read_records,
    safe_torch_load,
    spatial_ssim,
    valid_window_centers,
)


MODEL_VARS: Dict[str, List[str]] = {
    "envgated_full10": ["t2m", "u10", "v10", "msl", "pr", "ssr", "tcc", "tos", "sic"],
    "envgated_O_tos": ["tos", "sic"],
    "envgated_D_wind_msl": ["u10", "v10", "msl", "sic"],
    "envgated_T_thermal": ["t2m", "ssr", "tcc", "sic"],
    "envgated_P_pr": ["pr", "sic"],
}
METHODS = ("bilinear_sic", *MODEL_VARS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="环境引导全变量及O/D/T/P物理组：2024--2025全北极和NEP统一评价"
    )
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--mask-file", default=DEFAULT_MASK_FILE)
    parser.add_argument("--arctic-mask-var", default="mask_arctic_ocean")
    parser.add_argument("--nep-mask-var", default="mask_all_shipping_seas")
    parser.add_argument("--full-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--o-checkpoint", required=True, help="tos+sic+mask环境引导模型")
    parser.add_argument("--d-checkpoint", required=True, help="u10+v10+msl+sic+mask环境引导模型")
    parser.add_argument("--t-checkpoint", required=True, help="t2m+ssr+tcc+sic+mask环境引导模型")
    parser.add_argument("--p-checkpoint", required=True, help="pr+sic+mask环境引导模型")
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
        default="/home/ubuntu/sic_dsc/envgated_physical_groups_arctic_nep_2024_2025.csv",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_samples < 0:
        parser.error("batch-size必须>0，num-workers/max-samples必须>=0")
    if not 0.0 < args.zero_below < 1.0 or not 0.0 < args.ice_threshold < 1.0:
        parser.error("zero-below和ice-threshold必须位于(0,1)")
    return args


def build_model(
    checkpoint: Any, expected_vars: Sequence[str], device: torch.device, name: str
) -> Tuple[UNetWithResidual, Dict[str, Any]]:
    cfg = dict(checkpoint.get("config", {})) if isinstance(checkpoint, dict) else {}
    actual_vars = [str(value).lower() for value in cfg.get("INPUT_VARS", [])]
    expected_vars = [str(value).lower() for value in expected_vars]
    if actual_vars != expected_vars:
        raise RuntimeError(f"{name} INPUT_VARS={actual_vars}，预期={expected_vars}")
    if str(cfg.get("ARCHITECTURE_NAME", "")).lower() != "environment_guided_residual_unet":
        raise RuntimeError(f"{name}不是environment_guided_residual_unet")
    if str(cfg.get("MODEL_OUTPUT_MODE", "")).lower() != "sic_residual":
        raise RuntimeError(f"{name}不是sic_residual输出")
    if [int(value) for value in cfg.get("TIME_OFFSETS", [])] != [0]:
        raise RuntimeError(f"{name}不是T1模型: {cfg.get('TIME_OFFSETS')}")
    in_channels = len(expected_vars) + 1
    if int(cfg.get("INPUT_CHANNELS", -1)) != in_channels:
        raise RuntimeError(
            f"{name} INPUT_CHANNELS={cfg.get('INPUT_CHANNELS')}，预期={in_channels}"
        )
    expected_sic = expected_vars.index("sic")
    if int(cfg.get("SIC_CHANNEL_INDEX", -1)) != expected_sic:
        raise RuntimeError(f"{name} SIC_CHANNEL_INDEX不匹配")
    if int(cfg.get("MASK_CHANNEL_INDEX", -1)) != in_channels - 1:
        raise RuntimeError(f"{name} MASK_CHANNEL_INDEX不匹配")

    keys = (
        "MODEL_OUTPUT_MODE", "SIC_CHANNEL_INDEX", "MASK_CHANNEL_INDEX",
        "ENV_CHANNEL_INDICES", "SIC_RESIDUAL_SCALE", "SIC_BASELINE_INTERPOLATION",
        "ENV_GATE_INITIAL_BIAS", "ENV_GATE_SCALE", "SIC_RESIDUAL_ZERO_INIT",
    )
    existed = {key: hasattr(Config, key) for key in keys}
    previous = {key: getattr(Config, key, None) for key in keys}
    try:
        for key in keys:
            if key in cfg:
                setattr(Config, key, cfg[key])
        model = UNetWithResidual(in_channels=in_channels, out_channels=1)
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
    checkpoint_paths = {
        "envgated_full10": Path(args.full_checkpoint).expanduser().resolve(),
        "envgated_O_tos": Path(args.o_checkpoint).expanduser().resolve(),
        "envgated_D_wind_msl": Path(args.d_checkpoint).expanduser().resolve(),
        "envgated_T_thermal": Path(args.t_checkpoint).expanduser().resolve(),
        "envgated_P_pr": Path(args.p_checkpoint).expanduser().resolve(),
    }
    for name, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    cache = Path(args.cache).expanduser().resolve()
    meta = load_meta(cache)
    cache_names = list(meta.get("channel_names", []))
    if not cache_names:
        raise RuntimeError("meta.json缺少channel_names")
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
    season_counts = Counter(
        "melt_may_sep" if 5 <= int(row["month"]) <= 9 else "freeze_oct_apr"
        for row in records if int(row["month"])
    )

    mask_path = Path(args.mask_file).expanduser().resolve()
    arctic_mask, nep_mask = load_masks(
        mask_path, args.arctic_mask_var, args.nep_mask_var, tuple(y_shape[-2:])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    models: Dict[str, UNetWithResidual] = {}
    checkpoint_info: Dict[str, Dict[str, Any]] = {}
    channel_maps: Dict[str, List[int]] = {}
    for name, variables in MODEL_VARS.items():
        checkpoint = safe_torch_load(checkpoint_paths[name])
        model, cfg = build_model(checkpoint, variables, device, name)
        models[name] = model
        checkpoint_info[name] = {
            "path": str(checkpoint_paths[name]),
            "epoch": checkpoint.get("epoch", "") if isinstance(checkpoint, dict) else "",
            "val_loss": checkpoint.get("val_loss", "") if isinstance(checkpoint, dict) else "",
        }
        channel_maps[name] = [channel_index(cache_names, variable) for variable in variables]
        channel_maps[name].append(channel_index(cache_names, "mask"))
        print(
            f"{name}: vars={variables}+mask, channels={channel_maps[name]}, "
            f"epoch={checkpoint_info[name]['epoch']}, val_loss={checkpoint_info[name]['val_loss']}",
            flush=True,
        )
        del checkpoint

    # Dataset exposes all cache channels. Each model selects its own ordered subset.
    all_channels = list(range(x_shape[1]))
    dataset = CacheDataset(cache, meta, records, all_channels)
    loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size, "shuffle": False,
        "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    kernel = gaussian_kernel(device)
    static_masks = {"arctic_ocean": arctic_mask, "nep_all": nep_mask}
    complete_windows = {
        scope: valid_window_centers(mask, device) for scope, mask in static_masks.items()
    }
    stats: Dict[Tuple[str, str, str, str, str], Stats] = defaultdict(
        lambda: Stats(args.ice_threshold)
    )
    zero_name = f"zero_below_{args.zero_below:g}".replace(".", "p")
    bands = [
        ("target_0p00_0p05", 0.00, 0.05),
        ("target_0p05_0p15", 0.05, 0.15),
        ("target_0p15_0p30", 0.15, 0.30),
        ("target_0p30_0p80", 0.30, 0.80),
        ("target_0p80_1p00", 0.80, 1.000001),
    ]
    sic_cache_index = channel_index(cache_names, "sic")
    print("=== Common frozen evaluation ===", flush=True)
    print(f"samples={len(records)}, years={dict(year_counts)}, device={device}, AMP={use_amp}", flush=True)
    print("training/fine-tune=disabled", flush=True)

    with torch.inference_mode():
        for batch_index, (full_x, target, years, months) in enumerate(loader):
            full_x = full_x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).float()
            if not use_amp:
                full_x = full_x.float()
            if not torch.isfinite(full_x).all() or not torch.isfinite(target).all():
                raise RuntimeError(f"第{batch_index}批输入或真值包含非有限值")
            baseline_raw = F.interpolate(
                full_x[:, sic_cache_index:sic_cache_index + 1].float(),
                size=target.shape[-2:], mode="bilinear", align_corners=False,
            )
            outputs: Dict[str, torch.Tensor] = {"bilinear_sic": baseline_raw}
            for name, model in models.items():
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs[name] = model(full_x[:, channel_maps[name]])
                if not torch.isfinite(outputs[name]).all():
                    raise RuntimeError(f"第{batch_index}批{name}输出包含非有限值")

            truth_np = target.cpu().numpy()[:, 0]
            baseline_np = baseline_raw.clamp(0, 1).cpu().numpy()[:, 0]
            for process, zero_threshold in (("raw", 0.0), (zero_name, args.zero_below)):
                for method, raw_output in outputs.items():
                    prediction = postprocess(raw_output, zero_threshold)
                    prediction_np = prediction.cpu().numpy()[:, 0]
                    ssim_np = spatial_ssim(prediction, target, kernel).cpu().numpy()[:, 0]
                    for item in range(len(truth_np)):
                        truth = truth_np[item]
                        ice_zone = (truth >= args.ice_threshold) | (
                            baseline_np[item] >= args.ice_threshold
                        )
                        month = int(months[item])
                        groups = [("overall", "all"), ("year", str(int(years[item])))]
                        if month:
                            groups.append((
                                "season",
                                "melt_may_sep" if 5 <= month <= 9 else "freeze_oct_apr",
                            ))
                        scopes = {
                            "arctic_ocean": arctic_mask,
                            "arctic_ice_zone": arctic_mask & ice_zone,
                            "nep_all": nep_mask,
                            "nep_ice_zone": nep_mask & ice_zone,
                        }
                        for group_type, group_name in groups:
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
                                key = (
                                    process, "target_sic_band",
                                    f"{base_scope}:{band_name}", base_scope, method,
                                )
                                stats[key].update_pixels(prediction_np[item], truth, region)
                                centers = complete_windows[base_scope] & region[5:-5, 5:-5]
                                stats[key].update_ssim(ssim_np[item], centers)
            if batch_index == 0 or (batch_index + 1) % args.progress_every == 0:
                print(f"{batch_index + 1}/{len(loader)} batches", flush=True)

    rows: List[Dict[str, Any]] = []
    base_keys = sorted({key[:4] for key in stats})
    for process, group_type, group_name, scope in base_keys:
        metrics = {
            method: stats[(process, group_type, group_name, scope, method)].metrics()
            for method in METHODS
        }
        if group_type == "overall":
            sample_count = len(records)
        elif group_type == "year":
            sample_count = year_counts[int(group_name)]
        elif group_type == "season":
            sample_count = season_counts[group_name]
        else:
            sample_count = len(records)
        for metric in metrics["bilinear_sic"]:
            baseline_value = metrics["bilinear_sic"][metric]
            full_value = metrics["envgated_full10"][metric]
            for method in METHODS:
                value = metrics[method][metric]
                gain_baseline = improvement(value, baseline_value, metric)
                gain_full = improvement(value, full_value, metric)
                stat = stats[(process, group_type, group_name, scope, method)]
                info = checkpoint_info.get(method, {"path": "", "epoch": "", "val_loss": ""})
                rows.append({
                    "postprocess": process,
                    "stratum_type": group_type,
                    "stratum": group_name,
                    "scope": scope,
                    "metric": metric,
                    "direction": metric_direction(metric),
                    "method": method,
                    "value": value,
                    "gain_vs_bilinear": gain_baseline,
                    "gain_vs_bilinear_percent": (
                        100.0 * gain_baseline / abs(baseline_value)
                        if np.isfinite(baseline_value) and baseline_value != 0 else float("nan")
                    ),
                    "gain_vs_full10": gain_full,
                    "gain_vs_full10_percent": (
                        100.0 * gain_full / abs(full_value)
                        if np.isfinite(full_value) and full_value != 0 else float("nan")
                    ),
                    "sample_count": sample_count,
                    "pixel_count": stat.n,
                    "ssim_window_count": stat.ssim_n,
                    "checkpoint": info["path"],
                    "checkpoint_epoch": info["epoch"],
                    "checkpoint_val_loss": info["val_loss"],
                    "cache": str(cache),
                    "date_start": records[0]["date_key"],
                    "date_end": records[-1]["date_key"],
                    "mask_file": str(mask_path),
                    "arctic_mask_var": args.arctic_mask_var,
                    "nep_mask_var": args.nep_mask_var,
                    "ice_threshold": args.ice_threshold,
                    "ssim_definition": (
                        "11x11 Gaussian sigma1.5; data_range=1; valid windows wholly inside "
                        "static Arctic/NEP mask; dynamic zone/band selected by center pixel"
                    ),
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
