#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用四通道 PR+TOS+SIC+MASK 模型读取 CMCC memmap 缓存进行完整图预测。

默认先用 --dry-run 检查缓存结构。正式预测时只写 sic_pred 一个变量，
避免重复保存百分比变量造成两倍输出体积。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from netCDF4 import Dataset, date2num
from tqdm import tqdm

from config import Config
from model import UNetWithResidual


DEFAULT_CHECKPOINT = (
    "local-path/code/checkpoints/"
    "20260903_202642_multi_year_2014_2023_T1_d_rmse_fixed/best_model.pth"
)
DEFAULT_CACHE_DIR = (
    "local-path/data_cmcc/cache_prediction_input/"
    "cmcc_pr_tos_sic_mask_2050_float16"
)
# 只读取 y/x 网格，不读取其真值或时间轴；Bremen_2023 是静态网格模板。
DEFAULT_GRID_TEMPLATE = "local-path/data/dataset/target/Bremen_2023.nc"
DEFAULT_OUT_DIR = "local-path/data_cmcc/predict_pr_tos_sic_mask_2050"
EXPECTED_CHANNELS = ("pr", "tos", "sic", "mask")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="四通道 PR+TOS+SIC+MASK 模型对 CMCC memmap 缓存进行预测"
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--grid-template", default=DEFAULT_GRID_TEMPLATE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只预测按日期排序后的前 N 天；用于小规模试跑")
    parser.add_argument("--date", default=None,
                        help="只预测一个指定日期，例如 2050-08-01；不能与 --max-samples 同用")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--save-percent", action="store_true",
                        help="同时保存 sic_pred_percent（会明显增大输出文件）")
    parser.add_argument("--zero-below", type=float, default=0.0,
                        help="预测后将小于此值的像元置零；默认保留原始输出")
    parser.add_argument("--no-mask", action="store_true",
                        help="不将预测乘以缓存中的 mask 通道")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检查缓存和 checkpoint 配置，不进行预测或写文件")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} 不是 JSON 对象")
    return value


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def resolve_path(cache_dir: Path, value: Any, fallback: str) -> Path:
    path = Path(str(value if value is not None else fallback)).expanduser()
    return path if path.is_absolute() else cache_dir / path


def normalize_channel_name(name: str) -> str:
    base = str(name).lower().replace("_d", "")
    aliases = {"tp": "pr", "sst": "tos"}
    return aliases.get(base, base)


def load_cache(cache_dir: Path) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]], list[str]]:
    meta_path = cache_dir / "meta.json"
    index_path = cache_dir / "sample_index.csv"
    if not meta_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("缓存必须包含 meta.json 和 sample_index.csv")
    meta = load_json(meta_path)
    x_path = resolve_path(
        cache_dir,
        first_present(meta, ("X_path", "x_path", "x_file", "X_file")),
        "X.dat",
    )
    shape = first_present(meta, ("shape_X", "X_shape", "input_shape", "x_shape"))
    if shape is None:
        raise RuntimeError("meta.json 缺少 X 数组形状（shape_X/X_shape/input_shape）")
    if not x_path.is_file():
        raise FileNotFoundError(f"找不到缓存数组: {x_path}")
    dtype = np.dtype(str(meta.get("dtype", "float16")))
    expected_shape = tuple(int(v) for v in shape)
    # build_cmcc_prediction_input_cache.py 使用 np.lib.format.open_memmap() 写 X.npy。
    # 该格式带 NPY 头；若用 np.memmap 直接读取，会把文件头误认为样本值。
    if x_path.suffix.lower() == ".npy":
        x = np.load(x_path, mmap_mode="r")
    else:
        x = np.memmap(x_path, mode="r", dtype=dtype, shape=expected_shape)
    if tuple(x.shape) != expected_shape or np.dtype(x.dtype) != dtype:
        raise RuntimeError(
            f"缓存数组与 meta 不一致：array shape/dtype={tuple(x.shape)}/{x.dtype}, "
            f"meta={expected_shape}/{dtype}"
        )

    rows: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            sample_id = int(row.get("sample_id", row.get("cache_index", ordinal)))
            date = str(row.get("date", ""))
            if not date:
                raise RuntimeError(f"sample_index.csv 第 {ordinal + 2} 行没有 date")
            rows.append({"sample_id": sample_id, "date": date, "row": row})
    rows.sort(key=lambda r: (r["date"], r["sample_id"]))
    if len(rows) != x.shape[0]:
        raise RuntimeError(
            f"缓存数组有 {x.shape[0]} 样本，但 sample_index.csv 有 {len(rows)} 行"
        )

    raw_channels = first_present(
        meta, ("channel_names", "cache_channel_names", "feature_names", "input_vars")
    )
    channels = [str(v) for v in raw_channels] if isinstance(raw_channels, list) else []
    return meta, x, rows, channels


def check_cache_channels(x: np.ndarray, channels: list[str]) -> None:
    if x.ndim != 4 or x.shape[1] != 4:
        raise RuntimeError(f"要求四通道缓存 (N,4,H,W)，实际为 {tuple(x.shape)}")
    if channels:
        normalized = tuple(normalize_channel_name(v) for v in channels)
        if normalized != EXPECTED_CHANNELS:
            raise RuntimeError(
                f"缓存通道顺序必须为 {EXPECTED_CHANNELS}，实际为 {tuple(channels)}"
            )


def cache_channel_ranges(x: np.ndarray) -> list[tuple[float, float]]:
    """扫描四个缓存通道的全局范围，确保输入已按训练配置归一化。"""
    ranges: list[tuple[float, float]] = []
    for index in range(4):
        channel = x[:, index]
        ranges.append((float(np.nanmin(channel)), float(np.nanmax(channel))))
    for name, (vmin, vmax) in zip(EXPECTED_CHANNELS, ranges):
        if vmin < -1e-3 or vmax > 1.001:
            raise RuntimeError(
                f"缓存通道 {name} 的范围为 [{vmin:.6f}, {vmax:.6f}]，"
                "不符合训练输入归一化后的 [0,1]；请先检查缓存构建。"
            )
    return ranges


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[UNetWithResidual, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    saved_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if saved_config:
        for key, expected in (("INPUT_CHANNELS", 4), ("MODEL_OUTPUT_MODE", "sic_residual"), ("SIC_CHANNEL_INDEX", 2)):
            actual = saved_config.get(key)
            if actual != expected:
                raise RuntimeError(f"checkpoint 配置不匹配：{key}={actual!r}，期望 {expected!r}")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise RuntimeError("checkpoint 中未找到 model state_dict")

    model = UNetWithResidual(in_channels=4, out_channels=1).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, saved_config


def read_grid(grid_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with xr.open_dataset(grid_path) as ds:
        if "x" not in ds.coords or "y" not in ds.coords:
            raise RuntimeError(f"网格模板缺少 x/y 坐标: {grid_path}")
        grid_attrs: dict[str, Any] = {}
        if "polar_stereographic" in ds.variables:
            grid_attrs = dict(ds["polar_stereographic"].attrs)
        return ds["y"].values.astype(np.float64), ds["x"].values.astype(np.float64), grid_attrs


def create_output(
    out_path: Path,
    rows: list[dict[str, Any]],
    y_hr: np.ndarray,
    x_hr: np.ndarray,
    grid_attrs: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Dataset, Any, Any | None]:
    nc = Dataset(str(out_path), "w", format="NETCDF4")
    nc.createDimension("time", len(rows))
    nc.createDimension("y", len(y_hr))
    nc.createDimension("x", len(x_hr))
    time_var = nc.createVariable("time", "f8", ("time",))
    time_var.units = "days since 1970-01-01 00:00:00"
    time_var.calendar = "standard"
    time_var.standard_name = "time"
    time_var[:] = date2num(
        [np.datetime64(r["date"], "ms").astype(object) for r in rows],
        units=time_var.units,
        calendar=time_var.calendar,
    )
    y_var = nc.createVariable("y", "f8", ("y",))
    x_var = nc.createVariable("x", "f8", ("x",))
    y_var[:] = y_hr
    x_var[:] = x_hr
    y_var.units = "m"
    x_var.units = "m"
    y_var.standard_name = "projection_y_coordinate"
    x_var.standard_name = "projection_x_coordinate"
    grid = nc.createVariable("polar_stereographic", "i4")
    for key, value in grid_attrs.items():
        setattr(grid, key, value)
    if not grid_attrs:
        grid.grid_mapping_name = "polar_stereographic"
        grid.latitude_of_projection_origin = 90.0
        grid.standard_parallel = 70.0

    compression = {"zlib": True, "complevel": 4, "shuffle": True}
    chunks = (1, min(256, len(y_hr)), min(256, len(x_hr)))
    pred = nc.createVariable("sic_pred", "f4", ("time", "y", "x"), chunksizes=chunks, **compression)
    pred.units = "1"
    pred.long_name = "Predicted downscaled sea ice concentration"
    pred.grid_mapping = "polar_stereographic"
    percent = None
    if args.save_percent:
        percent = nc.createVariable("sic_pred_percent", "f4", ("time", "y", "x"), chunksizes=chunks, **compression)
        percent.units = "%"
        percent.long_name = "Predicted downscaled sea ice concentration in percent"
        percent.grid_mapping = "polar_stereographic"

    nc.title = "PR+TOS+SIC+MASK predicted downscaled CMCC SIC"
    nc.source = "year-fixed SIC-residual UNet inference from CMCC four-channel memmap cache"
    nc.model_path = str(Path(args.checkpoint).resolve())
    nc.cache_dir = str(Path(args.cache_dir).resolve())
    nc.channel_order = ",".join(EXPECTED_CHANNELS)
    nc.prediction_zero_below = float(args.zero_below)
    nc.mask_applied = np.int8(not args.no_mask)
    nc.Conventions = "CF-1.8"
    return nc, pred, percent


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    grid_path = Path(args.grid_template).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint_path}")
    if not grid_path.is_file():
        raise FileNotFoundError(f"网格模板不存在: {grid_path}")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples 必须为正整数")
    if args.date is not None and args.max_samples is not None:
        raise ValueError("--date 与 --max-samples 不能同时使用")

    meta, x, rows, channels = load_cache(cache_dir)
    check_cache_channels(x, channels)
    if args.date is not None:
        rows = [row for row in rows if row["date"] == args.date]
        if not rows:
            raise RuntimeError(f"缓存不含日期 {args.date}")
    elif args.max_samples is not None:
        rows = rows[: args.max_samples]
    print("=== CMCC four-channel cache ===")
    print(f"cache_dir    = {cache_dir}")
    print(f"X.shape      = {tuple(x.shape)}")
    print(f"dtype        = {x.dtype}")
    print(f"channels     = {channels or 'meta.json 未记录；已按4通道结构继续'}")
    print(f"date range   = {rows[0]['date']} to {rows[-1]['date']}")
    print(f"samples used = {len(rows)}")
    ranges = cache_channel_ranges(x)
    print("all-cache channel min/max:")
    for index, (name, (vmin, vmax)) in enumerate(zip(EXPECTED_CHANNELS, ranges)):
        print(f"  {index}: {name:4s} min={vmin:.6f}, max={vmax:.6f}")

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 但当前 CUDA 不可用")
    # dry-run 需要在显存已满时仍可完成结构核验，因此强制在 CPU 加载 checkpoint。
    device = torch.device("cpu") if args.dry_run else requested_device
    model, saved_config = load_model(checkpoint_path, device)
    print("=== Checkpoint ===")
    print(f"checkpoint   = {checkpoint_path}")
    print(f"epoch        = {saved_config.get('epoch', 'stored outside config') if saved_config else 'unknown'}")
    print(f"device       = {device}" + (f" (requested {requested_device}; dry-run uses CPU)" if args.dry_run else ""))
    print(f"AMP          = {not args.no_amp and device.type == 'cuda'}")
    if args.dry_run:
        print("[DRY RUN] 缓存和 checkpoint 已通过结构检查；未进行预测，也未写入文件。")
        return

    y_hr, x_hr, grid_attrs = read_grid(grid_path)
    expected_shape = (x.shape[2] * 4, x.shape[3] * 4)
    if (len(y_hr), len(x_hr)) != expected_shape:
        raise RuntimeError(
            f"网格模板尺寸 {(len(y_hr), len(x_hr))} 与缓存4倍上采样尺寸 {expected_shape} 不一致"
        )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"_{args.date.replace('-', '')}"
        if args.date is not None
        else f"_first{len(rows)}days" if args.max_samples is not None else ""
    )
    out_name = args.output_name or f"sic_pred_6p25km_cmcc_pr_tos_sic_mask_2050{suffix}.nc"
    out_path = out_dir / out_name
    if out_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖: {out_path}")

    nc, output_pred, output_percent = create_output(out_path, rows, y_hr, x_hr, grid_attrs, args)
    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(rows), args.batch_size), desc="CMCC prediction"):
                batch_rows = rows[start : start + args.batch_size]
                sample_ids = [int(r["sample_id"]) for r in batch_rows]
                xb = torch.from_numpy(np.asarray(x[sample_ids], dtype=np.float32)).to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(not args.no_amp and device.type == "cuda")):
                    prediction = model(xb).clamp(0.0, 1.0)
                    if not args.no_mask:
                        mask_hr = F.interpolate(xb[:, 3:4], size=prediction.shape[-2:], mode="bilinear", align_corners=False)
                        prediction = prediction * mask_hr.to(dtype=prediction.dtype)
                if args.zero_below > 0:
                    prediction = torch.where(prediction < args.zero_below, torch.zeros_like(prediction), prediction)
                values = prediction[:, 0].float().cpu().numpy()
                output_pred[start : start + len(batch_rows), :, :] = values
                if output_percent is not None:
                    output_percent[start : start + len(batch_rows), :, :] = values * 100.0
                nc.sync()
    finally:
        nc.close()
    print(f"[OK] 已保存: {out_path}")


if __name__ == "__main__":
    main()
