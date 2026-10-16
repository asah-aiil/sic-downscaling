#!/usr/bin/env python3
"""把单年 CMCC 25 km NetCDF 转成模型预测所需的 float16 输入缓存。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2050)
    p.add_argument("--input-dir", type=Path,
                   default=Path("/home/ubuntu/sic_dsc/data_cmcc/processed_25km"))
    p.add_argument("--cache-root", type=Path,
                   default=Path("/home/ubuntu/sic_dsc/data_cmcc/cache_prediction_input"))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def human_size(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return str(n)


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def normalize(arr, lo, hi):
    arr = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def date_strings(ds: xr.Dataset, year: int) -> list[str]:
    try:
        vals = ds["time"].dt.strftime("%Y-%m-%d").values
        dates = [str(v) for v in vals]
    except Exception:
        dates = [f"{year}-{m:02d}-{d:02d}" for m in range(1, 13)
                 for d in range(1, 32)
                 if not (m == 2 and d == 29)
                 and _valid_date(year, m, d)]
    if len(dates) != 365 or any(s.endswith("-02-29") for s in dates):
        raise RuntimeError(f"时间坐标不符合365天无闰日要求: count={len(dates)}")
    return dates


def _valid_date(year, month, day):
    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False


def main():
    args = args_parse()
    source = args.input_dir / f"input_25km_{args.year}_cmcc.nc"
    cache_dir = args.cache_root / f"cmcc_pr_tos_sic_mask_{args.year}_float16"
    final_x = cache_dir / "X.npy"
    tmp_x = cache_dir / "X.tmp.npy"

    if not source.exists():
        raise FileNotFoundError(source)
    if cache_dir.exists() and final_x.exists() and not args.overwrite:
        print(f"缓存已存在，未覆盖: {cache_dir}")
        print("需要重建时增加 --overwrite")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in (tmp_x, final_x, cache_dir / "meta.json", cache_dir / "sample_index.csv"):
        if p.exists():
            p.unlink()

    source_bytes = source.stat().st_size
    with xr.open_dataset(source, decode_times=True, mask_and_scale=True) as ds:
        required = ("pr", "tos", "sic")
        for name in required:
            if name not in ds:
                raise RuntimeError(f"输入缺少变量: {name}")
        if dict(ds.sizes) != {"time": 365, "y": 448, "x": 304}:
            raise RuntimeError(f"输入尺寸不正确: {dict(ds.sizes)}")
        dates = date_strings(ds, args.year)

        shape = (365, 4, 448, 304)
        X = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.float16, shape=shape)
        for t in range(365):
            pr = normalize(ds["pr"].isel(time=t).values, 0.0, 0.05)
            tos = normalize(ds["tos"].isel(time=t).values, 270.0, 310.0)
            sic_raw = np.nan_to_num(ds["sic"].isel(time=t).values, nan=0.0).astype(np.float32)
            sic01 = np.clip(sic_raw / 100.0 if np.nanmax(sic_raw) > 1.5 else sic_raw, 0.0, 1.0)
            mask = (sic01 >= 0.15).astype(np.float32)
            X[t, 0] = pr.astype(np.float16)
            X[t, 1] = tos.astype(np.float16)
            X[t, 2] = sic01.astype(np.float16)
            X[t, 3] = mask.astype(np.float16)
            if t == 0 or (t + 1) % 50 == 0 or t == 364:
                print(f"cache: {t + 1}/365", flush=True)
        X.flush()
        del X

    os.replace(tmp_x, final_x)
    with open(cache_dir / "sample_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "year", "time_idx", "date"])
        writer.writeheader()
        for i, date in enumerate(dates):
            writer.writerow({"sample_id": i, "year": args.year, "time_idx": i, "date": date})

    cache_bytes_before_meta = directory_size(cache_dir)
    meta = {
        "format": "cmcc_prediction_input_cache_v1",
        "complete": True,
        "x_file": "X.npy",
        "dtype": "float16",
        "N": 365,
        "C": 4,
        "coarse_shape": [448, 304],
        "x_shape": [365, 4, 448, 304],
        "years": [args.year],
        "input_vars": ["pr", "tos", "sic"],
        "time_offsets": [0],
        "add_mask_channel": True,
        "ice_mask_thr": 0.15,
        "channel_names": ["pr_d", "tos_d", "sic_d", "mask_d"],
        "feature_norm_ranges": {"pr": [0.0, 0.05], "tos": [270.0, 310.0], "sic": [0.0, 1.0]},
        "source_file": str(source),
        "source_file_bytes": source_bytes,
        "created_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 最终重新以 mmap 打开，确认 NPY 头、形状、类型和值域都可读。
    check = np.load(final_x, mmap_mode="r")
    if check.shape != (365, 4, 448, 304) or check.dtype != np.float16:
        raise RuntimeError(f"缓存校验失败: shape={check.shape}, dtype={check.dtype}")
    mins = np.asarray([np.min(check[:, c]) for c in range(4)], dtype=float)
    maxs = np.asarray([np.max(check[:, c]) for c in range(4)], dtype=float)
    cache_bytes = directory_size(cache_dir)
    free_bytes = shutil.disk_usage(cache_dir).free

    print("\n=== 2050缓存完成 ===" if args.year == 2050 else "\n=== 缓存完成 ===")
    print(f"原年度NetCDF : {source}")
    print(f"原文件大小    : {human_size(source_bytes)} ({source_bytes:,} bytes)")
    print(f"缓存目录      : {cache_dir}")
    print(f"缓存总大小    : {human_size(cache_bytes)} ({cache_bytes:,} bytes)")
    print(f"缓存/原文件   : {cache_bytes / source_bytes:.3f} 倍")
    print(f"磁盘剩余      : {human_size(free_bytes)}")
    print(f"X shape/dtype : {check.shape} / {check.dtype}")
    print(f"通道最小值    : {mins.tolist()}")
    print(f"通道最大值    : {maxs.tolist()}")
    print("通道顺序      : pr_d, tos_d, sic_d, mask_d")


if __name__ == "__main__":
    main()
