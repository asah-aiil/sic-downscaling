#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_temporal_memmap_cache.py
把 d-2、d-1、d 时间窗口样本预生成成连续 .npy memmap 缓存，避免训练时反复随机读取 NetCDF。

输出示例：
  /home/ubuntu/sic_dsc/data/cache_temporal_T3_float16/
    X.npy                # (N, C, Hc, Wc), float16/float32
    Y.npy                # (N, 1, Hh, Wh), float16/float32
    meta.json
    sample_index.csv

推荐先检查磁盘空间：
  df -h /home/ubuntu/sic_dsc/data

运行：
  cd /home/ubuntu/sic_dsc/code
  python build_temporal_memmap_cache.py

如需强制重建：
  python build_temporal_memmap_cache.py --force
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm

from config import Config


def get_time_offsets() -> List[int]:
    return list(getattr(Config, "TIME_OFFSETS", getattr(Config, "TEMPORAL_OFFSETS", [-2, -1, 0])))


def get_cache_dir(dtype_name: str) -> Path:
    default = f"/home/ubuntu/sic_dsc/data/cache_temporal_T{len(get_time_offsets())}_{dtype_name}"
    return Path(getattr(Config, "TEMPORAL_CACHE_DIR", default))


def _open_ds(path: str) -> xr.Dataset:
    engine = getattr(Config, "XR_ENGINE", None)
    if engine:
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            print(f"[WARN] engine={engine!r} 打开失败，回退默认引擎: {path}\n  {repr(e)}")
    return xr.open_dataset(path)


def _ensure_tyxdims(ds: xr.Dataset) -> xr.Dataset:
    dims = list(ds.dims)
    if "time" in dims and "y" in dims and "x" in dims:
        return ds.transpose("time", "y", "x")

    rename = {}
    if "latitude" in dims:
        rename["latitude"] = "y"
    if "longitude" in dims:
        rename["longitude"] = "x"
    if rename:
        ds = ds.rename(rename)
        dims = list(ds.dims)
        if "time" in dims and "y" in dims and "x" in dims:
            return ds.transpose("time", "y", "x")

    if "time" in dims and len(dims) == 3:
        others = [d for d in dims if d != "time"]
        return ds.transpose("time", *others)
    return ds


def _nan0(a: np.ndarray) -> np.ndarray:
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)


NORM_STATS: Dict[str, Tuple[float, float]] = {
    "t2m": (220.0, 320.0),
    "u10": (-30.0, 30.0),
    "v10": (-30.0, 30.0),
    "msl": (97000.0, 105000.0),
    "tp": (0.0, 0.05),
    "ssr": (0.0, 1.0e7),
    "tcc": (0.0, 1.0),
    "sst": (270.0, 310.0),
    "sic": (0.0, 1.0),
}


def _normalize01(arr: np.ndarray, var: str) -> np.ndarray:
    arr = _nan0(arr)
    if var not in NORM_STATS:
        return np.clip(arr, 0.0, 1.0).astype("float32", copy=False)
    vmin, vmax = NORM_STATS[var]
    out = (arr - vmin) / (vmax - vmin + 1e-6)
    return np.clip(out, 0.0, 1.0).astype("float32", copy=False)


def _scale_sic01(arr: np.ndarray) -> np.ndarray:
    arr = _nan0(arr)
    max_val = float(np.max(arr)) if arr.size else 0.0
    scale = 100.0 if max_val > 1.5 else 1.0
    return np.clip(arr / scale, 0.0, 1.0).astype("float32", copy=False)


def _load_year_arrays(ds_c: xr.Dataset, ds_h: xr.Dataset, input_vars: List[str], sic_hr_var: str,
                      load_to_ram: bool) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """按年预读到 RAM。若 load_to_ram=False，返回 (None, None)。"""
    if not load_to_ram:
        return None, None

    coarse: Dict[str, np.ndarray] = {}
    print("  [RAM] 读取该年 coarse 变量到内存...")
    for var in input_vars:
        coarse[var] = _nan0(np.asarray(ds_c[var].values))

    print("  [RAM] 读取该年 HR SIC 到内存...")
    hr = _nan0(np.asarray(ds_h[sic_hr_var].values))
    return coarse, hr


def _read_coarse(coarse_ram: Optional[Dict[str, np.ndarray]], ds_c: xr.Dataset, var: str, t_idx: int) -> np.ndarray:
    if coarse_ram is not None:
        return coarse_ram[var][t_idx]
    return _nan0(np.asarray(ds_c[var].isel(time=t_idx).values))


def _read_hr(hr_ram: Optional[np.ndarray], ds_h: xr.Dataset, sic_hr_var: str, t_idx: int) -> np.ndarray:
    if hr_ram is not None:
        return hr_ram[t_idx]
    return _nan0(np.asarray(ds_h[sic_hr_var].isel(time=t_idx).values))


def collect_samples() -> Tuple[List[Dict[str, object]], Tuple[int, int], Tuple[int, int], List[str]]:
    years = list(getattr(Config, "YEARS"))
    input_vars = list(getattr(Config, "INPUT_VARS"))
    sic_hr_var = getattr(Config, "SIC_HR_VAR", "z")
    offsets = get_time_offsets()
    min_off, max_off = min(offsets), max(offsets)

    samples: List[Dict[str, object]] = []
    coarse_shape: Optional[Tuple[int, int]] = None
    hr_shape: Optional[Tuple[int, int]] = None

    channel_names: List[str] = []
    for off in offsets:
        tag = "d" if off == 0 else f"d{off:+d}".replace("+", "+")
        # 为了保持和之前打印一致，把 d-2/d-1/d 写清楚
        tag = "d" if off == 0 else f"d{off}"
        for v in input_vars:
            channel_names.append(f"{v}_{tag}")
    if bool(getattr(Config, "ADD_MASK_CHANNEL", True)):
        channel_names.append("mask_d")

    for year in years:
        f_c = os.path.join(getattr(Config, "COARSE_DIR"), f"input_25km_{year}_25km.nc")
        f_h = os.path.join(getattr(Config, "HIGHRES_DIR"), f"Bremen_{year}.nc")
        if not os.path.exists(f_c) or not os.path.exists(f_h):
            print(f"[WARN] 文件缺失，跳过 {year}: {f_c} / {f_h}")
            continue

        ds_c = _ensure_tyxdims(_open_ds(f_c))
        ds_h = _ensure_tyxdims(_open_ds(f_h))
        try:
            t_c = ds_c["time"].values
            t_h = ds_h["time"].values
            if t_c.shape != t_h.shape or not np.array_equal(t_c, t_h):
                raise RuntimeError(f"{year} coarse 与 HR time 不一致")

            if coarse_shape is None:
                coarse_shape = tuple(ds_c[input_vars[0]].isel(time=0).shape)
            if hr_shape is None:
                hr_shape = tuple(ds_h[sic_hr_var].isel(time=0).shape)

            ntime = int(ds_c.sizes["time"])
            start = max(0, -min_off)
            end_exclusive = ntime - max(0, max_off)
            for t_idx in range(start, end_exclusive):
                samples.append({
                    "sample_id": len(samples),
                    "year": int(year),
                    "time_idx": int(t_idx),
                    "date": str(np.datetime_as_string(t_c[t_idx], unit="D")),
                })
        finally:
            ds_c.close()
            ds_h.close()

    if coarse_shape is None or hr_shape is None or not samples:
        raise RuntimeError("没有有效样本，请检查 Config 路径与年份。")

    return samples, coarse_shape, hr_shape, channel_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="如缓存目录已存在，先删除再重建")
    parser.add_argument("--dtype", default=getattr(Config, "TEMPORAL_CACHE_DTYPE", "float16"), choices=["float16", "float32"])
    parser.add_argument("--no-load-year-to-ram", action="store_true", help="不把单年 coarse/HR 预读到内存，省内存但更慢")
    args = parser.parse_args()

    dtype_name = args.dtype
    dtype = np.float16 if dtype_name == "float16" else np.float32
    cache_dir = get_cache_dir(dtype_name)

    if cache_dir.exists():
        if args.force:
            print(f"[Cache] 删除旧缓存: {cache_dir}")
            shutil.rmtree(cache_dir)
        else:
            raise SystemExit(f"缓存目录已存在: {cache_dir}\n如需重建，请加 --force")
    cache_dir.mkdir(parents=True, exist_ok=True)

    offsets = get_time_offsets()
    input_vars = list(getattr(Config, "INPUT_VARS"))
    sic_hr_var = getattr(Config, "SIC_HR_VAR", "z")
    add_mask = bool(getattr(Config, "ADD_MASK_CHANNEL", True))
    ice_thr = float(getattr(Config, "ICE_MASK_THR", 0.15))
    expected_c = len(input_vars) * len(offsets) + (1 if add_mask else 0)
    load_to_ram = not args.no_load_year_to_ram

    print("=== Collect samples ===")
    samples, coarse_shape, hr_shape, channel_names = collect_samples()
    N = len(samples)
    C = expected_c
    Hc, Wc = coarse_shape
    Hh, Wh = hr_shape

    print(f"samples      = {N}")
    print(f"X shape      = {(N, C, Hc, Wc)}")
    print(f"Y shape      = {(N, 1, Hh, Wh)}")
    print(f"dtype        = {dtype_name}")
    print(f"cache_dir    = {cache_dir}")
    print(f"load_to_ram  = {load_to_ram}")
    approx_gb = (N * C * Hc * Wc + N * 1 * Hh * Wh) * np.dtype(dtype).itemsize / 1024**3
    print(f"approx size  = {approx_gb:.1f} GiB")

    X_path = cache_dir / "X.npy"
    Y_path = cache_dir / "Y.npy"
    X = np.lib.format.open_memmap(X_path, mode="w+", dtype=dtype, shape=(N, C, Hc, Wc))
    Y = np.lib.format.open_memmap(Y_path, mode="w+", dtype=dtype, shape=(N, 1, Hh, Wh))

    # 写 meta 和 sample_index
    meta = {
        "format": "temporal_memmap_cache_v1",
        "x_file": "X.npy",
        "y_file": "Y.npy",
        "dtype": dtype_name,
        "N": N,
        "C": C,
        "coarse_shape": [Hc, Wc],
        "hr_shape": [Hh, Wh],
        "x_shape": [N, C, Hc, Wc],
        "y_shape": [N, 1, Hh, Wh],
        "years": list(getattr(Config, "YEARS")),
        "input_vars": input_vars,
        "time_offsets": offsets,
        "add_mask_channel": add_mask,
        "ice_mask_thr": ice_thr,
        "sic_hr_var": sic_hr_var,
        "channel_names": channel_names,
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(cache_dir / "sample_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "year", "time_idx", "date"])
        writer.writeheader()
        writer.writerows(samples)

    # year -> sample list
    by_year: Dict[int, List[Dict[str, object]]] = {}
    for s in samples:
        by_year.setdefault(int(s["year"]), []).append(s)

    t0 = time.time()
    for year in sorted(by_year):
        f_c = os.path.join(getattr(Config, "COARSE_DIR"), f"input_25km_{year}_25km.nc")
        f_h = os.path.join(getattr(Config, "HIGHRES_DIR"), f"Bremen_{year}.nc")
        print(f"\n===== Build year {year} | samples={len(by_year[year])} =====")
        ds_c = _ensure_tyxdims(_open_ds(f_c))
        ds_h = _ensure_tyxdims(_open_ds(f_h))
        coarse_ram = None
        hr_ram = None
        try:
            coarse_ram, hr_ram = _load_year_arrays(ds_c, ds_h, input_vars, sic_hr_var, load_to_ram)

            for s in tqdm(by_year[year], desc=f"year {year}"):
                sid = int(s["sample_id"])
                t_idx = int(s["time_idx"])
                ch = 0

                for off in offsets:
                    tt = t_idx + int(off)
                    for var in input_vars:
                        arr = _read_coarse(coarse_ram, ds_c, var, tt)
                        if var == "sic":
                            arr = _scale_sic01(arr)
                            arr = _normalize01(arr, "sic")
                        else:
                            arr = _normalize01(arr, var)
                        X[sid, ch] = arr.astype(dtype, copy=False)
                        ch += 1

                if add_mask:
                    sic_today = _read_coarse(coarse_ram, ds_c, "sic", t_idx)
                    sic_today = _scale_sic01(sic_today)
                    mask = (sic_today >= ice_thr).astype("float32")
                    X[sid, ch] = mask.astype(dtype, copy=False)

                hr = _read_hr(hr_ram, ds_h, sic_hr_var, t_idx)
                hr = _scale_sic01(hr)
                Y[sid, 0] = hr.astype(dtype, copy=False)

            X.flush(); Y.flush()
        finally:
            ds_c.close(); ds_h.close()
            del coarse_ram, hr_ram

    X.flush(); Y.flush()
    total = time.time() - t0
    print("\n=== DONE ===")
    print(f"cache_dir = {cache_dir}")
    print(f"time      = {total/60:.1f} min")
    print("下一步：把 data_loader_temporal_memmap.py 覆盖为 data_loader.py，然后运行 benchmark_memmap_loader.py。")


if __name__ == "__main__":
    main()
