#!/usr/bin/env python3
"""将 CMCC pr/tos/siconc 逐年处理到现有 25 km 模型输入网格。

默认先处理 2050 年；使用 --all-years 时按 2050 -> 2015 倒序处理。
程序逐日读取、逐日写盘，不会把全年三种原始场同时载入内存。
"""

from __future__ import annotations

import argparse
import gc
import os
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


DEFAULT_ROOT = Path("/home/ubuntu/sic_dsc/data_cmcc/data")
DEFAULT_TEMPLATE = Path("/home/ubuntu/sic_dsc/data/dataset/input/input_25km_2014_25km.nc")
DEFAULT_OUTPUT = Path("/home/ubuntu/sic_dsc/data_cmcc/processed_25km")

SOURCE_SPECS = {
    "pr": {
        "subdir": "pr",
        "pattern": "pr_day_CMCC-CM2-VHR4_highres-future_r1i1p1f1_gn_{year}0101-{year}1231.nc",
        "source_var": "pr",
        "out_units": "m",
        "long_name": "Daily total precipitation",
    },
    "tos": {
        "subdir": "tos",
        "pattern": "tos_Oday_CMCC-CM2-VHR4_highres-future_r1i1p1f1_gn_{year}0101-{year}1231.nc",
        "source_var": "tos",
        "out_units": "K",
        "long_name": "Sea surface temperature",
    },
    "sic": {
        "subdir": "sic",
        "pattern": "siconc_SIday_CMCC-CM2-VHR4_highres-future_r1i1p1f1_gn_{year}0101-{year}1231.nc",
        "source_var": "siconc",
        "out_units": "%",
        "long_name": "Sea ice concentration",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, default=2050, help="只处理指定年份，默认 2050")
    p.add_argument("--all-years", action="store_true", help="按 end-year 到 start-year 倒序批处理")
    p.add_argument("--start-year", type=int, default=2015)
    p.add_argument("--end-year", type=int, default=2050)
    p.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--radius-km", type=float, default=100.0, help="最近邻最大搜索距离")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def find_grid_mapping(ds: xr.Dataset) -> tuple[str, dict]:
    for name in ("polar_stereographic", "crs", "spatial_ref"):
        if name in ds.variables:
            attrs = dict(ds[name].attrs)
            if attrs:
                return name, attrs
    raise RuntimeError("模板中未找到 polar_stereographic/crs/spatial_ref 投影变量")


def crs_from_attrs(attrs: dict) -> CRS:
    for key in ("spatial_ref", "crs_wkt"):
        if attrs.get(key):
            return CRS.from_wkt(attrs[key])
    return CRS.from_cf(attrs)


def load_target(template: Path):
    with xr.open_dataset(template, decode_times=False) as ds:
        x = np.asarray(ds["x"].values, dtype=np.float64)
        y = np.asarray(ds["y"].values, dtype=np.float64)
        mapping_name, mapping_attrs = find_grid_mapping(ds)
        x_attrs, y_attrs = dict(ds["x"].attrs), dict(ds["y"].attrs)
    xx, yy = np.meshgrid(x, y)
    transformer = Transformer.from_crs(crs_from_attrs(mapping_attrs), "EPSG:4326", always_xy=True)
    target_lon, target_lat = transformer.transform(xx, yy)
    return x, y, target_lon, target_lat, mapping_name, mapping_attrs, x_attrs, y_attrs


def locate_source(root: Path, logical_var: str, year: int) -> Path:
    spec = SOURCE_SPECS[logical_var]
    exact = root / spec["subdir"] / spec["pattern"].format(year=year)
    if exact.exists():
        return exact
    candidates = sorted((root / spec["subdir"]).glob(f"*_{year}0101-{year}1231.nc"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"{logical_var} {year} 源文件无法唯一确定：{candidates}")
    return candidates[0]


def coord_names(ds: xr.Dataset) -> tuple[str, str]:
    lon = next((n for n in ("longitude", "lon", "nav_lon") if n in ds.variables), None)
    lat = next((n for n in ("latitude", "lat", "nav_lat") if n in ds.variables), None)
    if lon is None or lat is None:
        raise RuntimeError("源文件中未找到经纬度变量")
    return lon, lat


def sphere_xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.deg2rad(lon)
    lat_r = np.deg2rad(lat)
    cos_lat = np.cos(lat_r)
    return np.column_stack((cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r))).astype(np.float32)


def build_nearest_mapping(ds: xr.Dataset, target_lon: np.ndarray, target_lat: np.ndarray, radius_km: float):
    lon_name, lat_name = coord_names(ds)
    lon = np.asarray(ds[lon_name].values)
    lat = np.asarray(ds[lat_name].values)
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    # 只保留可能影响北极目标网格的源像元，显著降低 KDTree 内存。
    min_needed_lat = max(-90.0, float(np.nanmin(target_lat)) - 3.0)
    valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 720) & (np.abs(lat) <= 90)
    valid &= lat >= min_needed_lat
    flat_source_indices = np.flatnonzero(valid.ravel())
    if flat_source_indices.size == 0:
        raise RuntimeError("没有有效源网格坐标")

    tree = cKDTree(sphere_xyz(lon.ravel()[flat_source_indices], lat.ravel()[flat_source_indices]))
    earth_radius_km = 6371.0088
    max_chord = 2.0 * np.sin((radius_km / earth_radius_km) / 2.0)
    distances, local_indices = tree.query(
        sphere_xyz(target_lon.ravel(), target_lat.ravel()), k=1, distance_upper_bound=max_chord
    )
    mapped = np.full(target_lon.size, -1, dtype=np.int64)
    ok = np.isfinite(distances) & (local_indices < flat_source_indices.size)
    mapped[ok] = flat_source_indices[local_indices[ok]]
    print(f"    映射覆盖率: {ok.mean() * 100:.3f}%  源有效点: {flat_source_indices.size:,}")
    return mapped, target_lon.shape


def convert_values(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values[~np.isfinite(values) | (np.abs(values) >= 1.0e19)] = np.nan
    if name == "pr":
        values *= np.float32(86.4)  # kg m-2 s-1 -> m/day: 86400 / 1000
        values[values < 0] = 0
    elif name == "tos":
        values += np.float32(273.15)
    elif name == "sic":
        finite = values[np.isfinite(values)]
        if finite.size and np.nanmax(finite) <= 1.5:
            values *= np.float32(100.0)
        values = np.clip(values, 0.0, 100.0)
    return values


def make_dates(year: int) -> list[datetime]:
    dates, day = [], datetime(year, 1, 1)
    while day.year == year:
        if not (day.month == 2 and day.day == 29):
            dates.append(day)
        day += timedelta(days=1)
    assert len(dates) == 365
    return dates


def create_output(path: Path, year: int, x, y, mapping_name, mapping_attrs, x_attrs, y_attrs):
    root = nc.Dataset(path, "w", format="NETCDF4")
    root.createDimension("time", 365)
    root.createDimension("y", len(y))
    root.createDimension("x", len(x))
    xv = root.createVariable("x", "f8", ("x",))
    yv = root.createVariable("y", "f8", ("y",))
    xv[:] = x
    yv[:] = y
    xv.setncatts(x_attrs)
    yv.setncatts(y_attrs)
    tv = root.createVariable("time", "f8", ("time",))
    tv.units = f"days since {year}-01-01 00:00:00"
    tv.calendar = "proleptic_gregorian"
    tv.standard_name = "time"
    tv[:] = nc.date2num(make_dates(year), tv.units, tv.calendar)
    crsv = root.createVariable(mapping_name, "i4")
    crsv.setncatts(mapping_attrs)
    for name, spec in SOURCE_SPECS.items():
        v = root.createVariable(
            name, "f4", ("time", "y", "x"), fill_value=np.float32(np.nan),
            zlib=True, complevel=4, shuffle=True, chunksizes=(1, min(112, len(y)), min(76, len(x)))
        )
        v.units = spec["out_units"]
        v.long_name = spec["long_name"]
        v.grid_mapping = mapping_name
    root.title = "CMCC-CM2-VHR4 daily fields regridded to model 25 km grid"
    root.history = f"created {datetime.now().isoformat(timespec='seconds')}"
    root.source_calendar = "365_day; February 29 is omitted"
    root.regridding = "nearest neighbour on the sphere"
    return root


def validate_output(path: Path, year: int) -> tuple[bool, str]:
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            if dict(ds.sizes) != {"time": 365, "y": 448, "x": 304}:
                return False, f"尺寸错误: {dict(ds.sizes)}"
            for name in SOURCE_SPECS:
                if name not in ds or ds[name].shape != (365, 448, 304):
                    return False, f"变量 {name} 缺失或尺寸错误"
                sample = np.asarray(ds[name].isel(time=[0, 182, 364]).values)
                if not np.isfinite(sample).any():
                    return False, f"变量 {name} 抽样全部无效"
        return True, "365天、448×304、三个变量及抽样有效值检查通过"
    except Exception as exc:
        return False, str(exc)


def process_year(args, year: int, target_info) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / f"input_25km_{year}_cmcc.nc"
    temp_path = final_path.with_suffix(".tmp.nc")

    if final_path.exists() and not args.overwrite:
        ok, msg = validate_output(final_path, year)
        if ok:
            print(f"[{year}] 已存在且校验通过，跳过: {final_path}")
            return
        print(f"[{year}] 已有文件校验失败，将重新生成: {msg}")
    if temp_path.exists():
        temp_path.unlink()

    x, y, target_lon, target_lat, mapping_name, mapping_attrs, x_attrs, y_attrs = target_info
    print(f"[{year}] 开始 -> {final_path}")
    out = create_output(temp_path, year, x, y, mapping_name, mapping_attrs, x_attrs, y_attrs)
    try:
        for logical_name, spec in SOURCE_SPECS.items():
            source_path = locate_source(args.source_root, logical_name, year)
            print(f"  {logical_name}: {source_path}")
            with xr.open_dataset(source_path, decode_times=True, use_cftime=True, mask_and_scale=True) as src:
                var = src[spec["source_var"]]
                time_dim = next((d for d in var.dims if d in ("time", "valid_time")), None)
                if time_dim is None or var.sizes[time_dim] != 365:
                    raise RuntimeError(f"{source_path} 时间长度不是365: {var.sizes}")
                mapping, target_shape = build_nearest_mapping(src, target_lon, target_lat, args.radius_km)
                for ti in range(365):
                    source_flat = np.asarray(var.isel({time_dim: ti}).values, dtype=np.float32).ravel()
                    target_flat = np.full(mapping.size, np.nan, dtype=np.float32)
                    ok = mapping >= 0
                    target_flat[ok] = source_flat[mapping[ok]]
                    out[logical_name][ti, :, :] = convert_values(logical_name, target_flat.reshape(target_shape))
                    if ti == 0 or (ti + 1) % 50 == 0 or ti == 364:
                        print(f"    {logical_name}: {ti + 1}/365", flush=True)
                out.sync()
            del mapping
            gc.collect()
    except Exception:
        out.close()
        raise
    out.close()

    ok, msg = validate_output(temp_path, year)
    if not ok:
        raise RuntimeError(f"[{year}] 输出校验失败，临时文件保留在 {temp_path}: {msg}")
    os.replace(temp_path, final_path)
    print(f"[{year}] 完成: {msg}\n  {final_path}")


def main() -> None:
    args = parse_args()
    if not args.template.exists():
        raise FileNotFoundError(args.template)
    target_info = load_target(args.template)
    years = range(args.end_year, args.start_year - 1, -1) if args.all_years else [args.year]
    for year in years:
        process_year(args, int(year), target_info)


if __name__ == "__main__":
    main()
