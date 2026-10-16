#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按天用 A* 比较 CMCC 降尺度前后东北航道可通行性。

输出一行/日期/阈值，不计算通航率或窗口统计。默认阈值为 0.15、0.30、0.70。
降尺度前使用 CMCC 25 km SIC，降尺度后使用 TOS+PR 模型的 6.25 km SIC。
A* 默认允许在既有东北航道区域掩膜与静态海洋的交集内搜索；使用
--no-route-mask 时仅使用静态海洋和 SIC 阈值。
"""
from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from pyproj import Transformer


DEFAULT_BEFORE = "/home/ubuntu/sic_dsc/data_cmcc/processed_25km/input_25km_{year}_cmcc.nc"
DEFAULT_AFTER = "/home/ubuntu/sic_dsc/data_cmcc/predict_pr_tos_sic_mask_{year}/sic_pred_6p25km_cmcc_pr_tos_sic_mask_{year}.nc"
DEFAULT_TEMPLATE = "/home/ubuntu/sic_dsc/data/dataset/target/Bremen_2024.nc"
DEFAULT_ROUTE_MASK = "/home/ubuntu/sic_dsc/data/dataset/mask/eurasia_shipping_masks_6p25km_with_arctic.nc"
DEFAULT_OUTPUT = "/home/ubuntu/sic_dsc/data_cmcc/astar_daily_passability_2015_2050.csv"


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMCC daily A* passability comparison")
    p.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2051)))
    p.add_argument("--before-template", default=DEFAULT_BEFORE,
                   help="25 km SIC path template; use {year}")
    p.add_argument("--after-template", default=DEFAULT_AFTER,
                   help="6.25 km prediction path template; use {year}")
    p.add_argument("--grid-template", default=DEFAULT_TEMPLATE,
                   help="6.25 km Bremen template containing x/y and static land NaNs")
    p.add_argument("--route-mask-file", default=DEFAULT_ROUTE_MASK,
                   help="6.25 km Northeast Passage region-mask NetCDF")
    p.add_argument("--route-mask-var", default="mask_all_shipping_seas",
                   help="Northeast Passage region-mask variable")
    p.add_argument("--no-route-mask", action="store_true",
                   help="Disable the shipping-sea mask; use only land and SIC threshold")
    p.add_argument("--output-csv", default=DEFAULT_OUTPUT)
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.30, 0.70])
    p.add_argument("--start-lon", type=float, default=-174.0)
    p.add_argument("--start-lat", type=float, default=61.5)
    p.add_argument("--end-lon", type=float, default=20.0)
    p.add_argument("--end-lat", type=float, default=72.0)
    p.add_argument("--max-endpoint-snap-km", type=float, default=300.0,
                   help="端点吸附到航道海洋像元的最大距离；默认 300 km")
    p.add_argument("--before-var", default=None)
    p.add_argument("--after-var", default=None)
    p.add_argument("--time-name", default=None)
    p.add_argument("--max-days", type=int, default=None,
                   help="每年最多处理多少天，便于 smoke test")
    p.add_argument("--start-date", default=None, help="只处理不早于该日期的日期，例如 2050-08-01")
    p.add_argument("--end-date", default=None, help="只处理不晚于该日期的日期，例如 2050-08-03")
    p.add_argument("--stride-after", type=int, default=1,
                   help="A* 后处理网格步长；默认 1，保持 6.25 km")
    p.add_argument("--no-diagonal", action="store_true")
    return p.parse_args()


def find_time(ds: xr.Dataset, requested: str | None) -> str:
    if requested and requested in ds:
        return requested
    for name in ("time", "valid_time", "date"):
        if name in ds.coords or name in ds.variables:
            return name
    raise RuntimeError("找不到 time/valid_time/date 时间坐标")


def find_var(ds: xr.Dataset, requested: str | None, preferred: Iterable[str]) -> str:
    if requested:
        if requested not in ds:
            raise RuntimeError(f"找不到变量 {requested}；已有变量: {list(ds.data_vars)}")
        return requested
    for name in preferred:
        if name in ds.data_vars or name in ds.variables:
            return name
    candidates = []
    for name, var in ds.data_vars.items():
        if var.ndim >= 2 and np.issubdtype(var.dtype, np.number):
            candidates.append(name)
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"无法自动识别 SIC 变量；候选: {candidates}")


def norm_sic(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(a)
    if finite.any() and float(np.nanmax(a)) > 1.5:
        a = a / 100.0
    return np.clip(a, 0.0, 1.0)


def xy_from_ds(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    for xn, yn in (("x", "y"), ("xc", "yc")):
        if xn in ds.variables and yn in ds.variables:
            x = np.asarray(ds[xn].values, dtype=np.float64)
            y = np.asarray(ds[yn].values, dtype=np.float64)
            if x.ndim == 1 and y.ndim == 1:
                return x, y
    raise RuntimeError("网格文件缺少一维 x/y 坐标")


def get_2d(arr: np.ndarray, time_index: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 3:
        return np.asarray(a[time_index])
    if a.ndim == 2:
        return a
    raise RuntimeError(f"SIC 变量必须为 2D 或 3D，实际 {a.shape}")


def dates(ds: xr.Dataset, tname: str) -> list[str]:
    vals = ds[tname].values
    result = []
    for v in vals:
        try:
            result.append(str(np.datetime_as_string(np.datetime64(v), unit="D")))
        except Exception:
            result.append(str(v)[:10])
    return result


def nearest_xy(
    x: np.ndarray,
    y: np.ndarray,
    lon: float,
    lat: float,
    allowed: np.ndarray | None = None,
) -> tuple[int, int]:
    """返回离经纬度最近的网格点；给定 allowed 时仅在有效航道像元中查找。"""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    tx, ty = tr.transform(lon, lat)
    d2 = (y[:, None] - ty) ** 2 + (x[None, :] - tx) ** 2
    if allowed is not None:
        if allowed.shape != d2.shape:
            raise RuntimeError(f"端点有效区形状不匹配: allowed={allowed.shape}, grid={d2.shape}")
        if not np.any(allowed):
            raise RuntimeError("东北航道掩膜与海洋掩膜相交后为空")
        d2 = np.where(allowed, d2, np.inf)
    return tuple(int(v) for v in np.unravel_index(np.nanargmin(d2), d2.shape))


def map_boolean_mask(
    source: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    label: str,
) -> np.ndarray:
    """按投影坐标最近邻映射静态布尔掩膜，并拒绝不重合的网格。"""
    for name, axis in (
        ("source_x", source_x), ("source_y", source_y),
        ("target_x", target_x), ("target_y", target_y),
    ):
        if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
            raise RuntimeError(f"{label} 的 {name} 必须是一维非空有限坐标")
    if source.shape != (len(source_y), len(source_x)):
        raise RuntimeError(
            f"{label} 的数组与坐标不一致: array={source.shape}, "
            f"coordinates={(len(source_y), len(source_x))}"
        )
    ix = np.abs(source_x[None, :] - target_x[:, None]).argmin(axis=1)
    iy = np.abs(source_y[None, :] - target_y[:, None]).argmin(axis=1)
    dx = np.abs(source_x[ix] - target_x)
    dy = np.abs(source_y[iy] - target_y)
    sx = float(np.median(np.abs(np.diff(source_x)))) if len(source_x) > 1 else np.inf
    sy = float(np.median(np.abs(np.diff(source_y)))) if len(source_y) > 1 else np.inf
    if sx <= 0.0 or sy <= 0.0 or not np.isfinite([sx, sy]).all():
        raise RuntimeError(f"{label} 的源网格坐标间距无效: dx={sx}, dy={sy}")
    max_dx = float(dx.max()) if dx.size else 0.0
    max_dy = float(dy.max()) if dy.size else 0.0
    if max_dx > 0.51 * sx or max_dy > 0.51 * sy:
        raise RuntimeError(
            f"{label} 与目标网格未对齐: max_dx={max_dx:.1f} m, "
            f"max_dy={max_dy:.1f} m"
        )
    return source[np.ix_(iy, ix)].astype(bool)


def load_route_mask(
    path: str,
    variable: str,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    """读取既有 6.25 km 东北航道掩膜，并按 x/y 核对目标网格。"""
    mask_path = Path(path).expanduser()
    if not mask_path.is_file():
        raise FileNotFoundError(f"东北航道掩膜不存在: {mask_path}")
    with xr.open_dataset(mask_path, decode_times=False) as ds:
        if variable not in ds:
            available = ", ".join(ds.data_vars)
            raise RuntimeError(f"航道掩膜变量 {variable!r} 不存在；可选: {available}")
        data = ds[variable].squeeze(drop=True)
        if "y" in data.dims and "x" in data.dims:
            data = data.transpose("y", "x")
        values = np.asarray(data.values, dtype=np.float32)
        mask_x, mask_y = xy_from_ds(ds)
    if values.ndim != 2:
        raise RuntimeError(f"航道掩膜必须为二维，实际 shape={values.shape}")
    mask = np.isfinite(values) & (values > 0.5)
    if not mask.any():
        raise RuntimeError(f"航道掩膜 {variable!r} 没有有效像元")
    return map_boolean_mask(
        mask, mask_x, mask_y, target_x, target_y, "东北航道掩膜"
    )


def snap_distance_km(
    x: np.ndarray,
    y: np.ndarray,
    point: tuple[int, int],
    lon: float,
    lat: float,
) -> float:
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    tx, ty = tr.transform(lon, lat)
    py, px = point
    return float(np.hypot(x[px] - tx, y[py] - ty) / 1000.0)


def static_land_mask(grid_path: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """从 Bremen 高分辨率模板的 z/首个变量推断陆地，并映射到目标 x/y。"""
    with xr.open_dataset(grid_path) as ds:
        gx, gy = xy_from_ds(ds)
        var = ds["z"] if "z" in ds.data_vars else next(v for v in ds.data_vars.values() if v.ndim >= 2)
        a = np.asarray(var.isel({find_time(ds, None): 0}).values if var.ndim == 3 else var.values)
        source_land = ~np.isfinite(a)
    if np.array_equal(gx, x) and np.array_equal(gy, y):
        return source_land.astype(bool)
    # 投影网格规则且目标为同一极投影；最近邻映射比按索引假设更安全。
    ix = np.abs(gx[None, :] - x[:, None]).argmin(axis=1)
    iy = np.abs(gy[None, :] - y[:, None]).argmin(axis=1)
    return source_land[np.ix_(iy, ix)].astype(bool)


def coarse_boolean_from_hr(mask_hr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把 6.25 km 布尔掩膜映射到 25 km；4x4 内任一像元有效即保留。"""
    hy, hx = mask_hr.shape
    cy, cx = shape
    if hy == cy and hx == cx:
        return mask_hr.copy()
    if hy == cy * 4 and hx == cx * 4:
        return mask_hr.reshape(cy, 4, cx, 4).any(axis=(1, 3))
    # 仅用于兼容非 4 倍网格：最近邻映射。
    yy = np.linspace(0, hy - 1, cy).round().astype(int)
    xx = np.linspace(0, hx - 1, cx).round().astype(int)
    return mask_hr[np.ix_(yy, xx)]


def astar(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int], diagonal: bool) -> tuple[bool, int]:
    h, w = passable.shape
    sy, sx = start; gy, gx = goal
    if not (0 <= sy < h and 0 <= sx < w and 0 <= gy < h and 0 <= gx < w):
        return False, -1
    if not passable[sy, sx] or not passable[gy, gx]:
        return False, -1
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    if diagonal:
        moves += [(-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5)]
    g = {start: 0.0}; heap = [(0.0, 0.0, start)]; visited = set()
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur in visited: continue
        visited.add(cur)
        if cur == goal:
            return True, int(round(cost))
        y, x = cur
        for dy, dx, step in moves:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w and passable[ny, nx]): continue
            nxt = (ny, nx); nc = cost + step
            if nc < g.get(nxt, float("inf")):
                g[nxt] = nc
                est = nc + float(np.hypot(gy - ny, gx - nx))
                heapq.heappush(heap, (est, nc, nxt))
    return False, -1


def evaluate_thresholds(
    sic: np.ndarray,
    land: np.ndarray,
    route: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    thresholds: list[float],
    diagonal: bool,
) -> dict[float, bool]:
    """按从严到宽运行 A*；较严阈值已通时利用单调性跳过后续搜索。"""
    result: dict[float, bool] = {}
    already_passable = False
    finite_ocean_route = np.isfinite(sic) & (~land) & route
    for threshold in sorted(set(thresholds)):
        if already_passable:
            result[threshold] = True
            continue
        passable = finite_ocean_route & (sic <= threshold)
        ok, _ = astar(passable, start, goal, diagonal)
        result[threshold] = ok
        already_passable = ok
    return result


OUTPUT_FIELDS = ["year", "date", "threshold", "before_passable", "after_passable"]


def save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """覆盖写出当前累计结果；每完成一年调用一次，保留长任务的阶段结果。"""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    a = args()
    out = Path(a.output_csv).expanduser().resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    thresholds = [float(t) for t in a.thresholds]
    if not thresholds or any(not np.isfinite(t) or not 0.0 <= t <= 1.0 for t in thresholds):
        raise ValueError(f"SIC 阈值必须是 0--1 内的有限小数，实际为 {thresholds}")
    for year in a.years:
        before_path = Path(a.before_template.format(year=year)).expanduser()
        after_path = Path(a.after_template.format(year=year)).expanduser()
        if not before_path.exists() or not after_path.exists():
            print(f"SKIP {year}: before={before_path.exists()} after={after_path.exists()}")
            continue
        print(f"Loading {year}: before={before_path.name}, after={after_path.name}")
        with xr.open_dataset(before_path) as db, xr.open_dataset(after_path) as da:
            tb, ta = find_time(db, a.time_name), find_time(da, a.time_name)
            vb = find_var(db, a.before_var, ("sic", "siconc", "z"))
            va = find_var(da, a.after_var, ("sic_pred", "sic", "siconc", "z"))
            db_dates, da_dates = dates(db, tb), dates(da, ta)
            before_lookup = {d: i for i, d in enumerate(db_dates)}
            after_lookup = {d: i for i, d in enumerate(da_dates)}
            xhr, yhr = xy_from_ds(da)
            land_hr = static_land_mask(a.grid_template, xhr, yhr)
            before_shape = tuple(int(v) for v in db[vb].shape[-2:])
            if len(before_shape) != 2:
                raise RuntimeError(f"25 km SIC 空间形状异常: {db[vb].shape}")
            try:
                xbefore, ybefore = xy_from_ds(db)
            except RuntimeError:
                xbefore, ybefore = None, None
            if a.no_route_mask:
                route_hr = np.ones((len(yhr), len(xhr)), dtype=bool)
                print("  route mask: disabled; using all non-land grid cells")
            else:
                route_hr = load_route_mask(
                    a.route_mask_file, a.route_mask_var, xhr, yhr
                )
            if xbefore is not None:
                # 原生 25 km 网格按其格点中心取最近的 6.25 km 静态掩膜值；
                # 不对 SIC 做双线性插值，也不人为创造细尺度梯度。
                land_before = map_boolean_mask(
                    land_hr, xhr, yhr, xbefore, ybefore, "25 km 陆地掩膜"
                )
                route_before = map_boolean_mask(
                    route_hr, xhr, yhr, xbefore, ybefore, "25 km 东北航道掩膜"
                )
            else:
                land_before = coarse_boolean_from_hr(land_hr, before_shape)
                route_before = coarse_boolean_from_hr(route_hr, before_shape)
            after_stride = max(1, int(a.stride_after))
            land_after = land_hr[::after_stride, ::after_stride]
            route_after = route_hr[::after_stride, ::after_stride]
            xafter, yafter = xhr[::after_stride], yhr[::after_stride]
            start_hr = nearest_xy(
                xafter, yafter, a.start_lon, a.start_lat, route_after & (~land_after)
            )
            end_hr = nearest_xy(
                xafter, yafter, a.end_lon, a.end_lat, route_after & (~land_after)
            )
            if xbefore is not None:
                start_b = nearest_xy(
                    xbefore, ybefore, a.start_lon, a.start_lat,
                    route_before & (~land_before),
                )
                end_b = nearest_xy(
                    xbefore, ybefore, a.end_lon, a.end_lat,
                    route_before & (~land_before),
                )
            else:
                # 无 25 km 坐标时，先按网格比例估计，再吸附到最近的航道海洋像元。
                ratio_y = route_hr.shape[0] / before_shape[0]
                ratio_x = route_hr.shape[1] / before_shape[1]
                raw_start = (round(start_hr[0] * after_stride / ratio_y), round(start_hr[1] * after_stride / ratio_x))
                raw_end = (round(end_hr[0] * after_stride / ratio_y), round(end_hr[1] * after_stride / ratio_x))
                valid_before = route_before & (~land_before)
                if not valid_before.any():
                    raise RuntimeError("无 x/y 的 25 km 兼容映射后，东北航道海洋有效区为空")
                yy, xx = np.indices(valid_before.shape)
                start_d2 = (yy - raw_start[0]) ** 2 + (xx - raw_start[1]) ** 2
                end_d2 = (yy - raw_end[0]) ** 2 + (xx - raw_end[1]) ** 2
                start_b = tuple(int(v) for v in np.unravel_index(np.argmin(np.where(valid_before, start_d2, np.inf)), valid_before.shape))
                end_b = tuple(int(v) for v in np.unravel_index(np.argmin(np.where(valid_before, end_d2, np.inf)), valid_before.shape))
            static_before, _ = astar(
                route_before & (~land_before), start_b, end_b, not a.no_diagonal
            )
            static_after, _ = astar(
                route_after & (~land_after), start_hr, end_hr, not a.no_diagonal
            )
            print(
                f"  NEP ocean pixels: 25km={int((route_before & ~land_before).sum())}, "
                f"6.25km/stride={int((route_after & ~land_after).sum())}; "
                f"static connected: 25km={static_before}, 6.25km={static_after}"
            )
            if xbefore is not None:
                snap_distances = [
                    snap_distance_km(xbefore, ybefore, start_b, a.start_lon, a.start_lat),
                    snap_distance_km(xbefore, ybefore, end_b, a.end_lon, a.end_lat),
                    snap_distance_km(xafter, yafter, start_hr, a.start_lon, a.start_lat),
                    snap_distance_km(xafter, yafter, end_hr, a.end_lon, a.end_lat),
                ]
                print(
                    f"  endpoint snap distance (start/end): "
                    f"25km={snap_distances[0]:.1f}/{snap_distances[1]:.1f} km, "
                    f"6.25km={snap_distances[2]:.1f}/{snap_distances[3]:.1f} km"
                )
                if max(snap_distances) > a.max_endpoint_snap_km:
                    raise RuntimeError(
                        f"端点吸附距离超过 {a.max_endpoint_snap_km:.1f} km；"
                        "请核对经纬度或东北航道掩膜，不能静默改变航线端点。"
                    )
            else:
                print(
                    f"  endpoints: 25km={start_b}->{end_b}, "
                    f"6.25km={start_hr}->{end_hr}"
                )
            if not static_before or not static_after:
                raise RuntimeError(
                    "东北航道静态海域中起终点不连通；停止评估，避免把掩膜断裂误判为海冰阻航。"
                )
            # 静态检查通过后，每个年份只从 NetCDF 完整读取一次，避免每日反复解压。
            before_all = norm_sic(np.asarray(db[vb].values))
            after_all = norm_sic(np.asarray(da[va].values))
            usable_dates = [d for d in db_dates
                            if (a.start_date is None or d >= a.start_date)
                            and (a.end_date is None or d <= a.end_date)]
            if a.max_days:
                usable_dates = usable_dates[:a.max_days]
            for n, date in enumerate(usable_dates, 1):
                if date not in after_lookup:
                    print(f"  {date}: missing after prediction, skip")
                    continue
                before = get_2d(before_all, before_lookup[date])
                after = get_2d(after_all, after_lookup[date])
                after = after[::after_stride, ::after_stride]
                before_results = evaluate_thresholds(
                    before, land_before, route_before, start_b, end_b,
                    thresholds, not a.no_diagonal,
                )
                after_results = evaluate_thresholds(
                    after, land_after, route_after, start_hr, end_hr,
                    thresholds, not a.no_diagonal,
                )
                for threshold in thresholds:
                    rows.append({"year": year, "date": date, "threshold": threshold,
                                 "before_passable": int(before_results[threshold]),
                                 "after_passable": int(after_results[threshold])})
                if n % 20 == 0: print(f"  {date}: {n}/{len(usable_dates)} days")
        save_rows(out, rows)
        print(f"  checkpoint saved after {year}: rows={len(rows)}")
    save_rows(out, rows)
    print(f"[OK] rows={len(rows)} saved={out}")


if __name__ == "__main__":
    main()
