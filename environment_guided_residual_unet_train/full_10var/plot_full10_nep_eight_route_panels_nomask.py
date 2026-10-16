#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eight clean route-case maps for the two 2024 recovery dates.

For each date the script writes four borderless PNGs:
  1. Bremen reference SIC with its own A* route;
  2. full-10 downscaled SIC with its own A* route;
  3. 25-km bilinear SIC with the Bremen route and bilinear-blocked cells red;
  4. 25-km bilinear SIC with the downscaled route and blocked cells red.

The A* domain is the fixed NEP projected viewport minus land only.  The
external ``mask_all_shipping_seas`` is intentionally not used.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from scipy.ndimage import distance_transform_edt

from plot_full10_nep_path_buffer import (
    CACHE, CHECKPOINT, LAND, astar_path, build_model, land_mask,
    nearest_xy, read_index, safe_load, select_channels, view_mesh, TO_VIEW,
)


DEFAULT_DATES = ("2024-08-19", "2024-10-12")


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dates", default=",".join(DEFAULT_DATES))
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--land-file", default=LAND)
    p.add_argument("--threshold", type=float, default=.15)
    p.add_argument("--start-lon", type=float, default=-174.0)
    p.add_argument("--start-lat", type=float, default=61.5)
    p.add_argument("--end-lon", type=float, default=20.0)
    p.add_argument("--end-lat", type=float, default=72.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="/home/ubuntu/sic_dsc/fig/nep_eight_route_panels_nomask")
    p.add_argument("--view-xmin", type=float, default=-1_950_000.0)
    p.add_argument("--view-xmax", type=float, default=2_800_000.0)
    p.add_argument("--view-ymin", type=float, default=-1_950_000.0)
    p.add_argument("--view-ymax", type=float, default=-350_000.0)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--line-color", default="#174a6e")
    p.add_argument("--blocked-color", default="#d7191c")
    p.add_argument("--ocean-color", default="#d9edf7")
    p.add_argument("--land-color", default="#fff8dc")
    p.add_argument("--blocked-alpha", type=float, default=.95)
    p.add_argument("--clearance-scale-km", type=float, default=25.0,
                   help="e-folding distance for the soft clearance penalty")
    p.add_argument("--clearance-weight", type=float, default=4.0,
                   help="penalty for cells adjacent to ice/land/boundary")
    return p.parse_args()


def sid_for_date(rows: list[dict[str, str]], date_text: str) -> int:
    for row in rows:
        if str(row.get("date", ""))[:10] == date_text:
            return int(row["sample_id"])
    raise KeyError(f"{date_text} is absent from sample_index.csv")


def source_crop_for_view(x: np.ndarray, y: np.ndarray,
                         limits: tuple[float, float, float, float]):
    vx, vy = view_mesh(x, y)
    xmin, xmax, ymin, ymax = limits
    inside = np.isfinite(vx) & np.isfinite(vy)
    inside &= (vx >= xmin) & (vx <= xmax) & (vy >= ymin) & (vy <= ymax)
    if not inside.any():
        raise RuntimeError("no source grid intersects the viewport")
    iy, ix = np.where(inside)
    return float(x[ix].min()), float(x[ix].max()), float(y[iy].min()), float(y[iy].max())


def crop(a: np.ndarray, x: np.ndarray, y: np.ndarray,
         bounds: tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = bounds
    ix = np.flatnonzero((x >= xmin) & (x <= xmax))
    iy = np.flatnonzero((y >= ymin) & (y <= ymax))
    return a[np.ix_(iy, ix)], x[ix], y[iy], ix, iy


def crop_path(path: list[tuple[int, int]], ix: np.ndarray, iy: np.ndarray):
    if not path:
        return []
    x0, y0 = int(ix[0]), int(iy[0])
    xmax, ymax = x0 + len(ix), y0 + len(iy)
    return [(r - y0, c - x0) for r, c in path
            if y0 <= r < ymax and x0 <= c < xmax]


def draw_map(ax, field: np.ndarray, land: np.ndarray,
             route_path: list[tuple[int, int]], blocked_field: np.ndarray | None,
             x: np.ndarray, y: np.ndarray,
             limits: tuple[float, float, float, float], threshold: float,
             line_color: str, blocked_color: str, ocean_color: str,
             land_color: str, blocked_alpha: float) -> int:
    vx, vy = view_mesh(x, y)
    mesh_style = dict(shading="auto", edgecolors="none", antialiased=False,
                      linewidth=0, rasterized=True)
    ax.pcolormesh(vx, vy, np.ones_like(land, dtype=float),
                  cmap=ListedColormap([ocean_color]), vmin=0, vmax=1,
                  zorder=0, **mesh_style)
    ax.pcolormesh(vx, vy, np.ma.masked_where(~land, np.ones_like(land, dtype=float)),
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1,
                  zorder=1, **mesh_style)

    valid = ~land & np.isfinite(field) & (field >= .01)
    sic = np.ma.masked_where(~valid, field)
    cmap = LinearSegmentedColormap.from_list(
        "sic_purple_blue_eight_routes",
        ["#7ed5e5", "#1789d4", "#18368f", "#6e3d98"],
    )
    alpha = np.where(np.asarray(field) < threshold, .50, 1.0)
    alpha = np.where(valid, alpha, 0.0)
    ax.pcolormesh(vx, vy, sic, cmap=cmap, vmin=.01, vmax=1,
                  alpha=alpha, zorder=2, **mesh_style)

    blocked = 0
    if blocked_field is not None and route_path:
        rr = np.asarray([v[0] for v in route_path], dtype=int)
        cc = np.asarray([v[1] for v in route_path], dtype=int)
        hit = np.isfinite(blocked_field[rr, cc]) & (blocked_field[rr, cc] > threshold)
        blocked = int(hit.sum())
        red = np.full_like(blocked_field, np.nan, dtype=np.float32)
        red[rr[hit], cc[hit]] = 1.0
        ax.pcolormesh(vx, vy, np.ma.masked_invalid(red),
                      cmap=ListedColormap([blocked_color]), vmin=0, vmax=1,
                      # Draw blockage cells above the route line so that a
                      # blocked segment remains visible in the overlay.
                      alpha=blocked_alpha, zorder=8, **mesh_style)

    if route_path:
        rr = np.asarray([v[0] for v in route_path], dtype=int)
        cc = np.asarray([v[1] for v in route_path], dtype=int)
        # Point-wise projection is required here.  ``view_mesh`` creates a
        # 2-D mesh and would turn an N-cell path into an N-by-N grid.
        px, py = TO_VIEW.transform(x[cc], y[rr])
        ax.plot(px, py, color=line_color, linewidth=.72, alpha=.98,
                solid_capstyle="round", solid_joinstyle="round", zorder=6)
    ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
    ax.set_aspect("equal"); ax.set_axis_off()
    return blocked


def save_panel(out: Path, field: np.ndarray, land: np.ndarray,
               path: list[tuple[int, int]], blocked_field: np.ndarray | None,
               x: np.ndarray, y: np.ndarray, limits, a) -> int:
    fig = plt.figure(figsize=(10.0, 7.0), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    blocked = draw_map(ax, field, land, path, blocked_field, x, y, limits,
                       a.threshold, a.line_color, a.blocked_color,
                       a.ocean_color, a.land_color, a.blocked_alpha)
    fig.savefig(out, dpi=a.dpi, facecolor="white", edgecolor="none",
                bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return blocked


def main() -> None:
    a = arguments()
    if not 0 <= a.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    dates = [v.strip() for v in a.dates.split(",") if v.strip()]
    cache = Path(a.cache)
    checkpoint = safe_load(Path(a.checkpoint))
    with (cache / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    X = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    Y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    rows = read_index(cache)
    channels = select_channels(list(meta["channel_names"]))

    with xr.open_dataset(Path(a.land_file), decode_times=False) as ds:
        x = np.asarray(ds.x.values)
        y = np.asarray(ds.y.values)
    land = land_mask(Path(a.land_file), x, y)
    if Y.shape[-2:] != land.shape:
        raise RuntimeError(f"target shape {Y.shape[-2:]} != land shape {land.shape}")

    limits = (a.view_xmin, a.view_xmax, a.view_ymin, a.view_ymax)
    vx, vy = view_mesh(x, y)
    viewport = np.isfinite(vx) & np.isfinite(vy)
    viewport &= (vx >= a.view_xmin) & (vx <= a.view_xmax)
    viewport &= (vy >= a.view_ymin) & (vy <= a.view_ymax)
    allowed = viewport & ~land
    start = nearest_xy(x, y, a.start_lon, a.start_lat, allowed)
    goal = nearest_xy(x, y, a.end_lon, a.end_lat, allowed)
    dx_m = float(np.median(np.abs(np.diff(x))))
    dy_m = float(np.median(np.abs(np.diff(y))))
    bounds = source_crop_for_view(x, y, limits)
    _, _, _, ix, iy = crop(land, x, y, bounds)

    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"shipping-sea mask=DISABLED; dates={len(dates)}; device={device}")
    print(f"path mode=clearance-aware; scale={a.clearance_scale_km:g} km; "
          f"weight={a.clearance_weight:g}; grid={dx_m:.1f}x{dy_m:.1f} m")
    print(f"endpoints={start}->{goal}; viewport ocean cells={int(allowed.sum())}")

    def solve_path(field: np.ndarray):
        # SIC <= threshold remains the hard passability rule.  Clearance is
        # only a soft cost, so A* avoids ice/land edges when an open-water
        # alternative exists but can still cross a narrow valid strait.
        passable = allowed & np.isfinite(field) & (field <= a.threshold)
        clearance = distance_transform_edt(passable, sampling=(dy_m, dx_m))
        return astar_path(
            passable, start, goal,
            clearance_m=clearance,
            clearance_scale_m=a.clearance_scale_km * 1000.0,
            clearance_weight=a.clearance_weight,
        )

    for n, date_text in enumerate(dates, 1):
        sid = sid_for_date(rows, date_text)
        inputs = torch.from_numpy(np.array(X[sid, channels], copy=True))[None].to(device)
        truth = np.asarray(Y[sid, 0], dtype=np.float32)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            pred = model(inputs)
        pred = pred.float().clamp(0, 1).cpu().numpy()[0, 0]
        bilinear = F.interpolate(
            inputs[:, 8:9].float(), size=truth.shape, mode="bilinear", align_corners=False
        ).cpu().numpy()[0, 0]

        p_truth = solve_path(truth)
        p_model = solve_path(pred)
        if p_truth is None or p_model is None:
            raise RuntimeError(f"{date_text}: missing truth/model path")

        truth_b, cx, cy, _, _ = crop(truth, x, y, bounds)
        bilinear_b, _, _, _, _ = crop(bilinear, x, y, bounds)
        pred_b, _, _, _, _ = crop(pred, x, y, bounds)
        land_b, _, _, _, _ = crop(land, x, y, bounds)
        p_truth_b = crop_path(p_truth, ix, iy)
        p_model_b = crop_path(p_model, ix, iy)

        panels = (
            ("bremen_truth_with_bremen_route", truth_b, p_truth_b, None),
            ("downscaled_with_downscaled_route", pred_b, p_model_b, None),
            ("bilinear_with_bremen_route_blocked", bilinear_b, p_truth_b, bilinear_b),
            ("bilinear_with_downscaled_route_blocked", bilinear_b, p_model_b, bilinear_b),
        )
        for suffix, field, route_path, blocked_field in panels:
            out = out_dir / f"nep_{date_text}_{suffix}.png"
            blocked = save_panel(out, field, land_b, route_path, blocked_field,
                                 cx, cy, limits, a)
            print(f"[{n}/{len(dates)}] {suffix}: path={len(route_path)} "
                  f"bilinear_blocked={blocked} saved={out}")


if __name__ == "__main__":
    main()
