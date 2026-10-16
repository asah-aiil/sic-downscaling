#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Bremen/model A* routes on the 25-km bilinear SIC background.

For each selected date this writes two large, borderless PNGs:
  * the A* route obtained from the Bremen reference SIC on the bilinear field;
  * the A* route obtained from the full-10 downscaled SIC on the same field.

No ``mask_all_shipping_seas`` is used.  Connectivity is evaluated inside the
fixed NEP projected viewport after removing only land.  Red cells are limited
to cells along the displayed route where the bilinear SIC exceeds the chosen
threshold; the whole background is never marked red.
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

from plot_full10_nep_path_buffer import (
    CACHE, CHECKPOINT, LAND, astar_path, build_model, land_mask,
    nearest_xy, read_index, safe_load, select_channels, view_mesh,
)


DEFAULT_DATES = ("2024-08-19", "2024-10-12")
SRC_BILINEAR = "bilinear"


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
    p.add_argument("--output-dir", default="local-path/fig/nep_routes_on_bilinear_nomask")
    p.add_argument("--view-xmin", type=float, default=-1_950_000.0)
    p.add_argument("--view-xmax", type=float, default=2_800_000.0)
    p.add_argument("--view-ymin", type=float, default=-1_950_000.0)
    p.add_argument("--view-ymax", type=float, default=-350_000.0)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--line-color", default="#174a6e")
    p.add_argument("--blocked-color", default="#e31a1c")
    p.add_argument("--ocean-color", default="#cfe8f5")
    p.add_argument("--land-color", default="#f6e3a2")
    p.add_argument("--blocked-alpha", type=float, default=.95)
    return p.parse_args()


def date_to_sid(rows: list[dict[str, str]], date_text: str) -> int:
    normalized = date_text.strip()
    for row in rows:
        if str(row.get("date", ""))[:10] == normalized:
            return int(row["sample_id"])
    raise KeyError(f"{normalized} is absent from sample_index.csv")


def source_crop_for_view(x: np.ndarray, y: np.ndarray,
                         limits: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    vx, vy = view_mesh(x, y)
    xmin, xmax, ymin, ymax = limits
    inside = np.isfinite(vx) & np.isfinite(vy)
    inside &= (vx >= xmin) & (vx <= xmax) & (vy >= ymin) & (vy <= ymax)
    if not inside.any():
        raise RuntimeError("no source grid intersects the requested viewport")
    iy, ix = np.where(inside)
    return float(np.min(x[ix])), float(np.max(x[ix])), float(np.min(y[iy])), float(np.max(y[iy]))


def crop(a: np.ndarray, x: np.ndarray, y: np.ndarray,
         bounds: tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = bounds
    ix = np.flatnonzero((x >= xmin) & (x <= xmax))
    iy = np.flatnonzero((y >= ymin) & (y <= ymax))
    return a[np.ix_(iy, ix)], x[ix], y[iy]


def path_xy(path: list[tuple[int, int]], x: np.ndarray, y: np.ndarray):
    if not path:
        return np.empty(0), np.empty(0)
    rr = np.asarray([v[0] for v in path], dtype=int)
    cc = np.asarray([v[1] for v in path], dtype=int)
    return view_mesh(x[cc], y[rr])


def draw_route_map(ax, bilinear: np.ndarray, land: np.ndarray,
                   path: list[tuple[int, int]], x: np.ndarray, y: np.ndarray,
                   limits: tuple[float, float, float, float], threshold: float,
                   line_color: str, blocked_color: str, ocean_color: str,
                   land_color: str, blocked_alpha: float) -> int:
    vx, vy = view_mesh(x, y)
    # Pale ocean and land give the SIC field a stable geographic context.
    mesh_style = dict(shading="auto", edgecolors="none", antialiased=False,
                      linewidth=0, rasterized=True)
    ax.pcolormesh(vx, vy, np.ones_like(land, dtype=float),
                  cmap=ListedColormap([ocean_color]), vmin=0, vmax=1,
                  zorder=0, **mesh_style)
    ax.pcolormesh(vx, vy, np.ma.masked_where(~land, np.ones_like(land, dtype=float)),
                  cmap=ListedColormap([land_color]), vmin=0, vmax=1,
                  zorder=1, **mesh_style)
    ice = np.ma.masked_where(land | ~np.isfinite(bilinear) | (bilinear < .01), bilinear)
    sic_cmap = LinearSegmentedColormap.from_list(
        "sic_purple_blue_routes",
        ["#7ed5e5", "#1789d4", "#18368f", "#6e3d98"],
    )
    ax.pcolormesh(vx, vy, ice, cmap=sic_cmap,
                  vmin=.01, vmax=1, zorder=2, **mesh_style)

    blocked = 0
    if path:
        rr = np.asarray([v[0] for v in path], dtype=int)
        cc = np.asarray([v[1] for v in path], dtype=int)
        blocked_mask = np.asarray(bilinear[rr, cc] > threshold)
        blocked = int(blocked_mask.sum())
        if blocked:
            # Draw red at route-cell centres, not as a broad corridor.
            px, py = view_mesh(x[cc[blocked_mask]], y[rr[blocked_mask]])
            ax.scatter(px, py, s=8.0, c=blocked_color, alpha=blocked_alpha,
                       marker="s", linewidths=0, zorder=5)
        px, py = path_xy(path, x, y)
        # Thin route line, intentionally drawn after red cells but still narrow.
        ax.plot(px, py, color=line_color, linewidth=.72, alpha=.98,
                solid_capstyle="round", solid_joinstyle="round", zorder=6)

    ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
    ax.set_aspect("equal")
    ax.set_axis_off()
    return blocked


def main() -> None:
    a = arguments()
    if not 0 <= a.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    dates = [v.strip() for v in a.dates.split(",") if v.strip()]
    if not dates:
        raise ValueError("no dates selected")

    cache = Path(a.cache)
    checkpoint = safe_load(Path(a.checkpoint))
    with (cache / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    X = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    Y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    index_rows = read_index(cache)
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
    source_bounds = source_crop_for_view(x, y, limits)

    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device)
    out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print("shipping-sea mask = DISABLED")
    print(f"dates={len(dates)}; device={device}; endpoints={start}->{goal}")

    for n, date_text in enumerate(dates, 1):
        sid = date_to_sid(index_rows, date_text)
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

        p_truth = astar_path(allowed & np.isfinite(truth) & (truth <= a.threshold),
                             start, goal, clearance_weight=0.0)
        p_model = astar_path(allowed & np.isfinite(pred) & (pred <= a.threshold),
                             start, goal, clearance_weight=0.0)
        if p_truth is None or p_model is None:
            raise RuntimeError(f"{date_text}: truth/model A* path missing ({p_truth is not None}/{p_model is not None})")

        truth_b, cx, cy = crop(truth, x, y, source_bounds)
        bilinear_b, _, _ = crop(bilinear, x, y, source_bounds)
        pred_b, _, _ = crop(pred, x, y, source_bounds)
        land_b, _, _ = crop(land, x, y, source_bounds)
        # Crop paths into the same array coordinates.
        x0 = int(np.searchsorted(x, cx[0]))
        y0 = int(np.searchsorted(y, cy[0]))
        p_truth_b = [(r - y0, c - x0) for r, c in p_truth
                     if 0 <= r - y0 < len(cy) and 0 <= c - x0 < len(cx)]
        p_model_b = [(r - y0, c - x0) for r, c in p_model
                     if 0 <= r - y0 < len(cy) and 0 <= c - x0 < len(cx)]

        for route_name, route_path, route_field in (
            ("bremen_route", p_truth_b, truth_b),
            ("downscaled_route", p_model_b, pred_b),
        ):
            # The background is always the 25-km bilinear field.  route_field
            # is retained in the loop for explicit naming/documentation.
            del route_field
            fig = plt.figure(figsize=(10.0, 7.0), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1])
            blocked = draw_route_map(
                ax, bilinear_b, land_b, route_path, cx, cy, limits, a.threshold,
                a.line_color, a.blocked_color, a.ocean_color, a.land_color,
                a.blocked_alpha,
            )
            out = out_dir / f"nep_{date_text}_{route_name}_on_bilinear.png"
            fig.savefig(out, dpi=a.dpi, facecolor="white", edgecolor="none",
                        bbox_inches=None, pad_inches=0)
            plt.close(fig)
            print(f"[{n}/{len(dates)}] {route_name}: path_cells={len(route_path)} "
                  f"bilinear_blocked_cells={blocked} saved={out}")


if __name__ == "__main__":
    main()
