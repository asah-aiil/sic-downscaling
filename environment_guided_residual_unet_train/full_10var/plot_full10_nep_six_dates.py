#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Bremen, bilinear and full-10 SIC for six selected 2024 dates.

The script is intended to be copied into the ``full_10var`` server directory
next to ``plot_full10_nep_error_2x3.py``.  It reuses that script's verified
cache/checkpoint loading and NEP frame logic, but writes only clean three-panel
SIC figures: no error layers, route, axes, labels, legend or colour bar.
Pixels below SIC=0.15 are rendered at 50% alpha, matching the current figure
style.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import torch
import torch.nn.functional as F

from plot_full10_nep_error_2x3 import (
    CACHE,
    CHECKPOINT,
    LAND,
    MASK,
    build_model,
    channel_indices,
    crop,
    crop_from_route,
    fixed_nep_window,
    land_mask,
    sample_id_for_date,
    safe_load,
    view_mesh,
)


DEFAULT_DATES = (
    "2024-08-19",
    "2024-08-20",
    "2024-08-24",
    "2024-08-25",
    "2024-08-27",
    "2024-09-29",
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dates", default=",".join(DEFAULT_DATES),
                   help="Comma-separated cache dates; default is the six 2024 diagnostic dates")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--mask-file", default=MASK)
    p.add_argument("--mask-var", default="mask_all_shipping_seas")
    p.add_argument("--frame-mask-vars", default="mask_barents,mask_kara,mask_laptev,mask_east_siberian,mask_chukchi,mask_bering")
    p.add_argument("--land-file", default=LAND)
    p.add_argument("--output-dir", default="local-path/fig/nep_six_dates")
    p.add_argument("--device", default="cuda")
    p.add_argument("--view-xmin", type=float, default=-1_950_000.0)
    p.add_argument("--view-xmax", type=float, default=2_800_000.0)
    p.add_argument("--view-ymin", type=float, default=-1_950_000.0)
    p.add_argument("--view-ymax", type=float, default=-350_000.0)
    p.add_argument("--sic-alpha-threshold", type=float, default=.15)
    return p.parse_args()


def draw_sic(ax, sic: np.ndarray, land: np.ndarray, x: np.ndarray, y: np.ndarray,
             limits: tuple[float, float, float, float], threshold: float) -> None:
    """Draw SIC only, with white land and semi-transparent low SIC."""
    vx, vy = view_mesh(x, y)
    ax.pcolormesh(
        vx, vy,
        np.ma.masked_where(~land, np.ones_like(land, dtype=float)),
        shading="auto", cmap=ListedColormap(["#ffffff"]), vmin=0, vmax=1,
        zorder=0,
    )
    finite = np.isfinite(sic) & ~land & (sic >= .01)
    sic_masked = np.ma.masked_where(~finite, sic)
    cmap = LinearSegmentedColormap.from_list(
        "sic_purple_blue_six_dates",
        ["#7ed5e5", "#1789d4", "#18368f", "#6e3d98"],
    )
    alpha = np.where(np.asarray(sic) < threshold, .50, 1.0)
    alpha = np.where(finite, alpha, 0.0)
    ax.pcolormesh(
        vx, vy, sic_masked, shading="auto", cmap=cmap, vmin=.01, vmax=1,
        alpha=alpha, zorder=1,
    )
    ax.set_xlim(limits[:2])
    ax.set_ylim(limits[2:])
    ax.set_aspect("equal")
    ax.set_axis_off()


def main() -> None:
    a = arguments()
    dates = tuple(d.strip() for d in a.dates.split(",") if d.strip())
    if not dates:
        raise ValueError("--dates must contain at least one date")
    if not 0 < a.sic_alpha_threshold < 1:
        raise ValueError("--sic-alpha-threshold must be between 0 and 1")

    cache = Path(a.cache)
    checkpoint = safe_load(Path(a.checkpoint))
    with (cache / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    X = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    Y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    channels = channel_indices(list(meta["channel_names"]))

    # The frame and land geometry are static, so read them once rather than
    # reopening either NetCDF file for every date.
    import xarray as xr
    with xr.open_dataset(a.mask_file, decode_times=False) as ds:
        route = np.asarray(ds[a.mask_var].transpose("y", "x").values > .5)
        x = np.asarray(ds.x.values)
        y = np.asarray(ds.y.values)
        frame_route = fixed_nep_window(ds, a.frame_mask_vars, route)
    land = land_mask(Path(a.land_file), x, y)
    if route.shape != Y.shape[-2:] or land.shape != route.shape:
        raise RuntimeError(f"Grid mismatch: route={route.shape}, land={land.shape}, target={Y.shape[-2:]}")
    bounds = crop_from_route(frame_route, x, y)
    limits = (a.view_xmin, a.view_xmax, a.view_ymin, a.view_ymax)

    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dates={len(dates)}; device={device}; output_dir={out_dir}")

    for number, date_text in enumerate(dates, 1):
        sid = sample_id_for_date(cache, date_text)
        inputs = torch.from_numpy(np.array(X[sid, channels], copy=True))[None].to(device)
        truth = np.asarray(Y[sid, 0], dtype=np.float32)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            prediction = model(inputs)
        pred = prediction.float().clamp(0, 1).cpu().numpy()[0, 0]
        bilinear = F.interpolate(
            inputs[:, 8:9].float(), size=truth.shape, mode="bilinear", align_corners=False
        ).cpu().numpy()[0, 0]

        truth, cx, cy = crop(truth, x, y, bounds)
        bilinear, _, _ = crop(bilinear, x, y, bounds)
        pred, _, _ = crop(pred, x, y, bounds)
        land_crop, _, _ = crop(land, x, y, bounds)

        fig = plt.figure(figsize=(15.0, 4.35), frameon=False)
        axes = [fig.add_axes([i / 3.0, 0.0, 1 / 3.0, 1.0]) for i in range(3)]
        for ax, sic in zip(axes, (truth, bilinear, pred)):
            draw_sic(ax, sic, land_crop, cx, cy, limits, a.sic_alpha_threshold)
        out = out_dir / f"nep_full10_{date_text}_truth_bilinear_downscaled.png"
        fig.savefig(out, dpi=600, facecolor="white", edgecolor="none",
                    bbox_inches=None, pad_inches=0)
        plt.close(fig)
        print(f"[{number}/{len(dates)}] saved={out}")


if __name__ == "__main__":
    main()
