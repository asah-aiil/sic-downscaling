#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare 25-km bilinear Bremen SIC and full10 downscaling with daily NEP A*.

This is an *independent Bremen validation* diagnostic, not a CMCC projection:
the low-resolution SIC channel is bilinearly interpolated to 6.25 km and is
compared with the environment-guided full-10 model prediction against the
same route mask, endpoints, and SIC threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

# Reuse the exact checkpoint, cache-channel, land-mask, endpoint-snap and A*
# implementation used by the current spatial-figure workflow.
from plot_full10_nep_path_buffer import (
    CACHE, CHECKPOINT, LAND, MASK, build_model, land_mask, nearest_xy,
    read_index, safe_load, select_channels, astar_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--mask-file", default=MASK)
    p.add_argument("--mask-var", default="mask_all_shipping_seas")
    p.add_argument("--land-file", default=LAND)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--threshold", type=float, default=.15)
    p.add_argument("--start-lon", type=float, default=-174.0)
    p.add_argument("--start-lat", type=float, default=61.5)
    p.add_argument("--end-lon", type=float, default=20.0)
    p.add_argument("--end-lat", type=float, default=72.0)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Keep 1 for reliable full-resolution inference memory use")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-days", type=int, default=None, help="Optional smoke-test limit")
    p.add_argument("--output-csv", default="local-path/bremen_full10_bilinear_astar_2025.csv")
    p.add_argument("--disagreement-csv", default=None,
                   help="Default: a companion CSV containing only disagreement dates")
    return p.parse_args()


def path_length(path: list[tuple[int, int]] | None) -> int:
    return -1 if path is None else len(path)


def main() -> None:
    a = parse_args()
    if not 0.0 <= a.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    cache = Path(a.cache)
    meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
    X = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    Y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    rows = [r for r in read_index(cache) if str(r.get("date", ""))[:4] == str(a.year)]
    if a.max_days is not None:
        rows = rows[:a.max_days]
    if not rows:
        raise RuntimeError(f"No cache dates for {a.year}")
    channels = select_channels(list(meta["channel_names"]))

    with xr.open_dataset(a.mask_file, decode_times=False) as ds:
        x = np.asarray(ds.x.values)
        y = np.asarray(ds.y.values)
        route = np.asarray(ds[a.mask_var].transpose("y", "x").values > .5)
    land = land_mask(Path(a.land_file), x, y)
    if route.shape != Y.shape[-2:] or land.shape != route.shape:
        raise RuntimeError(f"Grid mismatch: route={route.shape}, land={land.shape}, Y={Y.shape[-2:]}")
    allowed = route & ~land
    start = nearest_xy(x, y, a.start_lon, a.start_lat, allowed)
    goal = nearest_xy(x, y, a.end_lon, a.end_lat, allowed)
    print(f"dates={len(rows)}; route-ocean cells={int(allowed.sum())}; endpoints={start}->{goal}")

    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(safe_load(Path(a.checkpoint)), device)
    out = Path(a.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    disagreement = Path(a.disagreement_csv) if a.disagreement_csv else out.with_name(out.stem + "_disagreement.csv")
    fields = ["year", "date", "threshold", "bilinear_passable", "downscaled_passable",
              "bilinear_path_cells", "downscaled_path_cells", "change"]
    result: list[dict[str, object]] = []

    for number, row in enumerate(rows, 1):
        sid = int(row["sample_id"])
        inputs = torch.from_numpy(np.array(X[sid, channels], copy=True))[None].to(device)
        truth = np.asarray(Y[sid, 0], dtype=np.float32)
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(inputs)
        prediction = prediction.float().clamp(0, 1).cpu().numpy()[0, 0]
        bilinear = F.interpolate(inputs[:, 8:9].float(), size=truth.shape, mode="bilinear", align_corners=False)
        bilinear = bilinear.cpu().numpy()[0, 0]

        # No clearance preference here: the test asks only whether any valid
        # A* connection exists, exactly as in the historical daily-window CSV.
        p_bilinear = astar_path(allowed & np.isfinite(bilinear) & (bilinear <= a.threshold), start, goal,
                                clearance_weight=0.0)
        p_model = astar_path(allowed & np.isfinite(prediction) & (prediction <= a.threshold), start, goal,
                             clearance_weight=0.0)
        b_ok, m_ok = int(p_bilinear is not None), int(p_model is not None)
        change = "same" if b_ok == m_ok else ("opened_by_downscaling" if m_ok else "closed_by_downscaling")
        result.append({"year": a.year, "date": str(row["date"])[:10], "threshold": a.threshold,
                       "bilinear_passable": b_ok, "downscaled_passable": m_ok,
                       "bilinear_path_cells": path_length(p_bilinear), "downscaled_path_cells": path_length(p_model),
                       "change": change})
        if number % 10 == 0 or number == len(rows):
            with out.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(result)
            print(f"{number}/{len(rows)} dates; latest={result[-1]['date']} {change}")

    only_changed = [r for r in result if r["change"] != "same"]
    with disagreement.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(only_changed)
    print(f"[OK] all days: {out}")
    print(f"[OK] disagreement days={len(only_changed)}: {disagreement}")


if __name__ == "__main__":
    main()
