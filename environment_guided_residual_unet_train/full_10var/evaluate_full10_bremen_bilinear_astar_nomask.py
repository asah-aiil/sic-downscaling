#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily A* comparison without the external shipping-sea mask.

This is a diagnostic experiment.  It compares Bremen target, 25-km bilinear
SIC and the environment-guided full-10 prediction using only the static land
mask plus the fixed NEP projected viewport.  The historical
``mask_all_shipping_seas`` is not read and does not constrain connectivity.
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

from plot_full10_nep_path_buffer import (
    CACHE, CHECKPOINT, LAND, build_model, land_mask, nearest_xy,
    read_index, safe_load, select_channels, astar_path, view_mesh,
)


DEFAULT_DATES = (
    "2024-08-19", "2024-08-20", "2024-08-24",
    "2024-08-25", "2024-08-27", "2024-09-29",
)
FIELDS = [
    "year", "date", "threshold", "truth_passable", "bilinear_passable",
    "downscaled_passable", "truth_path_cells", "bilinear_path_cells",
    "downscaled_path_cells", "change_vs_bilinear", "domain",
]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dates", default=None,
                   help="Comma-separated dates; if omitted, use --year or the six diagnostic dates")
    p.add_argument("--year", type=int, default=None,
                   help="Evaluate every date for this year from sample_index.csv")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--land-file", default=LAND)
    p.add_argument("--threshold", type=float, default=.15)
    p.add_argument("--start-lon", type=float, default=-174.0)
    p.add_argument("--start-lat", type=float, default=61.5)
    p.add_argument("--end-lon", type=float, default=20.0)
    p.add_argument("--end-lat", type=float, default=72.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-days", type=int, default=None)
    p.add_argument("--output-csv", default="/home/ubuntu/sic_dsc/bremen_full10_bilinear_astar_nomask_2024.csv")
    p.add_argument("--view-xmin", type=float, default=-1_950_000.0)
    p.add_argument("--view-xmax", type=float, default=2_800_000.0)
    p.add_argument("--view-ymin", type=float, default=-1_950_000.0)
    p.add_argument("--view-ymax", type=float, default=-350_000.0)
    return p.parse_args()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def path_cells(path) -> int:
    return -1 if path is None else len(path)


def main() -> None:
    a = arguments()
    if not 0 <= a.threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    cache = Path(a.cache)
    with (cache / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    rows_index = read_index(cache)
    if a.dates:
        dates = [v.strip() for v in a.dates.split(",") if v.strip()]
    elif a.year is not None:
        dates = [str(r.get("date", ""))[:10] for r in rows_index
                 if str(r.get("date", ""))[:4] == str(a.year)]
    else:
        dates = list(DEFAULT_DATES)
    if a.max_days is not None:
        dates = dates[:a.max_days]
    if not dates:
        raise ValueError("no dates selected")

    checkpoint = safe_load(Path(a.checkpoint))
    X = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    Y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    channels = select_channels(list(meta["channel_names"]))

    # Coordinates and land are static.  The viewport is a geographic display
    # boundary only; it is not a shipping-sea or route mask.
    with xr.open_dataset(Path(a.land_file), decode_times=False) as ds:
        x = np.asarray(ds.x.values)
        y = np.asarray(ds.y.values)
    land = land_mask(Path(a.land_file), x, y)
    limits = (a.view_xmin, a.view_xmax, a.view_ymin, a.view_ymax)
    vx, vy = view_mesh(x, y)
    viewport = np.isfinite(vx) & np.isfinite(vy)
    viewport &= (vx >= a.view_xmin) & (vx <= a.view_xmax)
    viewport &= (vy >= a.view_ymin) & (vy <= a.view_ymax)
    allowed = viewport & ~land
    if not allowed.any():
        raise RuntimeError("no ocean cells remain inside the fixed NEP viewport")
    start = nearest_xy(x, y, a.start_lon, a.start_lat, allowed)
    goal = nearest_xy(x, y, a.end_lon, a.end_lat, allowed)
    print(f"shipping-sea mask = DISABLED; domain = land-free fixed NEP viewport")
    print(f"viewport x[{a.view_xmin:.0f},{a.view_xmax:.0f}] y[{a.view_ymin:.0f},{a.view_ymax:.0f}]")
    print(f"allowed ocean cells={int(allowed.sum())}; endpoints={start}->{goal}")

    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(checkpoint, device)
    result: list[dict[str, object]] = []
    for number, date_text in enumerate(dates, 1):
        sid = next((int(r["sample_id"]) for r in rows_index if str(r.get("date", ""))[:10] == date_text), None)
        if sid is None:
            raise KeyError(f"{date_text} is absent from {cache / 'sample_index.csv'}")
        inputs = torch.from_numpy(np.array(X[sid, channels], copy=True))[None].to(device)
        truth = np.asarray(Y[sid, 0], dtype=np.float32)
        with torch.inference_mode(), torch.amp.autocast(
            device_type=device.type, enabled=device.type == "cuda"
        ):
            prediction = model(inputs)
        prediction = prediction.float().clamp(0, 1).cpu().numpy()[0, 0]
        bilinear = F.interpolate(
            inputs[:, 8:9].float(), size=truth.shape, mode="bilinear", align_corners=False
        ).cpu().numpy()[0, 0]

        def daily_path(field: np.ndarray):
            passable = allowed & np.isfinite(field) & (field <= a.threshold)
            return astar_path(passable, start, goal, clearance_weight=0.0)

        p_truth = daily_path(truth)
        p_bilinear = daily_path(bilinear)
        p_model = daily_path(prediction)
        truth_ok, bilinear_ok, model_ok = map(lambda p: int(p is not None), (p_truth, p_bilinear, p_model))
        change = "same" if bilinear_ok == model_ok else (
            "opened_by_downscaling" if model_ok else "closed_by_downscaling"
        )
        result.append({
            "year": int(date_text[:4]), "date": date_text, "threshold": a.threshold,
            "truth_passable": truth_ok, "bilinear_passable": bilinear_ok,
            "downscaled_passable": model_ok,
            "truth_path_cells": path_cells(p_truth),
            "bilinear_path_cells": path_cells(p_bilinear),
            "downscaled_path_cells": path_cells(p_model),
            "change_vs_bilinear": change,
            "domain": "land_free_fixed_nep_viewport",
        })
        write_rows(Path(a.output_csv), result)
        print(f"[{number}/{len(dates)}] {date_text}: truth={truth_ok} bilinear={bilinear_ok} downscaled={model_ok} ({change})")

    out = Path(a.output_csv)
    disagreement = out.with_name(out.stem + "_disagreement.csv")
    write_rows(disagreement, [r for r in result if r["change_vs_bilinear"] != "same"])
    print(f"[OK] rows={len(result)} saved={out}")
    print(f"[OK] disagreement rows={len([r for r in result if r['change_vs_bilinear'] != 'same'])} saved={disagreement}")


if __name__ == "__main__":
    main()
