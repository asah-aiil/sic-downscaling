#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export NEP SIC/error layers clipped to a Bremen A* route buffer.

The route is found on the Bremen reference SIC for one independent-cache day.
The same route buffer is then applied to Bremen, 25-km bilinear SIC, and the
environment-guided 10-variable prediction.  It is therefore a reproducible
visualisation corridor, not a second evaluation mask.

Run this from the full_10var directory.  It only performs inference and image
export; it never modifies a model, cache, or NetCDF input.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F
import xarray as xr

from config import Config
from model import UNetWithResidual

CHECKPOINT = "local-path/code/checkpoints/20260910_015743_envgated_t2m_u10_v10_msl_pr_ssr_tcc_tos_sic_mask_year_fixed_sic_residual_2014_2023_T1_d_rmse_fixed/best_model.pth"
CACHE = "local-path/data/cache_temporal_T3_float16_2024_2025_rebuilt"
MASK = "local-path/data/dataset/mask/eurasia_shipping_masks_6p25km_with_arctic.nc"
LAND = "local-path/data/dataset/target/Bremen_2025.nc"
OUT = "local-path/fig"
LAND_FILL = "#FFFDF0"  # very pale yellow, distinct from unfilled ocean
LAND_SHADOW = "#CBD5E1"

ALIASES = {
    "t2m": ["t2m_d"], "u10": ["u10_d"], "v10": ["v10_d"], "msl": ["msl_d"],
    "pr": ["pr_d", "tp_d"], "ssr": ["ssr_d"], "tcc": ["tcc_d"],
    "tos": ["tos_d", "sst_d"], "sic": ["sic_d", "siconc_d"],
    "mask": ["mask_d", "sic_mask_d"],
}
VARS = ["t2m", "u10", "v10", "msl", "pr", "ssr", "tcc", "tos", "sic", "mask"]
SRC_CRS = CRS.from_epsg(3413)
VIEW_CRS = CRS.from_proj4(
    "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=112.5 +datum=WGS84 +units=m +no_defs"
)
TO_VIEW = Transformer.from_crs(SRC_CRS, VIEW_CRS, always_xy=True)
TO_GEO = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2025-07-10", help="Requested Bremen/cache day")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--mask-file", default=MASK)
    p.add_argument("--mask-var", default="mask_all_shipping_seas")
    p.add_argument("--route-domain", choices=("shipping-mask", "named-nep-seas", "named-nep-seas-capped"), default="shipping-mask",
                   help="A* domain: historical mask; named sea union; or named union with an automatically capped northern envelope for an illustrative route")
    p.add_argument("--named-sea-vars", default="mask_barents,mask_kara,mask_laptev,mask_east_siberian,mask_chukchi,mask_bering",
                   help="Named sea masks used when --route-domain named-nep-seas")
    p.add_argument("--named-sea-expand-km", type=float, default=0.0,
                   help="Expand each named sea mask by this physical distance before A* (0 keeps masks unchanged)")
    p.add_argument("--cap-min-lat", type=float, default=62.0,
                   help="Southern geographic limit of the automatic named-sea cap (degrees N)")
    p.add_argument("--land-file", default=LAND)
    p.add_argument("--output-dir", default=OUT)
    p.add_argument("--threshold", type=float, default=.15)
    p.add_argument("--buffer-km", type=float, default=300.0)
    p.add_argument("--block-highlight-pixels", type=float, default=3.0,
                   help="Half-width of red blockage highlight in grid cells (6.25-km grid; default 3)")
    p.add_argument("--clearance-scale-km", type=float, default=25.0,
                   help="Distance over which proximity to ice/land is penalised (default: 25 km)")
    p.add_argument("--clearance-weight", type=float, default=4.0,
                   help="Soft safety preference: 0 restores a purely shortest A* path")
    p.add_argument("--start-lon", type=float, default=-174.0)
    p.add_argument("--start-lat", type=float, default=61.5)
    p.add_argument("--end-lon", type=float, default=20.0)
    p.add_argument("--end-lat", type=float, default=72.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-coastline", action="store_true",
                   help="Do not overlay the static Bremen land/sea boundary")
    p.add_argument("--show-ice-edge", action="store_true",
                   help="Overlay the SIC=0.15 contour (off by default for clean route-case panels)")
    p.add_argument("--no-route-line", action="store_true",
                   help="Hide the A* route tube while retaining buffer/coastline/red blockage layers")
    p.add_argument("--no-route-domain-clip", action="store_true",
                   help="Keep the full path buffer outside the shipping/named-sea mask (land is still removed)")
    p.add_argument("--no-date-fallback", action="store_true",
                   help="Fail if the requested Bremen day is not passable instead of finding another passable day in that year")
    p.add_argument("--route-path-csv", default=None,
                   help="Reuse an earlier A* route CSV for the requested date instead of recalculating the route")
    p.add_argument("--stats-only", action="store_true",
                   help="Print colour-scale and pixel-distribution diagnostics without saving figures")
    p.add_argument("--f-only", action="store_true",
                   help="Regenerate only panel f, leaving the other map panels untouched")
    p.add_argument("--min-sic-change", type=float, default=0.0,
                   help="Only colour/statistically evaluate pixels where |downscaled - bilinear SIC| reaches this fraction")
    p.add_argument("--view-xmin", type=float, default=-1_950_000.0)
    p.add_argument("--view-xmax", type=float, default=2_800_000.0)
    p.add_argument("--view-ymin", type=float, default=-1_950_000.0)
    p.add_argument("--view-ymax", type=float, default=-350_000.0)
    return p.parse_args()


def safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Cannot find model state dict in checkpoint")


def build_model(checkpoint: Any, device: torch.device) -> UNetWithResidual:
    cfg = dict(checkpoint.get("config", {}))
    if [str(v).lower() for v in cfg.get("INPUT_VARS", [])] != VARS[:-1]:
        raise RuntimeError("Checkpoint is not the expected environment-guided 10-variable model")
    keys = ("MODEL_OUTPUT_MODE", "SIC_CHANNEL_INDEX", "MASK_CHANNEL_INDEX", "ENV_CHANNEL_INDICES",
            "SIC_RESIDUAL_SCALE", "SIC_BASELINE_INTERPOLATION", "ENV_GATE_INITIAL_BIAS",
            "ENV_GATE_SCALE", "SIC_RESIDUAL_ZERO_INIT")
    old = {key: getattr(Config, key, None) for key in keys}
    existed = {key: hasattr(Config, key) for key in keys}
    try:
        for key in keys:
            if key in cfg:
                setattr(Config, key, cfg[key])
        model = UNetWithResidual(in_channels=10, out_channels=1)
    finally:
        for key in keys:
            if existed[key]:
                setattr(Config, key, old[key])
            elif hasattr(Config, key):
                delattr(Config, key)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return model.to(device).eval()


def select_channels(names: list[str]) -> list[int]:
    out = []
    for variable in VARS:
        name = next((v for v in ALIASES[variable] if v in names), None)
        if name is None:
            raise RuntimeError(f"Missing channel {variable}; cache contains {names}")
        out.append(names.index(name))
    return out


def read_index(cache: Path) -> list[dict[str, str]]:
    with (cache / "sample_index.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_date(text: str) -> str:
    return datetime.strptime(text, "%Y-%m-%d").date().isoformat()


def sample_for_date(rows: list[dict[str, str]], date_text: str) -> int:
    date_text = normalize_date(date_text)
    for row in rows:
        if str(row.get("date", ""))[:10] == date_text:
            return int(row["sample_id"])
    raise KeyError(f"{date_text} is absent from sample_index.csv")


def land_mask(path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Read one Bremen 2-D slice only; land is static."""
    with xr.open_dataset(path) as ds:
        name = "z" if "z" in ds else next(k for k, v in ds.data_vars.items() if v.ndim >= 2)
        da = ds[name]
        for dim in tuple(da.dims):
            if dim not in ("x", "y"):
                da = da.isel({dim: 0})
        da = da.notnull()
        if "y" in da.dims and "x" in da.dims:
            da = da.transpose("y", "x")
        if da.shape != (len(y), len(x)):
            da = da.interp(x=x, y=y, method="nearest")
        return ~np.asarray(da.values, dtype=bool)


def expand_sea_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray, expand_km: float) -> np.ndarray:
    """Dilate a sea mask by a physical distance on the projected grid."""
    mask = np.asarray(mask, dtype=bool)
    if expand_km <= 0:
        return mask
    if not mask.any():
        return mask
    dx_m = float(np.median(np.abs(np.diff(x))))
    dy_m = float(np.median(np.abs(np.diff(y))))
    distance_m = distance_transform_edt(~mask, sampling=(dy_m, dx_m))
    return mask | (distance_m <= float(expand_km) * 1000.0)


def route_domain(ds: xr.Dataset, mode: str, shipping_variable: str, named_variables: str,
                 x: np.ndarray, y: np.ndarray, cap_min_lat: float,
                 named_sea_expand_km: float = 0.0) -> np.ndarray:
    """Return either the historical route mask or the broad named-sea union."""
    if mode == "shipping-mask":
        if shipping_variable not in ds:
            raise RuntimeError(f"Route mask variable is absent: {shipping_variable}")
        out = np.asarray(ds[shipping_variable].transpose("y", "x").values > .5)
        # The combined historical mask can be narrow around the Chukchi and
        # East Siberian seas.  If requested, add physically dilated named-sea
        # masks while retaining the original shipping mask everywhere else.
        if named_sea_expand_km > 0:
            expanded = []
            for name in (v.strip() for v in named_variables.split(",") if v.strip()):
                if name in ds:
                    sea = np.asarray(ds[name].transpose("y", "x").values > .5)
                    out |= expand_sea_mask(sea, x, y, named_sea_expand_km)
                    expanded.append(name)
            if expanded:
                print(f"expanded named sea masks by {named_sea_expand_km:g} km = {', '.join(expanded)}")
        return out
    out = np.zeros((ds.sizes["y"], ds.sizes["x"]), dtype=bool)
    sea_masks: list[tuple[str, np.ndarray]] = []
    found = []
    for name in (v.strip() for v in named_variables.split(",") if v.strip()):
        if name in ds:
            sea = np.asarray(ds[name].transpose("y", "x").values > .5)
            sea = expand_sea_mask(sea, x, y, named_sea_expand_km)
            out |= sea
            found.append(name)
            sea_masks.append((name, sea))
    if not found or not out.any():
        raise RuntimeError("No named NEP sea masks found for illustrative route domain")
    print("illustrative route domain masks =", ", ".join(found))
    if mode == "named-nep-seas":
        return out

    # Build the red-line-equivalent cap reproducibly: take the northmost
    # projected point of each named sea, order these tips west-to-east, join
    # them, then fill the high-latitude ocean south of that upper envelope.
    # This fills gaps between adjacent sea masks (notably around Chukchi) but
    # does not change any already-reported navigation-window statistics.
    xx, yy = np.meshgrid(x, y)
    vx, vy = TO_VIEW.transform(xx, yy)
    _, lat = TO_GEO.transform(xx, yy)
    anchors = []
    for name, sea in sea_masks:
        rr, cc = np.where(sea)
        if not rr.size:
            continue
        k = int(np.argmax(vy[rr, cc]))
        anchors.append((float(vx[rr[k], cc[k]]), float(vy[rr[k], cc[k]]), name))
    anchors.sort(key=lambda item: item[0])
    if len(anchors) < 2:
        raise RuntimeError("Automatic NEP cap needs at least two named-sea anchor points")
    ax = np.asarray([v[0] for v in anchors]); ay = np.asarray([v[1] for v in anchors])
    upper = np.interp(vx, ax, ay, left=np.nan, right=np.nan)
    cap = np.isfinite(upper) & (vy <= upper) & (lat >= cap_min_lat)
    for px, py, name in anchors:
        lon, plat = TO_GEO.transform(*Transformer.from_crs(VIEW_CRS, SRC_CRS, always_xy=True).transform(px, py))
        print(f"cap anchor {name:>18s}: lon={lon % 360:.2f}E lat={plat:.2f}N")
    print(f"automatic cap cells={int(cap.sum())}; cap_min_lat={cap_min_lat:g}N")
    return out | cap


def nearest_xy(x: np.ndarray, y: np.ndarray, lon: float, lat: float,
               allowed: np.ndarray) -> tuple[int, int]:
    """Snap a geographic terminal to the closest *route-ocean* cell.

    This must match the daily A* evaluation exactly.  Snapping merely to the
    nearest grid node can place either terminal on land or outside the route
    mask, making every date appear blocked even when the evaluation CSV says
    it is passable.
    """
    px, py = Transformer.from_crs("EPSG:4326", SRC_CRS, always_xy=True).transform(lon, lat)
    distance2 = (y[:, None] - py) ** 2 + (x[None, :] - px) ** 2
    if allowed.shape != distance2.shape or not np.any(allowed):
        raise RuntimeError("Route-ocean mask has an invalid shape or no valid cells")
    return tuple(int(v) for v in np.unravel_index(np.argmin(np.where(allowed, distance2, np.inf)), distance2.shape))


def astar_path(passable: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
               clearance_m: np.ndarray | None = None, clearance_scale_m: float = 25_000.0,
               clearance_weight: float = 4.0) -> list[tuple[int, int]] | None:
    """Eight-neighbour, clearance-aware A* returning row/column cells.

    ``passable`` still gives the scientific threshold rule.  Diagonal moves
    require both adjacent orthogonal cells to be passable, so a route cannot
    slip between two blocked cells that meet only at a corner. Clearance is a
    *soft* cost only: a narrow but valid strait remains usable, while a broad
    open-water area is preferred whenever it is available.
    """
    height, width = passable.shape
    if not (passable[start] and passable[goal]):
        return None
    moves = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5))
    heuristic = lambda r, c: float(np.hypot(goal[0] - r, goal[1] - c))
    frontier: list[tuple[float, float, tuple[int, int]]] = [(heuristic(*start), 0.0, start)]
    score = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        _, cost, current = heapq.heappop(frontier)
        if cost != score.get(current):
            continue
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1]
        r, c = current
        for dr, dc, step in moves:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width or not passable[nr, nc]:
                continue
            if dr and dc and (not passable[r, nc] or not passable[nr, c]):
                continue
            # This is an exponential safety penalty.  A cell immediately next
            # to land, the route boundary, or SIC > threshold costs up to
            # ``1 + clearance_weight`` times as much as a centre-of-channel
            # cell.  Farther than ~3 scale lengths it is effectively 1.
            if clearance_m is None or clearance_weight <= 0:
                safety_factor = 1.0
            else:
                distance = max(0.0, float(clearance_m[nr, nc]))
                safety_factor = 1.0 + clearance_weight * np.exp(-distance / max(clearance_scale_m, 1.0))
            new_cost = cost + step * safety_factor
            node = (nr, nc)
            if new_cost < score.get(node, float("inf")):
                score[node] = new_cost
                parent[node] = current
                heapq.heappush(frontier, (new_cost + heuristic(nr, nc), new_cost, node))
    return None


def find_truth_path(rows, Y, requested_date, allowed, threshold, start, goal, fallback,
                    dx_m, dy_m, clearance_scale_km, clearance_weight):
    requested_date = normalize_date(requested_date)
    sid = sample_for_date(rows, requested_date)
    def solve(candidate: int):
        passable = allowed & (np.asarray(Y[candidate, 0], dtype=np.float32) <= threshold)
        # Distance is measured from every available navigation cell to the
        # nearest forbidden cell (land, outside NEP mask, or SIC > threshold).
        clearance = distance_transform_edt(passable, sampling=(dy_m, dx_m))
        return astar_path(passable, start, goal, clearance, clearance_scale_km * 1000.0, clearance_weight)

    path = solve(sid)
    if path is not None:
        return requested_date, sid, path
    if not fallback:
        raise RuntimeError(f"Bremen A* has no path on {requested_date} at SIC <= {threshold:g}")
    year = requested_date[:4]
    print(f"[warning] no Bremen path on {requested_date}; seeking the first passable {year} day")
    for row in rows:
        date = str(row.get("date", ""))[:10]
        if not date.startswith(year):
            continue
        candidate = int(row["sample_id"])
        path = solve(candidate)
        if path is not None:
            return date, candidate, path
    raise RuntimeError(f"No Bremen A* path in {year} at SIC <= {threshold:g}")


def load_route_path_csv(path: Path, date: str, shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Read and validate a previously exported grid-cell A* route."""
    if date not in path.name:
        raise RuntimeError(f"Route filename does not contain requested date {date}: {path.name}")
    with path.open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    if len(records) < 2:
        raise RuntimeError(f"Route CSV has fewer than two cells: {path}")
    route: list[tuple[int, int]] = []
    for sequence, record in enumerate(records):
        if int(record["sequence"]) != sequence:
            raise RuntimeError(f"Route CSV sequence is broken at row {sequence}")
        row, col = int(record["row"]), int(record["column"])
        if not (0 <= row < shape[0] and 0 <= col < shape[1]):
            raise RuntimeError(f"Route cell {(row, col)} is outside grid {shape}")
        if route and max(abs(row - route[-1][0]), abs(col - route[-1][1])) != 1:
            raise RuntimeError(f"Route CSV contains a non-adjacent step at row {sequence}")
        route.append((row, col))
    return route


def route_buffer(path: list[tuple[int, int]], shape: tuple[int, int], x: np.ndarray, y: np.ndarray, km: float) -> np.ndarray:
    on_path = np.zeros(shape, dtype=bool)
    rr, cc = np.asarray(path).T
    on_path[rr, cc] = True
    dy = float(np.median(np.abs(np.diff(y))))
    dx = float(np.median(np.abs(np.diff(x))))
    distance_m = distance_transform_edt(~on_path, sampling=(dy, dx))
    return distance_m <= km * 1000.0


def view_mesh(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx, yy = np.meshgrid(x, y)
    return TO_VIEW.transform(xx, yy)


def crop_indices(x: np.ndarray, y: np.ndarray, limits: tuple[float, float, float, float]):
    vx, vy = view_mesh(x, y)
    in_frame = (vx >= limits[0]) & (vx <= limits[1]) & (vy >= limits[2]) & (vy <= limits[3])
    rows, cols = np.where(in_frame)
    if not rows.size:
        raise RuntimeError("The selected view contains no grid cells")
    return slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1)


def draw_panel(ax, field, visible, land, x, y, limits, *, kind: str, cmap=None, norm=None,
               coastline: bool = True, ice_edge: np.ndarray | None = None,
               bilinear_blocked: np.ndarray | None = None):
    """A completely borderless data layer.  Outside route buffer is transparent."""
    vx, vy = view_mesh(x, y)
    if kind == "sic":
        cmap = LinearSegmentedColormap.from_list("sic_purple_blue_buffer", ["#7ed5e5", "#1789d4", "#18368f", "#6e3d98"])
        values = np.ma.masked_where(~visible | ~np.isfinite(field) | (field < .01), field)
        # Low-concentration ice remains visible but recedes to 50% opacity;
        # the >=0.15 ice edge and denser ice stay fully opaque.
        alpha = np.where(np.ma.getmaskarray(values), 0.0,
                         np.where(np.asarray(values, dtype=float) < .15, .50, 1.0))
        ax.pcolormesh(vx, vy, values, shading="auto", cmap=cmap, vmin=.01, vmax=1,
                      alpha=alpha)
    else:
        values = np.ma.masked_where(~visible | ~np.isfinite(field), field)
        ax.pcolormesh(vx, vy, values, shading="auto", cmap=cmap, norm=norm)
    if coastline:
        # A slightly offset cool-grey shadow plus pale land fill gives the
        # coast a restrained floating effect, while retaining transparent ocean
        # and keeping the geographic reference distinct from all route lines.
        ax.contourf(vx + 3200.0, vy - 3200.0, land.astype(np.uint8), levels=[.5, 1.5],
                    colors=[LAND_SHADOW], alpha=.40, zorder=3)
        ax.contourf(vx, vy, land.astype(np.uint8), levels=[.5, 1.5],
                    colors=[LAND_FILL], alpha=1.0, zorder=4)
        ax.contour(vx, vy, land.astype(np.uint8), levels=[.5], colors="#64748b",
                   linewidths=.42, alpha=.95, zorder=5)
    if ice_edge is not None:
        # A common 0.15 SIC edge makes positional ice-edge changes readable in
        # both SIC and signed-bias layers.  It is masked to the same corridor.
        edge = np.ma.masked_where(~visible | ~np.isfinite(ice_edge), ice_edge)
        try:
            ax.contour(vx, vy, edge, levels=[.15], colors="#7c3aed", linewidths=.56,
                       linestyles="solid", alpha=.96, zorder=6)
        except ValueError:
            # A fully ice-free/ice-covered local corridor has no 0.15 contour.
            pass
    ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:]); ax.set_aspect("equal"); ax.set_axis_off()


def add_overlay(ax, buffer, path, x, y, limits, draw_route: bool = True):
    vx, vy = view_mesh(x, y)
    # The buffer boundary and the actual shortest path are intentionally kept
    # as very thin overlays, so the SIC / error layer remains the visual focus.
    # Cyan dashed line: buffer edge.  It differs deliberately from the purple
    # SIC=0.15 edge and grey geographic coastline.
    ax.contour(vx, vy, buffer.astype(np.uint8), levels=[.5], colors="#0891b2",
               linewidths=.72, linestyles="dashed", alpha=.92)
    if draw_route:
        rows, cols = np.asarray(path).T
        px, py = TO_VIEW.transform(x[cols], y[rows])
        # Thin opaque blue rim plus a much more transparent white inner
        # highlight.  This reads as a light pipeline without obscuring the
        # SIC/error texture.
        ax.plot(px, py, color="#075985", linewidth=.78, alpha=.98, solid_capstyle="round")
        ax.plot(px, py, color="#ffffff", linewidth=.28, alpha=.25, solid_capstyle="round")
    ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:]); ax.set_aspect("equal"); ax.set_axis_off()


def save_path_csv(path: Path, route: list[tuple[int, int]], x: np.ndarray, y: np.ndarray) -> None:
    rows, cols = np.asarray(route).T
    lon, lat = TO_GEO.transform(x[cols], y[rows])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("sequence", "row", "column", "x_epsg3413_m", "y_epsg3413_m", "longitude", "latitude"))
        for i, (r, c, lo, la) in enumerate(zip(rows, cols, lon, lat)):
            writer.writerow((i, int(r), int(c), f"{x[c]:.3f}", f"{y[r]:.3f}", f"{lo:.7f}", f"{la:.7f}"))


def save_png(path: Path, field, visible, land, x, y, limits, *, kind, cmap=None, norm=None,
             buffer=None, route=None, coastline=True, ice_edge=None, bilinear_blocked=None):
    fig = plt.figure(figsize=(6.8, 3.0), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_panel(ax, field, visible, land, x, y, limits, kind=kind, cmap=cmap, norm=norm,
               coastline=coastline, ice_edge=ice_edge, bilinear_blocked=bilinear_blocked)
    if buffer is not None:
        add_overlay(ax, buffer, route or [], x, y, limits, draw_route=route is not None)
    fig.savefig(path, dpi=600, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print("saved=", path)


def save_changed_boxplot(output: Path, signed_b: np.ndarray, signed_m: np.ndarray,
                         improvement: np.ndarray, selected: np.ndarray) -> None:
    """Compare error and signed gain on the same meaningfully changed pixels."""
    data = [np.abs(signed_b[selected]), np.abs(signed_m[selected]), improvement[selected]]
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Tinos",
                                                              "Liberation Serif", "DejaVu Serif"]})
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor((1, 1, 1, 0))
    box = ax.boxplot(data, widths=.58, patch_artist=True, whis=(5, 95),
                     showfliers=False, showmeans=True,
                     medianprops=dict(color="#ffffff", linewidth=1.2),
                     meanprops=dict(marker="D", markerfacecolor="#ffffff",
                                    markeredgecolor="#263238", markersize=4),
                     whiskerprops=dict(color="#45515b", linewidth=.95),
                     capprops=dict(color="#45515b", linewidth=.95),
                     boxprops=dict(edgecolor="#263238", linewidth=.9))
    for patch, color in zip(box["boxes"], ("#7b8798", "#238b45", "#d95f5f")):
        patch.set_facecolor(color)
        patch.set_alpha(.88)
    ax.set_xticks([1, 2, 3], ["Bilinear", "Downscaled", "Improvement"])
    ax.set_ylabel("SIC error / improvement")
    ax.axhline(0, color="#45515b", linewidth=.7)
    ax.grid(axis="y", color="#b0b0b0", alpha=.35, linewidth=.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for suffix in ("png", "svg"):
        fig.savefig(Path(f"{output}.{suffix}"), dpi=600, transparent=True,
                    bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main() -> None:
    a = arguments()
    if not 0 <= a.min_sic_change <= 1:
        raise ValueError("--min-sic-change must be a SIC fraction in [0, 1]")
    cache = Path(a.cache)
    with (cache / "meta.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    X = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    Y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    rows = read_index(cache)
    channels = select_channels(list(meta["channel_names"]))

    with xr.open_dataset(a.mask_file, decode_times=False) as ds:
        x = np.asarray(ds.x.values)
        y = np.asarray(ds.y.values)
        route_mask = route_domain(ds, a.route_domain, a.mask_var, a.named_sea_vars,
                                  x, y, a.cap_min_lat, a.named_sea_expand_km)
    full_x, full_y = x.copy(), y.copy()
    land = land_mask(Path(a.land_file), x, y)
    if route_mask.shape != Y.shape[-2:] or land.shape != route_mask.shape:
        raise RuntimeError(f"Grid mismatch: route={route_mask.shape}, land={land.shape}, Y={Y.shape[-2:]}")
    allowed = route_mask & ~land
    start = nearest_xy(x, y, a.start_lon, a.start_lat, allowed)
    end = nearest_xy(x, y, a.end_lon, a.end_lat, allowed)
    dx_m = float(np.median(np.abs(np.diff(x))))
    dy_m = float(np.median(np.abs(np.diff(y))))
    if a.route_path_csv:
        selected_date = normalize_date(a.date)
        sid = sample_for_date(rows, selected_date)
        path = load_route_path_csv(Path(a.route_path_csv), selected_date, allowed.shape)
        path_rows, path_cols = np.asarray(path).T
        truth_on_path = np.asarray(Y[sid, 0, path_rows, path_cols], dtype=np.float32)
        blocked = ~np.isfinite(truth_on_path) | (truth_on_path > a.threshold + 1e-6)
        on_land = land[path_rows, path_cols]
        if np.any(blocked) or np.any(on_land):
            raise RuntimeError(
                f"Reused route is not valid on {selected_date}: "
                f"SIC-blocked={int(blocked.sum())}, land={int(on_land.sum())}"
            )
        outside_domain = int((~allowed[path_rows, path_cols]).sum())
        if outside_domain:
            print(f"[warning] reused route has {outside_domain} cells outside current route domain; "
                  "these cells will be retained for this figure")
            route_mask[path_rows, path_cols] = True
            allowed[path_rows, path_cols] = True
        # The old route's terminals are authoritative. Re-snapping the
        # geographic endpoints after a mask expansion can select newly added,
        # ice-blocked cells and incorrectly report that this date has no route.
        start, end = path[0], path[-1]
        print(f"reused A* route from {a.route_path_csv}; date={selected_date}; cells={len(path)}")
    else:
        selected_date, sid, path = find_truth_path(
            rows, Y, a.date, allowed, a.threshold, start, end, not a.no_date_fallback,
            dx_m, dy_m, a.clearance_scale_km, a.clearance_weight,
        )
    print(f"path date={selected_date}; cells={len(path)}; threshold={a.threshold:g}; buffer={a.buffer_km:g} km")
    print(f"route domain={a.route_domain}; route-ocean cells={int(allowed.sum())}; "
          f"named-sea expansion={a.named_sea_expand_km:g} km")
    print(f"path mode=clearance-aware; scale={a.clearance_scale_km:g} km; weight={a.clearance_weight:g}")

    buffer = route_buffer(path, route_mask.shape, x, y, a.buffer_km)
    # Route mask keeps the corridor within the NEP navigation domain, while
    # land is removed from all SIC/error layers (transparent background).
    visible = (buffer & ~land) if a.no_route_domain_clip else (buffer & route_mask & ~land)
    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = safe_load(Path(a.checkpoint))
    model = build_model(checkpoint, device)
    inputs = torch.from_numpy(np.array(X[sid, channels], copy=True))[None].to(device)
    truth = np.asarray(Y[sid, 0], dtype=np.float32)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        pred = model(inputs)
    pred = pred.float().clamp(0, 1).cpu().numpy()[0, 0]
    bilinear = F.interpolate(inputs[:, 8:9].float(), size=truth.shape, mode="bilinear", align_corners=False).cpu().numpy()[0, 0]
    signed_b = bilinear - truth
    signed_m = pred - truth
    improvement = np.abs(signed_b) - np.abs(signed_m)  # positive = downscaling improved

    limits = (a.view_xmin, a.view_xmax, a.view_ymin, a.view_ymax)
    rs, cs = crop_indices(x, y, limits)
    x, y = x[cs], y[rs]
    truth, bilinear, pred = truth[rs, cs], bilinear[rs, cs], pred[rs, cs]
    signed_b, signed_m, improvement = signed_b[rs, cs], signed_m[rs, cs], improvement[rs, cs]
    visible, buffer, land_crop = visible[rs, cs], buffer[rs, cs], land[rs, cs]
    path_crop = [(r - rs.start, c - cs.start) for r, c in path if rs.start <= r < rs.stop and cs.start <= c < cs.stop]
    if not path_crop:
        raise RuntimeError("A* path is outside the selected map frame")

    # Use the historical shortest Bremen-reference path for the red blockage
    # annotation.  The clearance-aware presentation route can intentionally
    # detour around coarse-grid blockers and would hide the diagnostic.
    truth_full = np.asarray(Y[sid, 0], dtype=np.float32)
    reference_path = astar_path(
        allowed & np.isfinite(truth_full) & (truth_full <= a.threshold),
        start, end, clearance_weight=0.0,
    )
    reference_crop = ([(r - rs.start, c - cs.start) for r, c in reference_path
                       if rs.start <= r < rs.stop and cs.start <= c < cs.stop]
                     if reference_path else path_crop)

    # Restrict the optional red annotation to the actual route neighbourhood,
    # rather than painting the full 300-km presentation buffer.
    on_path = np.zeros(visible.shape, dtype=bool)
    pr, pc = np.asarray(path_crop).T
    on_path[pr, pc] = True
    path_band = distance_transform_edt(~on_path) <= a.block_highlight_pixels
    red_path = np.zeros(visible.shape, dtype=bool)
    rr, rc = np.asarray(reference_crop).T
    red_path[rr, rc] = True
    red_band = distance_transform_edt(~red_path) <= a.block_highlight_pixels

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"nep_full10_{selected_date}_astar{a.threshold:g}_buffer{a.buffer_km:g}km"
    # White at zero keeps the broad low-error corridor visually quiet.  The
    # former YlOrRd starts yellow, which made negligible errors look like a
    # filled background rather than allowing the orange/red hotspots to lead.
    # Signed bias: blue = SIC underestimation, white = unbiased, red = SIC
    # overestimation.  The same diverging palette is also used for absolute-
    # error improvement (blue = worsening, red = improvement).
    bias_cmap = LinearSegmentedColormap.from_list(
        "sic_bias_blue_white_red",
        ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
    )
    bias_cmap.set_bad(alpha=0)
    bias_limit = max(.04, float(np.nanpercentile(np.abs(np.r_[signed_b[visible], signed_m[visible]]), 98)))
    bias_norm = TwoSlopeNorm(vmin=-bias_limit, vcenter=0, vmax=bias_limit)
    gain_native_p98 = max(.04, float(np.nanpercentile(np.abs(improvement[visible]), 98)))
    gain_limit = bias_limit  # Shared d/e/f colour scale for directly comparable intensity.
    gain_cmap = bias_cmap
    gain_norm = bias_norm
    if a.stats_only:
        print(f"shared d/e/f colour limit: ±{bias_limit:.6f}; "
              f"f native |gain| p98 would be {gain_native_p98:.6f}")
        sic_change = np.abs(pred - bilinear)
        finite_visible = visible & np.isfinite(sic_change) & np.isfinite(improvement)
        print("|downscaled - bilinear| quantiles: " + ",".join(
            f"{v:.6f}" for v in np.percentile(sic_change[finite_visible],
                                                [0, 25, 50, 75, 90, 95, 99, 100])))
        for change_floor in (.005, .01, .02, .05):
            changed = finite_visible & (sic_change >= change_floor)
            selected_gain = improvement[changed]
            print(f"SIC-change >= {change_floor:.3f}: n={selected_gain.size}, "
                  f"fraction={selected_gain.size / finite_visible.sum():.3%}, "
                  f"improved={(selected_gain > 0).mean():.3%}, "
                  f"worsened={(selected_gain < 0).mean():.3%}, "
                  f"median-gain={np.median(selected_gain):.6f}, "
                  f"p95-gain={np.percentile(selected_gain, 95):.6f}")
        for name, field, limit in (("bilinear bias", signed_b, bias_limit),
                                   ("downscaled bias", signed_m, bias_limit),
                                   ("improvement", improvement, gain_limit)):
            sample = np.asarray(field[visible & np.isfinite(field)], dtype=float)
            q = np.percentile(sample, [0, 1, 5, 25, 50, 75, 95, 98, 99, 100])
            print(f"{name}: n={sample.size}; min,p01,p05,q1,median,q3,p95,p98,p99,max="
                  + ",".join(f"{v:.6f}" for v in q)
                  + f"; positive={(sample > 0).mean():.3%}; "
                  + f"colour-saturated={(np.abs(sample) > limit).mean():.3%}")
        return
    selected = visible & np.isfinite(truth) & np.isfinite(bilinear) & np.isfinite(pred)
    selected &= np.isfinite(improvement)
    eligible_count = int(selected.sum())
    if a.min_sic_change > 0:
        selected &= np.abs(pred - bilinear) >= a.min_sic_change
        if not selected.any():
            raise RuntimeError("No visible pixels exceed --min-sic-change")
        changed_gain = improvement[selected]
        tag = f"{a.min_sic_change:g}".replace(".", "p")
        summary = {
            "date": selected_date,
            "sic_change_threshold": a.min_sic_change,
            "eligible_pixels": eligible_count,
            "selected_pixels": int(selected.sum()),
            "selected_fraction": float(selected.sum() / eligible_count),
            "improved_pixels": int((changed_gain > 0).sum()),
            "worsened_pixels": int((changed_gain < 0).sum()),
            "unchanged_error_pixels": int((changed_gain == 0).sum()),
            "improved_fraction_selected": float((changed_gain > 0).mean()),
            "worsened_fraction_selected": float((changed_gain < 0).mean()),
            "median_signed_improvement": float(np.median(changed_gain)),
            "mean_signed_improvement": float(np.mean(changed_gain)),
            "improvement_q1": float(np.percentile(changed_gain, 25)),
            "improvement_q3": float(np.percentile(changed_gain, 75)),
            "improvement_p05": float(np.percentile(changed_gain, 5)),
            "improvement_p95": float(np.percentile(changed_gain, 95)),
            "shared_colour_limit": bias_limit,
        }
        summary_path = out / f"{prefix}_sicchange_ge{tag}_stats.csv"
        with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
        save_changed_boxplot(out / f"{prefix}_sicchange_ge{tag}_boxplot",
                             signed_b, signed_m, improvement, selected)
        print(f"selected SIC-change >= {a.min_sic_change:g}: "
              f"{summary['selected_pixels']}/{eligible_count} pixels; "
              f"improved={summary['improved_fraction_selected']:.1%}; "
              f"worsened={summary['worsened_fraction_selected']:.1%}; "
              f"median gain={summary['median_signed_improvement']:.5f}")
    overlay_buffer = buffer if a.no_route_domain_clip else (buffer & route_mask[rs, cs])
    # Cells along the actual route that are blocked in the 25-km/bilinear
    # field (SIC > threshold).  This is deliberately independent of the model
    # prediction: red means "the coarse route is blocked here", not necessarily
    # that downscaling opens it.  The layer is shown only on panel b.
    bilinear_blocked = red_band & (bilinear > a.threshold)
    print(f"bilinear-blocked cells on route ±{a.block_highlight_pixels:g} pixels = {int(bilinear_blocked.sum())}")
    # Six single, transparent-background PNG layers for PowerPoint layout.
    for suffix, field, kind, cmap, norm in (
        ("a_reference_sic", truth, "sic", None, None),
        ("b_bilinear_sic", bilinear, "sic", None, None),
        ("c_downscaled_sic", pred, "sic", None, None),
        ("d_signed_bias_bilinear", signed_b, "error", bias_cmap, bias_norm),
        ("e_signed_bias_downscaled", signed_m, "error", bias_cmap, bias_norm),
        ("f_absolute_error_improvement", improvement, "error", gain_cmap, gain_norm),
    ):
        if a.f_only and suffix != "f_absolute_error_improvement":
            continue
        panel_visible = selected if (a.min_sic_change > 0 and suffix == "f_absolute_error_improvement") else visible
        panel_suffix = (f"{suffix}_sicchange_ge{tag}"
                        if a.min_sic_change > 0 and suffix == "f_absolute_error_improvement" else suffix)
        save_png(out / f"{prefix}_{panel_suffix}.png", field, panel_visible, land_crop, x, y, limits,
                 kind=kind, cmap=cmap, norm=norm, buffer=overlay_buffer,
                 route=(None if a.no_route_line else path_crop),
                 coastline=not a.no_coastline,
                 ice_edge=((field if kind == "sic" else truth) if a.show_ice_edge else None),
                 bilinear_blocked=(bilinear_blocked if suffix == "b_bilinear_sic" else None))
    if a.f_only:
        return
    # Overlay-only layer, so buffer/path styling can also be adjusted in PPT.
    fig = plt.figure(figsize=(6.8, 3.0), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    if not a.no_coastline:
        vx, vy = view_mesh(x, y)
        ax.contour(vx, vy, land_crop.astype(np.uint8), levels=[.5], colors="#4b5563",
                   linewidths=.38, alpha=.9)
    add_overlay(ax, overlay_buffer, path_crop, x, y, limits, draw_route=not a.no_route_line)
    fig.savefig(out / f"{prefix}_route_buffer_overlay.png", dpi=600, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    fig = plt.figure(figsize=(6.8, 3.0), frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    vx, vy = view_mesh(x, y)
    ax.contourf(vx, vy, bilinear_blocked.astype(np.uint8), levels=[.5, 1.5],
                colors=["#e11d48"], alpha=.68)
    ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:]); ax.set_aspect("equal"); ax.set_axis_off()
    fig.savefig(out / f"{prefix}_bilinear_blocked_route_only_overlay.png", dpi=600,
                transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    save_path_csv(out / f"{prefix}_route_path.csv", path, full_x, full_y)


if __name__ == "__main__":
    main()
