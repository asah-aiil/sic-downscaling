#!/usr/bin/env python3
"""Plot a clean NEP study-region base map with a route corridor.

The figure contains no SIC field: pale-blue ocean, pale-yellow land, the
route buffer and a thin route line are exported on a transparent background.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pyproj import CRS, Transformer
from scipy.ndimage import distance_transform_edt
import xarray as xr

SRC_CRS = CRS.from_epsg(3413)
VIEW_CRS = CRS.from_proj4(
    "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=105 +datum=WGS84 +units=m +no_defs"
)
TO_VIEW = Transformer.from_crs(SRC_CRS, VIEW_CRS, always_xy=True)

MASK = "/home/ubuntu/sic_dsc/data/dataset/mask/eurasia_shipping_masks_6p25km_with_arctic.nc"
LAND = "/home/ubuntu/sic_dsc/data/dataset/target/Bremen_2025.nc"
ROUTE = "/home/ubuntu/sic_dsc/fig/nep_full10_2024-10-12_astar0.15_buffer300km_route_path.csv"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask-file", default=MASK)
    p.add_argument("--land-file", default=LAND)
    p.add_argument("--route-path-csv", default=ROUTE)
    p.add_argument("--buffer-km", type=float, default=400.0)
    p.add_argument("--include-named-seas", action="store_true",
                   help="Include the six named NEP sea masks in addition to the route buffer")
    p.add_argument("--named-sea-expand-km", type=float, default=100.0)
    p.add_argument("--no-route-line", action="store_true",
                   help="Do not draw the centerline; retain only the corridor outline")
    p.add_argument("--grid-lons", default="",
                   help="Comma-separated longitude grid lines in degrees east, e.g. 80,100,120")
    p.add_argument("--grid-lats", default="",
                   help="Comma-separated latitude grid lines in degrees north, e.g. 60,70,80")
    p.add_argument("--label-lons", default="",
                   help="Optional sparser longitude labels; defaults to --grid-lons")
    p.add_argument("--label-lats", default="",
                   help="Optional sparser latitude labels; defaults to --grid-lats")
    p.add_argument("--coastline-width", type=float, default=0.85)
    p.add_argument("--grid-width", type=float, default=0.70)
    p.add_argument("--show-grid-labels", action="store_true",
                   help="Add longitude/latitude labels around the map frame")
    p.add_argument("--arrow-step", type=int, default=140,
                   help="Route points per schematic arrow segment")
    p.add_argument("--arrow-color", default="#e87918")
    p.add_argument("--arrow-width", type=float, default=2.6)
    p.add_argument("--output", default="/home/ubuntu/sic_dsc/fig/nep_study_region.png")
    p.add_argument("--view-xmin", type=float, default=-2500000.0)
    p.add_argument("--view-xmax", type=float, default=3300000.0)
    p.add_argument("--view-ymin", type=float, default=-2400000.0)
    p.add_argument("--view-ymax", type=float, default=600000.0)
    p.add_argument("--fan-lon-min", type=float, default=20.0)
    p.add_argument("--fan-lon-max", type=float, default=190.0)
    p.add_argument("--fan-lat-min", type=float, default=55.0,
                   help="Southern latitude boundary; lowered to show the Eurasian Arctic coast")
    p.add_argument("--fan-lat-max", type=float, default=85.0)
    return p.parse_args()


def read_route(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["row"]), int(r["column"])))
    if not rows:
        raise RuntimeError(f"empty route file: {path}")
    return rows


def read_land(path: Path, x, y):
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


def main():
    a = args()
    with xr.open_dataset(a.mask_file, decode_times=False) as ds:
        x = np.asarray(ds.x.values)
        y = np.asarray(ds.y.values)
        named = np.zeros((len(y), len(x)), dtype=bool)
        if a.include_named_seas:
            for name in ("mask_barents", "mask_kara", "mask_laptev", "mask_east_siberian", "mask_chukchi", "mask_bering"):
                if name not in ds:
                    continue
                sea = np.asarray(ds[name].transpose("y", "x").values > .5)
                if a.named_sea_expand_km > 0 and sea.any():
                    dy0 = float(np.median(np.abs(np.diff(y))))
                    dx0 = float(np.median(np.abs(np.diff(x))))
                    dist = distance_transform_edt(~sea, sampling=(dy0, dx0))
                    sea |= dist <= a.named_sea_expand_km * 1000.0
                named |= sea
    land = read_land(Path(a.land_file), x, y)
    # Bremen's static mask can contain a spurious polar cap at the grid edge.
    # The study frame treats the region north of 88N as ocean.
    xx0, yy0 = np.meshgrid(x, y)
    lon0, lat0 = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True).transform(xx0, yy0)
    lon0 = lon0 % 360.0
    fan = ((lon0 >= a.fan_lon_min) & (lon0 <= a.fan_lon_max) &
           (lat0 >= a.fan_lat_min) & (lat0 <= a.fan_lat_max))
    # Fit the canvas to the annular sector so its centreline is centred in the
    # image rather than leaving a large rectangular margin around the map.
    glon = np.linspace(a.fan_lon_min, a.fan_lon_max, 721)
    boundary_lon = np.r_[glon, np.full(721, a.fan_lon_max), glon[::-1], np.full(721, a.fan_lon_min)]
    boundary_lat = np.r_[np.full(721, a.fan_lat_max),
                         np.linspace(a.fan_lat_max, a.fan_lat_min, 721),
                         np.full(721, a.fan_lat_min),
                         np.linspace(a.fan_lat_min, a.fan_lat_max, 721)]
    geo_to_view = Transformer.from_crs("EPSG:4326", VIEW_CRS, always_xy=True)
    bx, by = geo_to_view.transform(boundary_lon, boundary_lat)
    pad_x = .012 * (np.nanmax(bx) - np.nanmin(bx)); pad_y = .012 * (np.nanmax(by) - np.nanmin(by))
    a.view_xmin, a.view_xmax = float(np.nanmin(bx)-pad_x), float(np.nanmax(bx)+pad_x)
    a.view_ymin, a.view_ymax = float(np.nanmin(by)-pad_y), float(np.nanmax(by)+pad_y)
    land &= (lat0 < 88.0) & fan
    route = read_route(Path(a.route_path_csv))
    route_mask = np.zeros((len(y), len(x)), dtype=bool)
    rr, cc = np.asarray(route).T
    route_mask[rr, cc] = True
    dy = float(np.median(np.abs(np.diff(y))))
    dx = float(np.median(np.abs(np.diff(x))))
    corridor = distance_transform_edt(~route_mask, sampling=(dy, dx)) <= a.buffer_km * 1000.0
    if a.include_named_seas:
        corridor |= named
    corridor &= ~land & fan

    xx, yy = np.meshgrid(x, y)
    vx, vy = TO_VIEW.transform(xx, yy)
    fig = plt.figure(figsize=(10.5, 5.2), dpi=300, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ocean = np.where(fan, 1.0, np.nan)
    land_plot = np.where(fan, land.astype(float), np.nan)
    corridor_plot = np.where(fan, corridor.astype(float), np.nan)
    ax.pcolormesh(vx, vy, ocean, shading="auto", cmap=plt.matplotlib.colors.ListedColormap(["#e4f3f8"]), zorder=0)
    ax.contourf(vx, vy, land_plot, levels=[.5, 1.5], colors=["#fffbea"], zorder=2)
    ax.contour(vx, vy, land_plot, levels=[.5], colors="#7890a8", linewidths=a.coastline_width, zorder=3)
    # Corridor is a subtle translucent white area with a thin cyan boundary.
    ax.contourf(vx, vy, corridor_plot, levels=[.5, 1.5], colors=["#a9dceb"], alpha=.42, zorder=4)
    ax.contour(vx, vy, corridor_plot, levels=[.5], colors="#3b9fc3", linewidths=.70, zorder=5)
    ax.plot(bx, by, color="#303030", linewidth=.85, zorder=8)
    # Geographic grid lines without labels; labels can be added later in PPT.
    def draw_grid(values, is_lon):
        for value in values:
            if is_lon:
                # Keep meridians strictly inside the geographic study frame.
                lat_line = np.linspace(a.fan_lat_min, a.fan_lat_max, 600)
                lon_line = np.full_like(lat_line, value)
            else:
                # Sampling -180..180 then filtering modulo 360 created two
                # disjoint pieces that Matplotlib joined across the map.
                # Sample the selected continuous longitude span directly.
                lon_line = np.linspace(a.fan_lon_min, a.fan_lon_max, 800)
                lat_line = np.full_like(lon_line, value)
            gx, gy = geo_to_view.transform(lon_line, lat_line)
            ax.plot(gx, gy, color="#6b7280", linewidth=a.grid_width, alpha=.52,
                    linestyle=(0, (3, 4)), zorder=7)
    draw_grid([float(v) for v in a.grid_lons.split(",") if v.strip()], True)
    draw_grid([float(v) for v in a.grid_lats.split(",") if v.strip()], False)
    if a.show_grid_labels:
        xmin, xmax, ymin, ymax = a.view_xmin, a.view_xmax, a.view_ymin, a.view_ymax
        xrng, yrng = xmax - xmin, ymax - ymin
        def lon_label(v): return f"{abs(v):g}°{'E' if v >= 0 else 'W'}"
        def lat_label(v): return f"{abs(v):g}°{'N' if v >= 0 else 'S'}"
        label_lons = a.label_lons or a.grid_lons
        label_lats = a.label_lats or a.grid_lats
        for value in [float(v) for v in label_lons.split(",") if v.strip()]:
            gx, gy = geo_to_view.transform(value, a.fan_lat_min + .8)
            ax.text(gx, gy + .008 * yrng, lon_label(value), fontsize=9,
                    family="Tinos", color="#374151", ha="center", va="bottom",
                    clip_on=True, zorder=9)
        for value in [float(v) for v in label_lats.split(",") if v.strip()]:
            gx, gy = geo_to_view.transform(a.fan_lon_min + .8, value)
            ax.text(gx + .008 * xrng, gy - .006 * yrng, lat_label(value), fontsize=9,
                    family="Tinos", color="#374151", ha="left", va="top",
                    clip_on=True, zorder=9)
    route_keep = fan[rr, cc]
    px, py = TO_VIEW.transform(x[cc[route_keep]], y[rr[route_keep]])
    if not a.no_route_line:
        ax.plot(px, py, color="#075985", linewidth=.85, alpha=.95, zorder=6,
                solid_capstyle="round", solid_joinstyle="round")
    ax.set_xlim(a.view_xmin, a.view_xmax)
    ax.set_ylim(a.view_ymin, a.view_ymax)
    ax.set_aspect("equal")
    ax.set_axis_off()
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, transparent=False, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    # Export independently editable arrow segments on transparent backgrounds.
    arrow_dir = out.parent / (out.stem + "_arrows")
    arrow_dir.mkdir(parents=True, exist_ok=True)
    for k, start_i in enumerate(range(0, len(px) - 1, max(2, a.arrow_step)), 1):
        end_i = min(start_i + max(2, a.arrow_step), len(px) - 1)
        if end_i <= start_i:
            continue
        af = plt.figure(figsize=(10.5, 5.2), dpi=300, frameon=False)
        aa = af.add_axes([0, 0, 1, 1])
        aa.add_patch(FancyArrowPatch(
            (px[start_i], py[start_i]), (px[end_i], py[end_i]),
            arrowstyle="-|>", mutation_scale=15,
            linewidth=a.arrow_width, color=a.arrow_color,
            alpha=.92, connectionstyle="arc3,rad=0.0",
            capstyle="round", joinstyle="round"))
        aa.set_xlim(a.view_xmin, a.view_xmax); aa.set_ylim(a.view_ymin, a.view_ymax)
        aa.set_aspect("equal"); aa.set_axis_off()
        af.savefig(arrow_dir / f"route_arrow_{k:02d}.png", transparent=True,
                   bbox_inches=None, pad_inches=0)
        plt.close(af)
    print(f"saved={out}")
    print(f"arrow layers={arrow_dir}")
    print(f"route cells={len(route)}; buffer={a.buffer_km:g} km")


if __name__ == "__main__":
    main()
