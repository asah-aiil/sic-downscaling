#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2x3 Bremen error-improvement map for the environment-guided full-10 model.

Copy this file into the full_10var training directory before running: it imports
that directory's config.py and model.py.  It performs inference only for the
requested independent 2024/2025 cache date; it never trains or changes weights.
"""
from __future__ import annotations

import argparse, csv, json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, ListedColormap
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from pyproj import CRS, Transformer

from config import Config
from model import UNetWithResidual

CHECKPOINT = "local-path/code/checkpoints/20260910_015743_envgated_t2m_u10_v10_msl_pr_ssr_tcc_tos_sic_mask_year_fixed_sic_residual_2014_2023_T1_d_rmse_fixed/best_model.pth"
CACHE = "local-path/data/cache_temporal_T3_float16_2024_2025_rebuilt"
MASK = "local-path/data/dataset/mask/eurasia_shipping_masks_6p25km_with_arctic.nc"
LAND = "local-path/data/dataset/target/Bremen_2025.nc"
ALIASES = {"t2m": ["t2m_d"], "u10": ["u10_d"], "v10": ["v10_d"], "msl": ["msl_d"],
           "pr": ["pr_d", "tp_d"], "ssr": ["ssr_d"], "tcc": ["tcc_d"],
           "tos": ["tos_d", "sst_d"], "sic": ["sic_d", "siconc_d"],
           "mask": ["mask_d", "sic_mask_d"]}
VARS = ["t2m", "u10", "v10", "msl", "pr", "ssr", "tcc", "tos", "sic", "mask"]
SRC_CRS = CRS.from_epsg(3413)
# Match the NEP study-area map (diejiajiancha.py), rather than centring on the
# pan-Arctic grid.
VIEW_CRS = CRS.from_proj4("+proj=stere +lat_0=90 +lat_ts=70 +lon_0=105 +datum=WGS84 +units=m +no_defs")
TO_VIEW = Transformer.from_crs(SRC_CRS, VIEW_CRS, always_xy=True)
TO_GEO = Transformer.from_crs(SRC_CRS, "EPSG:4326", always_xy=True)
VIEW_TO_GEO = Transformer.from_crs(VIEW_CRS, "EPSG:4326", always_xy=True)
GEO_TO_VIEW = Transformer.from_crs("EPSG:4326", VIEW_CRS, always_xy=True)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2024-08-01")
    p.add_argument("--checkpoint", default=CHECKPOINT)
    p.add_argument("--cache", default=CACHE)
    p.add_argument("--mask-file", default=MASK)
    p.add_argument("--mask-var", default="mask_all_shipping_seas")
    p.add_argument("--frame-mask-vars", default="mask_barents,mask_kara,mask_laptev,mask_east_siberian,mask_chukchi,mask_bering",
                   help="Comma-separated NEP subregion masks that define the fixed map window")
    p.add_argument("--land-file", default=LAND)
    p.add_argument("--output", default=None)
    p.add_argument("--boxplot-output", default=None,
                   help="Optional standalone transparent PNG for the three error distributions")
    p.add_argument("--export-six-panels", action="store_true",
                   help="Export six borderless, data-only PNG panels and skip the combined figure")
    p.add_argument("--panel-dir", default="local-path/fig",
                   help="Destination directory used with --export-six-panels")
    p.add_argument("--device", default="cuda")
    # The map frame is deliberately derived from the shipping-seas mask, as in
    # diejiajiancha.py.  A longitude/latitude bounding box is wrong here: in a
    # polar projection it includes an unwanted sector of the central Arctic.
    p.add_argument("--frame-lat-min", type=float, default=60.0,
                   help="Fixed NEP figure window: lower latitude bound")
    p.add_argument("--frame-lat-max", type=float, default=78.0,
                   help="Fixed NEP figure window: upper latitude bound; excludes central Arctic")
    p.add_argument("--frame-pad-x", type=float, default=.035)
    p.add_argument("--frame-pad-y", type=float, default=.030)
    # Final paper-frame limits in the NEP display projection (metres).  These
    # are intentionally explicit: a fitted bounding box can contain (0, 0),
    # i.e. the North Pole, even if the source mask itself does not.
    p.add_argument("--view-xmin", type=float, default=-1_950_000.0)
    p.add_argument("--view-xmax", type=float, default= 2_800_000.0)
    p.add_argument("--view-ymin", type=float, default=-1_950_000.0)
    p.add_argument("--view-ymax", type=float, default=  -350_000.0)
    return p.parse_args()


def safe_load(path: Path) -> Any:
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")


def state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(checkpoint.get(key), dict): return checkpoint[key]
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()): return checkpoint
    raise RuntimeError("Cannot find model state dict in checkpoint")


def build_model(checkpoint: Any, device: torch.device) -> UNetWithResidual:
    cfg = dict(checkpoint.get("config", {}))
    if [str(v).lower() for v in cfg.get("INPUT_VARS", [])] != VARS[:-1]:
        raise RuntimeError("Checkpoint is not the expected 10-variable full model")
    keys = ("MODEL_OUTPUT_MODE", "SIC_CHANNEL_INDEX", "MASK_CHANNEL_INDEX", "ENV_CHANNEL_INDICES",
            "SIC_RESIDUAL_SCALE", "SIC_BASELINE_INTERPOLATION", "ENV_GATE_INITIAL_BIAS",
            "ENV_GATE_SCALE", "SIC_RESIDUAL_ZERO_INIT")
    old = {key: getattr(Config, key, None) for key in keys}; exists = {key: hasattr(Config, key) for key in keys}
    try:
        for key in keys:
            if key in cfg: setattr(Config, key, cfg[key])
        model = UNetWithResidual(in_channels=10, out_channels=1)
    finally:
        for key in keys:
            if exists[key]: setattr(Config, key, old[key])
            elif hasattr(Config, key): delattr(Config, key)
    model.load_state_dict(state_dict(checkpoint), strict=True)
    return model.to(device).eval()


def channel_indices(names: list[str]) -> list[int]:
    chosen = []
    for variable in VARS:
        hit = next((candidate for candidate in ALIASES[variable] if candidate in names), None)
        if hit is None: raise RuntimeError(f"Missing {variable}; cache channels={names}")
        chosen.append(names.index(hit))
    return chosen


def sample_id_for_date(cache: Path, date_text: str) -> int:
    # Accept both 2025-07-10 and the shell-friendly 2025-7-10 spelling.
    # ``strptime`` accepts both 2025-07-10 and the shell-friendly 2025-7-10.
    normalized = datetime.strptime(date_text, "%Y-%m-%d").date().isoformat()
    with (cache / "sample_index.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("date", ""))[:10] == normalized: return int(row["sample_id"])
    raise KeyError(f"{normalized} is absent from {cache}/sample_index.csv")


def land_mask(path: Path, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with xr.open_dataset(path) as ds:
        name = "z" if "z" in ds else next(k for k, v in ds.data_vars.items() if v.ndim >= 2)
        da = ds[name]
        # Land is static.  Do *not* reduce over every daily Bremen field here:
        # that needlessly reads a whole year of 6.25-km data merely to obtain
        # a land mask.  One 2-D slice has the same land/sea geometry.
        for dim in tuple(da.dims):
            if dim not in ("x", "y"):
                da = da.isel({dim: 0})
        da = da.notnull()
        if "y" in da.dims and "x" in da.dims: da = da.transpose("y", "x")
        if da.shape != (len(y), len(x)):
            da = da.interp(x=x, y=y, method="nearest")
        return ~np.asarray(da.values, dtype=bool)


def crop_from_route(route: np.ndarray, x: np.ndarray, y: np.ndarray, pad: float = .025) -> tuple[float, float, float, float]:
    yy, xx = np.where(route)
    if not xx.size: raise RuntimeError("Route mask is empty")
    xmin, xmax = sorted((float(x[xx.min()]), float(x[xx.max()])))
    ymin, ymax = sorted((float(y[yy.min()]), float(y[yy.max()])))
    return xmin-(xmax-xmin)*pad, xmax+(xmax-xmin)*pad, ymin-(ymax-ymin)*pad, ymax+(ymax-ymin)*pad


def crop_from_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray, pad_cells: int = 2) -> tuple[float, float, float, float]:
    """Crop source grids to the complete requested geographic fan, not route data."""
    yy, xx = np.where(mask)
    if not xx.size:
        raise RuntimeError("Fan mask does not intersect the source grid")
    x0 = max(0, int(xx.min()) - pad_cells); x1 = min(len(x) - 1, int(xx.max()) + pad_cells)
    y0 = max(0, int(yy.min()) - pad_cells); y1 = min(len(y) - 1, int(yy.max()) + pad_cells)
    return float(min(x[x0], x[x1])), float(max(x[x0], x[x1])), float(min(y[y0], y[y1])), float(max(y[y0], y[y1]))


def fixed_nep_window(ds: xr.Dataset, names_text: str, fallback: np.ndarray) -> np.ndarray:
    """Build the same six-sea NEP window used by diejiajiancha.py.

    ``mask_all_shipping_seas`` can include a broad Arctic component suitable
    for routing, but unsuitable as a figure extent.  The named subregions give
    a stable, intentionally smaller paper figure window.
    """
    out = np.zeros_like(fallback, dtype=bool)
    found = []
    for name in (part.strip() for part in names_text.split(",") if part.strip()):
        if name in ds:
            out |= np.asarray(ds[name].transpose("y", "x").values > .5)
            found.append(name)
    if not found:
        print("[warning] no requested frame masks found; falling back to", "mask_all_shipping_seas")
        return fallback
    print("frame masks =", ", ".join(found))
    return out


def crop(a: np.ndarray, x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    ix = np.flatnonzero((x >= xmin) & (x <= xmax)); iy = np.flatnonzero((y >= ymin) & (y <= ymax))
    return a[np.ix_(iy, ix)], x[ix], y[iy]


def view_mesh(x: np.ndarray, y: np.ndarray):
    xx, yy = np.meshgrid(x, y)
    return TO_VIEW.transform(xx, yy)


def fan_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Annular NEP sector: 20–190E and 62–85N; poleward area is blank."""
    xx, yy = np.meshgrid(x, y)
    lon, lat = TO_GEO.transform(xx, yy)
    lon = lon % 360.0
    return (lon >= 20.0) & (lon <= 190.0) & (lat >= 62.0) & (lat <= 85.0)


def fan_view_limits(pad: float = .012) -> tuple[float, float, float, float]:
    lon = np.linspace(20.0, 190.0, 721)
    lon_all = np.r_[lon, np.full(721, 190.0), lon[::-1], np.full(721, 20.0)]
    lat_all = np.r_[np.full(721, 85.0), np.linspace(85.0, 62.0, 721),
                    np.full(721, 62.0), np.linspace(62.0, 85.0, 721)]
    px, py = GEO_TO_VIEW.transform(lon_all, lat_all)
    xmin, xmax, ymin, ymax = np.nanmin(px), np.nanmax(px), np.nanmin(py), np.nanmax(py)
    return (xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin),
            ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))


def frame_limits_from_shipping_mask(route, x, y, pad_x=.035, pad_y=.030,
                                    lat_min=60.0, lat_max=78.0):
    """Frame the actual NEP shipping-seas mask, not a polar lon/lat box.

    This mirrors ``set_centered_mask_extent`` in diejiajiancha.py.  It keeps
    the Kara--Bering coastal corridor and removes the unrelated central-Arctic
    sector that a geographic rectangle retains after reprojection.
    """
    if not np.any(route):
        raise RuntimeError("Shipping-seas mask is empty")
    xx, yy = np.meshgrid(x, y)
    _, latitude = TO_GEO.transform(xx, yy)
    # The six named seas alone can still extend towards the polar cap.  This
    # is the deliberate, fixed coastal NEP band requested for the paper map.
    route = route & np.isfinite(latitude) & (latitude >= lat_min) & (latitude <= lat_max)
    if not np.any(route):
        raise RuntimeError("No NEP frame pixels remain after latitude clipping")
    vx, vy = TO_VIEW.transform(xx, yy)
    selected_x, selected_y = vx[route], vy[route]
    xmin, xmax = np.nanmin(selected_x), np.nanmax(selected_x)
    ymin, ymax = np.nanmin(selected_y), np.nanmax(selected_y)
    width, height = xmax - xmin, ymax - ymin
    return (xmin - width * pad_x, xmax + width * pad_x,
            ymin - height * pad_y, ymax + height * pad_y)


def _unwrap_lon(lon: np.ndarray, center: float = 112.5) -> np.ndarray:
    return (lon - center + 180.0) % 360.0 - 180.0 + center


def _label_lon(value: float) -> str:
    value = value % 360.0
    if np.isclose(value, 180.0): return "180°"
    if value <= 180.0: return f"{value:.0f}°E"
    return f"{360.0 - value:.0f}°W"


def _label_lat(value: float) -> str:
    return f"{value:.0f}°N"


def geographic_ticks(limits: tuple[float, float, float, float]):
    """Return five latitude and eight longitude ticks fitted to the viewport."""
    xmin, xmax, ymin, ymax = limits
    xs = np.linspace(xmin, xmax, 80); ys = np.linspace(ymin, ymax, 80)
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = VIEW_TO_GEO.transform(xx, yy)
    good = np.isfinite(lon) & np.isfinite(lat)
    lon = _unwrap_lon(lon[good]); lat = lat[good]
    # Keep evenly distributed labels even for a narrow latitude span.
    lat_ticks = np.linspace(np.nanmin(lat), np.nanmax(lat), 5)
    lon_ticks = np.linspace(np.nanmin(lon), np.nanmax(lon), 8)
    return lat_ticks, lon_ticks


def add_graticule(ax, limits, lat_ticks, lon_ticks, show_left=False,
                  show_right=False, show_bottom=False):
    """Draw projected geographic grid lines with labels only on outer panels."""
    xmin, xmax, ymin, ymax = limits
    lon_line = np.linspace(np.min(lon_ticks) - 18, np.max(lon_ticks) + 18, 361)
    lat_line = np.linspace(max(35, np.min(lat_ticks) - 8), min(89.5, np.max(lat_ticks) + 8), 241)
    line_style = dict(color="#607d8b", linewidth=.42, alpha=.42, zorder=1.5)
    for lat in lat_ticks:
        gx, gy = GEO_TO_VIEW.transform(lon_line, np.full_like(lon_line, lat))
        valid = np.isfinite(gx) & np.isfinite(gy)
        ax.plot(gx[valid], gy[valid], **line_style)
        if show_left or show_right:
            # Locate where this parallel meets each vertical plot edge.
            if show_left:
                i = np.nanargmin(np.abs(gx - xmin)); y = gy[i]
                if ymin <= y <= ymax:
                    ax.text(-.012, (y-ymin)/(ymax-ymin), _label_lat(lat), transform=ax.transAxes,
                            ha="right", va="center", fontsize=6.7, color="#45616d", clip_on=False)
            if show_right:
                i = np.nanargmin(np.abs(gx - xmax)); y = gy[i]
                if ymin <= y <= ymax:
                    ax.text(1.012, (y-ymin)/(ymax-ymin), _label_lat(lat), transform=ax.transAxes,
                            ha="left", va="center", fontsize=6.7, color="#45616d", clip_on=False)
    for lon in lon_ticks:
        gx, gy = GEO_TO_VIEW.transform(np.full_like(lat_line, lon), lat_line)
        valid = np.isfinite(gx) & np.isfinite(gy)
        ax.plot(gx[valid], gy[valid], **line_style)
        if show_bottom:
            i = np.nanargmin(np.abs(gy - ymin)); x = gx[i]
            if xmin <= x <= xmax:
                ax.text((x-xmin)/(xmax-xmin), -.055, _label_lon(lon), transform=ax.transAxes,
                        ha="center", va="top", fontsize=6.7, color="#45616d", clip_on=False)


def draw_base(ax, sic, land, x, y, panel_label, limits, with_ice=True):
    """White land, unfilled ocean, and the established blue-to-purple SIC scale."""
    vx, vy = view_mesh(x, y)
    # The ocean is deliberately masked (no fill); land remains white for a
    # clean paper/PPT background without the former blue/yellow basemap.
    fan = fan_mask(x, y)
    ax.pcolormesh(vx, vy, np.ma.masked_where(~(land & fan), np.ones_like(land, dtype=float)),
                  shading="auto", cmap=ListedColormap(["#ffffff"]), vmin=0, vmax=1)
    image = None
    if with_ice:
        ice = np.ma.masked_where((land & ~fan) | ~fan | ~np.isfinite(sic) | (sic < .01), sic)
        cmap = LinearSegmentedColormap.from_list("sic_purple_blue", ["#7ed5e5", "#1789d4", "#18368f", "#6e3d98"])
        image = ax.pcolormesh(vx, vy, ice, shading="auto", cmap=cmap, vmin=.01, vmax=1)
    ax.text(.018, .962, panel_label, transform=ax.transAxes, ha="left", va="top",
            fontsize=10.5, fontweight="bold", color="#111111", zorder=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal"); ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
    return image, vx, vy


def draw_data_only(ax, sic, land, x, y, limits, *, with_ice=True,
                   value=None, cmap=None, norm=None, threshold=0.0):
    """Draw only the map data layer for later layout in PowerPoint.

    There are intentionally no axes, frame, label, grid, title, or colourbar.
    The pale land/ocean base remains so an exported error layer retains its
    geographical context without overlaying SIC.
    """
    vx, vy = view_mesh(x, y)
    fan = fan_mask(x, y)
    ax.pcolormesh(vx, vy, np.ma.masked_where(~(land & fan), np.ones_like(land, dtype=float)),
                  shading="auto", cmap=ListedColormap(["#ffffff"]), vmin=0, vmax=1,
                  zorder=0)
    if with_ice:
        ice = np.ma.masked_where(~fan | land | ~np.isfinite(sic) | (sic < .01), sic)
        sic_cmap = LinearSegmentedColormap.from_list(
            "sic_purple_blue_export", ["#7ed5e5", "#1789d4", "#18368f", "#6e3d98"]
        )
        ax.pcolormesh(vx, vy, ice, shading="auto", cmap=sic_cmap,
                      vmin=.01, vmax=1, zorder=1)
    elif value is not None:
        ax.pcolormesh(vx, vy, np.ma.masked_where((~fan) | land | (np.abs(value) < threshold), value),
                      shading="auto", cmap=cmap, norm=norm, alpha=.92, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
    ax.set_axis_off()


def export_error_boxplot(output: Path, signed_b: np.ndarray, signed_m: np.ndarray,
                         gain: np.ndarray, land: np.ndarray,
                         x: np.ndarray, y: np.ndarray) -> None:
    """Export the inset statistics as a standalone transparent PNG."""
    valid = fan_mask(x, y) & (~land) & np.isfinite(signed_b) & np.isfinite(signed_m) & np.isfinite(gain)
    data = [np.abs(signed_b[valid]).ravel(),
            np.abs(signed_m[valid]).ravel(),
            gain[valid].ravel()]
    fig, axis = plt.subplots(figsize=(4.4, 3.4), dpi=600)
    bp = axis.boxplot(
        data, patch_artist=True, widths=.58, showfliers=False,
        medianprops=dict(color="#202020", linewidth=1.2),
        whiskerprops=dict(color="#444444", linewidth=.9),
        capprops=dict(color="#444444", linewidth=.9),
        boxprops=dict(edgecolor="#333333", linewidth=.8),
    )
    for patch, color in zip(bp["boxes"], ("#7b8798", "#238b45", "#d95f5f")):
        patch.set_facecolor(color); patch.set_alpha(.90)
    axis.set_xticks([1, 2, 3], ["Interpolation", "Downscaling", "Improvement"])
    axis.set_ylabel("SIC error / improvement")
    axis.axhline(0, color="#555555", linewidth=.7, alpha=.6)
    axis.grid(axis="y", color="#b0b0b0", alpha=.35, linewidth=.5)
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    fig.patch.set_alpha(0); axis.set_facecolor((1, 1, 1, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, transparent=True, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    print("boxplot saved=", output)


def export_six_panels(panel_dir, date_text, truth, bilinear, pred, land, x, y,
                      limits, signed_b, signed_m, gain, common_cmap, common_limit):
    """Write six standalone 600-dpi map panels without decorative elements."""
    panel_dir = Path(panel_dir); panel_dir.mkdir(parents=True, exist_ok=True)
    improvement_norm = TwoSlopeNorm(vmin=-common_limit, vcenter=0, vmax=common_limit)
    error_limit = max(.04, float(np.nanpercentile(np.abs(np.r_[signed_b.ravel(), signed_m.ravel()]), 98)))
    error_cmap = plt.colormaps["YlOrRd"].copy(); error_cmap.set_bad(alpha=0)
    panels = (
        ("a_reference_sic", truth, True, None),
        ("b_bilinear_sic", bilinear, True, None),
        ("c_downscaled_sic", pred, True, None),
        ("d_absolute_error_bilinear", truth, False, np.abs(signed_b), error_cmap, Normalize(vmin=0, vmax=error_limit)),
        ("e_absolute_error_downscaled", truth, False, np.abs(signed_m), error_cmap, Normalize(vmin=0, vmax=error_limit)),
        ("f_absolute_error_improvement", truth, False, gain, common_cmap, improvement_norm),
    )
    for panel in panels:
        suffix, sic, with_ice, value = panel[:4]
        cmap, norm = (None, None) if with_ice else panel[4:]
        # 6.8 x 3.0 in preserves the wide NEP framing while yielding a crisp
        # 4080 x 1800 pixel layer at 600 dpi.
        fig = plt.figure(figsize=(6.8, 3.0), frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        draw_data_only(ax, sic, land, x, y, limits, with_ice=with_ice,
                       value=value, cmap=cmap, norm=norm, threshold=.005)
        out = panel_dir / f"nep_full10_{date_text}_{suffix}.png"
        fig.savefig(out, dpi=600, transparent=False, facecolor="white", edgecolor="none",
                    bbox_inches=None, pad_inches=0)
        plt.close(fig)
        print("panel saved=", out)


def main() -> None:
    a = arguments(); cache = Path(a.cache); checkpoint = safe_load(Path(a.checkpoint))
    with (cache / "meta.json").open(encoding="utf-8") as f: meta = json.load(f)
    X = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r"); Y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    sid = sample_id_for_date(cache, a.date); channels = channel_indices(list(meta["channel_names"]))
    device = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    inputs = torch.from_numpy(np.array(X[sid, channels], copy=True))[None].to(device)
    truth = np.asarray(Y[sid, 0], dtype=np.float32)
    model = build_model(checkpoint, device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        prediction = model(inputs)
    bilinear = F.interpolate(inputs[:, 8:9].float(), size=truth.shape, mode="bilinear", align_corners=False)
    pred = prediction.float().clamp(0, 1).cpu().numpy()[0, 0]; bilinear = bilinear.cpu().numpy()[0, 0]
    with xr.open_dataset(a.mask_file, decode_times=False) as ds:
        route_da = ds[a.mask_var].transpose("y", "x")
        route = np.asarray(route_da.values > .5); x, y = np.asarray(ds.x.values), np.asarray(ds.y.values)
        frame_route = fixed_nep_window(ds, a.frame_mask_vars, route)
    if route.shape != truth.shape: raise RuntimeError(f"mask shape {route.shape} != target {truth.shape}")
    land = land_mask(Path(a.land_file), x, y); bounds = crop_from_mask(fan_mask(x, y), x, y)
    truth, cx, cy = crop(truth, x, y, bounds); bilinear, _, _ = crop(bilinear, x, y, bounds); pred, _, _ = crop(pred, x, y, bounds); land, _, _ = crop(land, x, y, bounds); frame_route, _, _ = crop(frame_route, x, y, bounds)
    signed_b, signed_m = bilinear-truth, pred-truth; gain = np.abs(signed_b)-np.abs(signed_m)
    fan = fan_mask(cx, cy)
    common_limit = max(.04, float(np.nanpercentile(np.abs(np.r_[signed_b.ravel(), signed_m.ravel(), gain.ravel()]), 96)))
    common_cmap = plt.colormaps["RdBu_r"].copy(); common_cmap.set_bad(alpha=0)
    boxplot_output = Path(a.boxplot_output or f"local-path/figures/nep_full10_error_boxplot_{a.date}.png")
    export_error_boxplot(boxplot_output, signed_b, signed_m, gain, land, cx, cy)
    # Keep a deterministic, publication-scale window.  The mask is still used
    # to crop the source arrays efficiently, but never allowed to expand the
    # visible frame back towards the North Pole.
    limits = fan_view_limits()
    xmin, xmax, ymin, ymax = limits
    print("fixed NEP latitude band = " f"{a.frame_lat_min:.2f} to {a.frame_lat_max:.2f} N")
    print("projected figure limits (m) = " f"x[{xmin:.0f}, {xmax:.0f}], y[{ymin:.0f}, {ymax:.0f}]")
    # Corner values describe the actual rectangular plotting frame after its
    # projection-based mask fit; longitude is normalized to 0--360 degrees.
    for label, px, py in (("lower-left", xmin, ymin), ("lower-right", xmax, ymin),
                          ("upper-right", xmax, ymax), ("upper-left", xmin, ymax)):
        lon, lat = VIEW_TO_GEO.transform(px, py)
        print(f"figure corner {label:>11s} = lon {lon % 360:.3f} E, lat {lat:.3f} N")
    if a.export_six_panels:
        export_six_panels(a.panel_dir, a.date, truth, bilinear, pred, land, cx, cy,
                          limits, signed_b, signed_m, gain, common_cmap, common_limit)
        return
    fig = plt.figure(figsize=(15.0, 7.55)); grid = fig.add_gridspec(4, 3, height_ratios=[1,.042,1,.042], left=.050,right=.950,top=.975,bottom=.085,wspace=.040,hspace=.085)
    ax = np.array([[fig.add_subplot(grid[0, i]) for i in range(3)], [fig.add_subplot(grid[2, i]) for i in range(3)]])
    sic_img, vx, vy = draw_base(ax[0,0], truth, land, cx, cy, "(a)", limits)
    draw_base(ax[0,1], bilinear, land, cx, cy, "(b)", limits)
    draw_base(ax[0,2], pred, land, cx, cy, "(c)", limits)
    # Small errors are transparent: lower panels retain the shared ocean/land base.
    display_threshold = .005
    for axis, error, label in ((ax[1,0],signed_b,"(d)"),(ax[1,1],signed_m,"(e)")):
        draw_base(axis, truth, land, cx, cy, label, limits, with_ice=False); image=axis.pcolormesh(vx,vy,np.ma.masked_where((~fan) | land | (np.abs(error)<display_threshold),error),shading="auto",cmap=common_cmap,norm=TwoSlopeNorm(vmin=-common_limit,vcenter=0,vmax=common_limit),alpha=.92)
    draw_base(ax[1,2], truth, land, cx, cy, "(f)", limits, with_ice=False)
    improve_img=ax[1,2].pcolormesh(vx,vy,np.ma.masked_where((~fan) | land | (np.abs(gain)<display_threshold),gain),shading="auto",cmap=common_cmap,norm=TwoSlopeNorm(vmin=-common_limit,vcenter=0,vmax=common_limit),alpha=.92)
    # Compact statistical inset: retain only valid ocean pixels in the
    # displayed NEP window.  The third distribution keeps the signed
    # improvement convention: positive values mean downscaling reduced error.
    valid = (~land) & np.isfinite(signed_b) & np.isfinite(signed_m) & np.isfinite(gain)
    box_data = [np.abs(signed_b[valid]).ravel(),
                np.abs(signed_m[valid]).ravel(),
                gain[valid].ravel()]
    inset = ax[1,2].inset_axes([0.045, 0.055, 0.31, 0.30])
    bp = inset.boxplot(box_data, patch_artist=True, widths=.62,
                       showfliers=False, medianprops=dict(color="#202020", linewidth=1.0),
                       whiskerprops=dict(color="#444444", linewidth=.75),
                       capprops=dict(color="#444444", linewidth=.75),
                       boxprops=dict(edgecolor="#333333", linewidth=.65))
    for patch, color in zip(bp["boxes"], ("#7b8798", "#238b45", "#d95f5f")):
        patch.set_facecolor(color); patch.set_alpha(.88)
    inset.set_xticks([1, 2, 3], ["Interp.", "Downsc.", "Gain"], fontsize=5.5)
    inset.tick_params(axis="y", labelsize=5.5, length=2)
    inset.set_facecolor((1, 1, 1, .88)); inset.grid(axis="y", color="#bbbbbb", alpha=.35, linewidth=.35)
    inset.set_title("Error statistics", fontsize=6.2, pad=2)
    inset.set_zorder(20)
    cb1=fig.colorbar(sic_img,cax=fig.add_subplot(grid[1,:]),orientation="horizontal"); cb1.set_label("Sea-ice concentration")
    cb2=fig.colorbar(improve_img,cax=fig.add_subplot(grid[3,:]),orientation="horizontal"); cb2.set_label("Signed error (d,e) / absolute-error improvement (f)")
    lat_ticks, lon_ticks = geographic_ticks(limits)
    for row in range(2):
        add_graticule(ax[row, 0], limits, lat_ticks, lon_ticks, show_left=True)
        add_graticule(ax[row, 1], limits, lat_ticks, lon_ticks)
        add_graticule(ax[row, 2], limits, lat_ticks, lon_ticks, show_right=True,
                      show_bottom=(row == 1))
    output=Path(a.output or f"local-path/figures/nep_full10_error_2x3_{a.date}.png"); output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=400,bbox_inches="tight",facecolor="white"); plt.close(fig); print(f"saved={output}")

if __name__ == "__main__": main()
