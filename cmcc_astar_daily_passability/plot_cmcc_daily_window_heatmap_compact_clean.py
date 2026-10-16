#!/usr/bin/env python3
"""Refined 2015–2050 daily NEP passability heatmap (SIC threshold 0.15)."""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from cmcc_times_font import use_times_new_roman


ROOT = Path("/home/ubuntu/sic_dsc")
INPUT_2015_2049 = ROOT / "data_cmcc/astar_daily_nep_2015_2050_threshold015.csv"
INPUT_2050 = ROOT / "data_cmcc/astar_daily_nep_2050_test.csv"
OUTPUT_DIR = ROOT / "data_cmcc/cmcc_daily_window_heatmap_compact_clean"
YEARS = list(range(2015, 2051))
FIRST_DAY = date(2001, 7, 1)
LAST_DAY = date(2001, 11, 5)
DAYS = [(FIRST_DAY + timedelta(days=i)).strftime("%m-%d")
        for i in range((LAST_DAY - FIRST_DAY).days + 1)]

# A muted blue / amber palette replaces the original red / green pairing.
COLORS = ["#ffffff", "#47789e", "#dda356", "#9c79a7"]
LABELS = ["Blocked in both (blank)", "Passable in both", "Added by downscaling",
          "Lost after downscaling"]


def read_daily(path: Path, *, year_filter=None) -> dict[tuple[int, str], int]:
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["threshold"]) - 0.15) > 1e-8:
                continue
            year = int(row["year"])
            if year_filter is not None and year != year_filter:
                continue
            day = row["date"][5:]
            if day not in DAYS:
                continue
            key = (year, day)
            if key in records:
                raise ValueError(f"Duplicate daily record: {key}")
            before = int(row["before_passable"])
            after = int(row["after_passable"])
            if before not in (0, 1) or after not in (0, 1):
                raise ValueError(f"Nonbinary passability at {key}: {before}, {after}")
            records[key] = {(0, 0): 0, (1, 1): 1, (0, 1): 2, (1, 0): 3}[(before, after)]
    return records


def main() -> None:
    records = read_daily(INPUT_2015_2049)
    records.update(read_daily(INPUT_2050, year_filter=2050))
    missing = [(year, day) for year in YEARS for day in DAYS
               if (year, day) not in records]
    if missing:
        raise ValueError(f"Missing {len(missing)} daily records; first: {missing[:5]}")
    grid = np.array([[records[(year, day)] for year in YEARS] for day in DAYS],
                    dtype=np.int8)

    use_times_new_roman()
    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(13.4, 6.2), dpi=200)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    cmap = ListedColormap(COLORS)
    cmap.set_bad(alpha=0)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    x_edges = np.arange(len(YEARS) + 1) - 0.5
    y_edges = np.arange(len(DAYS) + 1) - 0.5
    ax.pcolormesh(x_edges, y_edges, np.ma.masked_equal(grid, 0), cmap=cmap, norm=norm,
                  edgecolors="none", linewidth=0, shading="flat",
                  antialiased=False, zorder=1)

    # Full-height year separators keep completely blocked years legible.
    ax.vlines(np.arange(0.5, len(YEARS) - 0.5, 1), -0.5, len(DAYS) - 0.5,
              colors="#8996a4", linewidth=0.55, alpha=0.63, zorder=2)
    month_starts = [DAYS.index(day) for day in ("07-01", "08-01", "09-01", "10-01", "11-01")]
    for pos in month_starts[1:]:
        ax.axhline(pos - 0.5, color="#8794a0", linewidth=0.65,
                   alpha=0.38, zorder=2)

    ax.set_xlim(-0.5, len(YEARS) - 0.5)
    ax.set_ylim(len(DAYS) - 0.5, -0.5)
    ax.set_xticks([i for i, year in enumerate(YEARS)
                   if (year - 2015) % 2 == 0 or year == 2050])
    ax.set_xticklabels([str(YEARS[i]) for i in ax.get_xticks().astype(int)], fontsize=8.5)
    ax.set_yticks(month_starts)
    ax.set_yticklabels(["Jul", "Aug", "Sep", "Oct", "Nov"], fontsize=9)
    ax.tick_params(axis="both", length=0, pad=5, colors="#4b5563")
    for spine in ax.spines.values():
        spine.set_color("#8f9baa")
        spine.set_linewidth(0.65)

    counts = np.bincount(grid.ravel(), minlength=4)
    handles = [Patch(facecolor="none", edgecolor="#9aa7b3", linewidth=0.7,
                     label=LABELS[0])]
    handles.extend(Patch(facecolor=color, edgecolor="none", label=label)
                   for color, label in zip(COLORS[1:], LABELS[1:]))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.015), fontsize=9, columnspacing=2.3,
               handlelength=1.5)
    fig.subplots_adjust(left=0.043, right=0.992, top=0.98, bottom=0.12)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "cmcc_nep_daily_passability_2015_2050_jul_nov_compact_clean.png"
    svg = png.with_suffix(".svg")
    fig.savefig(png, dpi=300, transparent=True)
    fig.savefig(svg, transparent=True)
    plt.close(fig)
    print(f"daily records={grid.size}, counts={dict(zip(LABELS, counts))}")
    print(f"saved={png}")
    print(f"saved={svg}")


if __name__ == "__main__":
    main()
