#!/usr/bin/env python3
"""Pair annual navigable-day lines with a rotated daily status heatmap."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np

from cmcc_times_font import use_times_new_roman

from plot_cmcc_daily_window_heatmap_compact_clean import (
    COLORS, DAYS, INPUT_2015_2049, INPUT_2050, ROOT, YEARS, read_daily,
)


OUT = ROOT / "data_cmcc/cmcc_line_with_rotated_daily_heatmap_large_type"


def main() -> None:
    records = read_daily(INPUT_2015_2049)
    records.update(read_daily(INPUT_2050, year_filter=2050))
    missing = [(year, day) for year in YEARS for day in DAYS
               if (year, day) not in records]
    if missing:
        raise ValueError(f"Missing daily records: {missing[:5]}")
    # Rows now correspond to years and columns to calendar days.
    status = np.asarray([[records[(year, day)] for day in DAYS]
                         for year in YEARS], dtype=np.int8)
    before = np.count_nonzero((status == 1) | (status == 3), axis=1)
    after = np.count_nonzero((status == 1) | (status == 2), axis=1)

    use_times_new_roman()
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 14})
    fig = plt.figure(figsize=(15.8, 6.8), dpi=200)
    fig.patch.set_alpha(0)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.17,
                          left=0.067, right=0.945, top=0.92, bottom=0.19)
    line_ax = fig.add_subplot(gs[0, 0])
    heat_ax = fig.add_subplot(gs[0, 1])

    line_ax.plot(YEARS, before, "--o", color="#7b8798", linewidth=1.5,
                 markersize=3.2, label="Before downscaling", zorder=3)
    line_ax.plot(YEARS, after, "-o", color=COLORS[1], linewidth=1.9,
                 markersize=3.2, label="After downscaling", zorder=4)
    line_ax.fill_between(YEARS, before, after, where=after >= before,
                         color=COLORS[1], alpha=0.12, zorder=2)
    line_ax.set_xlim(2014.5, 2050.5)
    line_ax.set_ylim(0, max(120, int(np.ceil(after.max() / 20) * 20)))
    line_ax.set_xticks(list(range(2015, 2051, 5)))
    line_ax.set_xlabel("Year")
    line_ax.set_ylabel("Navigable days")
    line_ax.grid(axis="y", color="#cbd5dc", alpha=0.58, linewidth=0.65)
    line_ax.legend(loc="upper left", frameon=False, fontsize=11.5, ncol=2)

    cmap = ListedColormap(COLORS)
    cmap.set_bad(alpha=0)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)
    heat_ax.pcolormesh(np.arange(len(DAYS) + 1) - 0.5,
                       np.arange(len(YEARS) + 1) - 0.5,
                       np.ma.masked_equal(status, 0), cmap=cmap, norm=norm,
                       shading="flat", edgecolors="none", linewidth=0, zorder=1)
    # Year separators show zero-window years without white cell outlines.
    heat_ax.hlines(np.arange(0.5, len(YEARS) - 0.5), -0.5, len(DAYS) - 0.5,
                   colors="#8996a4", linewidth=0.40, alpha=0.32, zorder=2)
    month_days = ("07-01", "08-01", "09-01", "10-01", "11-01")
    month_pos = [DAYS.index(day) for day in month_days]
    for pos in month_pos[1:]:
        heat_ax.axvline(pos - 0.5, color="#8794a0", linewidth=0.55,
                        alpha=0.28, zorder=2)
    heat_ax.set_xlim(-0.5, len(DAYS) - 0.5)
    heat_ax.set_ylim(-0.5, len(YEARS) - 0.5)
    heat_ax.set_xticks(month_pos)
    heat_ax.set_xticklabels(["Jul", "Aug", "Sep", "Oct", "Nov"])
    heat_ax.set_yticks([year - YEARS[0] for year in range(2015, 2051, 5)])
    heat_ax.set_yticklabels([str(year) for year in range(2015, 2051, 5)])
    heat_ax.yaxis.tick_right()
    heat_ax.yaxis.set_label_position("right")
    heat_ax.set_ylabel("Year")
    heat_ax.set_xlabel("Day of year")
    for ax in (line_ax, heat_ax):
        ax.tick_params(axis="both", colors="#475569", length=3, labelsize=12)
        for spine in ax.spines.values():
            spine.set_color("#92a0ad")
            spine.set_linewidth(0.65)
    for ax, letter in ((line_ax, "a"), (heat_ax, "b")):
        ax.text(0, 1.025, letter, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=17, fontweight="bold", color="#253443")
    labels = ("Blocked in both (blank)", "Passable in both",
              "Added by downscaling", "Lost after downscaling")
    handles = [Patch(facecolor="none", edgecolor="#a4afb9", linewidth=0.7,
                     label=labels[0])]
    handles.extend(Patch(facecolor=color, edgecolor="none", label=label)
                   for color, label in zip(COLORS[1:], labels[1:]))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.055), fontsize=11.5, columnspacing=1.7,
               handlelength=1.6)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = "cmcc_threshold015_annual_line_rotated_daily_heatmap_large_type"
    fig.savefig(OUT / f"{stem}.png", dpi=300, transparent=True)
    fig.savefig(OUT / f"{stem}.svg", transparent=True)
    plt.close(fig)
    print(f"2015-2050 counts validated: before={before.sum()}, after={after.sum()}")
    print(f"saved={OUT / (stem + '.png')}")


if __name__ == "__main__":
    main()
