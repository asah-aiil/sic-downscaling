#!/usr/bin/env python3
"""Export the exact, text-free diverging bars used by the 2024-10-12 maps."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


OUTPUT = Path("local-path/fig/2024-10-12_rect_landpale")
COLORS = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]
LIMITS = {
    "de_signed_bias": 0.190732,
    "f_absolute_error_improvement": 0.190732,
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("sic_bias_blue_white_red", COLORS)
    for name, limit in LIMITS.items():
        edges = np.linspace(-limit, limit, 513)
        xx, yy = np.meshgrid(edges, [0.0, 1.0])
        values = ((edges[:-1] + edges[1:]) / 2.0)[None, :]
        fig = plt.figure(figsize=(8, 0.45), dpi=600, facecolor="none")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.pcolormesh(xx, yy, values, cmap=cmap,
                      norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                      shading="flat", edgecolors="none", antialiased=False)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        for suffix in ("png", "svg"):
            path = OUTPUT / f"colorbar_{name}_transparent.{suffix}"
            fig.savefig(path, transparent=True, facecolor="none", edgecolor="none",
                        dpi=600, bbox_inches=None, pad_inches=0)
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
