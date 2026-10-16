"""Register the user's Times New Roman font files for reproducible CMCC figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


FONT_DIR = Path("/home/ubuntu/.local/share/fonts/dsc_times")
FONT_FILES = ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")


def use_times_new_roman() -> None:
    missing = [name for name in FONT_FILES if not (FONT_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Times New Roman font files missing in {FONT_DIR}: {missing}")
    for name in FONT_FILES:
        font_manager.fontManager.addfont(str(FONT_DIR / name))
    regular = Path(font_manager.findfont(
        font_manager.FontProperties(family="Times New Roman"),
        fallback_to_default=False))
    bold = Path(font_manager.findfont(
        font_manager.FontProperties(family="Times New Roman", weight="bold"),
        fallback_to_default=False))
    if regular.name.lower() != "times.ttf" or bold.name.lower() != "timesbd.ttf":
        raise RuntimeError(f"Wrong font substitution: regular={regular}, bold={bold}")
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
    })
    print(f"font={regular}; bold={bold}", flush=True)
