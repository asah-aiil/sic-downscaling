#!/usr/bin/env python3
"""Render the established 66 routes over predicted SIC on 2048-07-25."""
import shutil

import rerender_cmcc_multiyear_full_routes as renderer
from plot_cmcc_multiyear_aug_routes import ROOT

BASE_DIR = ROOT / 'fig/cmcc_66_routes_ice_edge_on_2050-08-10'
OUTPUT_DIR = ROOT / 'fig/cmcc_66_routes_2048_jul25_graticule'
BACKGROUND_DATE = '2048-07-25'


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ('selected_routes.csv', 'route_manifest.csv'):
        shutil.copy2(BASE_DIR / filename, OUTPUT_DIR / filename)
    renderer.OUT_DIR = OUTPUT_DIR
    renderer.BACKGROUND_FILE = (ROOT / 'data_cmcc/predict_pr_tos_sic_mask_2048' /
        'sic_pred_6p25km_cmcc_pr_tos_sic_mask_2048.nc')
    renderer.BACKGROUND_VARIABLE = 'sic_pred'
    renderer.BACKGROUND_DATE = BACKGROUND_DATE
    renderer.COLORBAR_LABEL = f'Downscaled predicted sea-ice concentration ({BACKGROUND_DATE})'
    renderer.OUTPUT_STEM = 'cmcc_66_routes_on_2048-07-25_graticule'
    renderer.VIEW = (-2_080_000.0, 3_230_000.0, -2_400_000.0, 600_000.0)
    renderer.CLEAN_MAP = True
    renderer.DRAW_TARGET_GRATICULE = True
    renderer.OCEAN_COLOR = '#e8f6fa'
    renderer.LAND_COLOR = '#fbf9f3'
    renderer.LAND_EDGE_COLOR = '#55514a'
    renderer.LAND_EDGE_WIDTH = 0.5
    renderer.SIC_CMAP_STOPS = [
        '#e8f6fa', '#45bdd4', '#167ab5', '#253c91', '#6d3a91',
    ]
    renderer.EXPECTED_COUNTS = {.15: 43, .30: 15, .70: 8}
    renderer.LEGEND_LABELS = {
        .15: 'SIC ≤ 0.15 · 43 routes (multiple years)',
        .30: 'SIC ≤ 0.30 · 15 routes (multiple years)',
        .70: 'SIC ≤ 0.70 · 8 routes (multiple years)',
    }
    renderer.LEGEND_FIGSIZE = (7.2, 1.6)
    renderer.main()
    (OUTPUT_DIR / 'figure_metadata.txt').write_text(
        'Background: downscaled CMCC predicted SIC on 2048-07-25.\n'
        'All 66 routes are unchanged from the previous figure.\n'
        'Zero-SIC ocean and ocean background: #e8f6fa.\n'
        'Land fill: #fbf9f3; coastline: #55514a at 0.5 pt.\n'
        'Graticule: 20E-180E every 10 degrees, 70N-85N every 5 degrees; '
        'gray short dashes, no labels.\n'
        'Higher-SIC palette stops and route styles are unchanged.\n'
        'Multiyear routes use SIC from their own year/date, not this single '
        'display background.\n',
        encoding='utf-8',
    )


if __name__ == '__main__':
    main()
