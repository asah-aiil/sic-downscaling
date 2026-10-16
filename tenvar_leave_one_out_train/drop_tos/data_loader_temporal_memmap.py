#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容旧文件名；实际实现统一放在 data_loader.py。"""

from data_loader import (  # noqa: F401
    TemporalMemmapDataset,
    create_dataloaders,
    create_test_loader,
    expected_channel_names,
    get_cache_dir,
    get_time_offsets,
    split_indices,
    validate_cache_meta,
)

__all__ = [
    "TemporalMemmapDataset",
    "create_dataloaders",
    "create_test_loader",
    "expected_channel_names",
    "get_cache_dir",
    "get_time_offsets",
    "split_indices",
    "validate_cache_meta",
]
