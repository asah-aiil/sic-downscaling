#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py —— 时间窗口 memmap 缓存版
前提：先运行 build_temporal_memmap_cache.py 生成 X.npy / Y.npy / meta.json。
优点：训练时不再读取 NetCDF，只从连续 memmap 缓存读取样本。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


def get_time_offsets():
    return list(getattr(Config, "TIME_OFFSETS", getattr(Config, "TEMPORAL_OFFSETS", [-2, -1, 0])))


def get_cache_dir() -> Path:
    dtype_name = getattr(Config, "TEMPORAL_CACHE_DTYPE", "float16")
    default = f"/home/ubuntu/sic_dsc/data/cache_temporal_T{len(get_time_offsets())}_{dtype_name}"
    return Path(getattr(Config, "TEMPORAL_CACHE_DIR", default))


def split_indices(N: int, seed: Optional[int] = None):
    if seed is None:
        seed = int(getattr(Config, "SEED", 42))
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    train_ratio = float(getattr(Config, "TRAIN_RATIO", 0.6))
    val_ratio = float(getattr(Config, "VAL_RATIO", 0.2))
    test_ratio = float(getattr(Config, "TEST_RATIO", 0.2))
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO 必须等于 1.0")

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]
    return idx_train, idx_val, idx_test


class TemporalMemmapDataset(Dataset):
    def __init__(self, split: str = "train", indices: Optional[np.ndarray] = None):
        super().__init__()
        self.cache_dir = get_cache_dir()
        self.meta_path = self.cache_dir / "meta.json"
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"找不到缓存 meta.json: {self.meta_path}\n"
                f"请先运行: python build_temporal_memmap_cache.py"
            )
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.x_path = self.cache_dir / self.meta.get("x_file", "X.npy")
        self.y_path = self.cache_dir / self.meta.get("y_file", "Y.npy")
        self.N = int(self.meta["N"])
        self.channel_names = list(self.meta.get("channel_names", []))

        if indices is None:
            tr, va, te = split_indices(self.N)
            if split == "train":
                indices = tr
            elif split == "val":
                indices = va
            elif split == "test":
                indices = te
            elif split == "all":
                indices = np.arange(self.N)
            else:
                raise ValueError(f"unknown split: {split}")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.split = split

        self.return_float32 = bool(getattr(Config, "CACHE_RETURN_FLOAT32", False))

        # 每个 worker 内懒加载，避免父进程打开的 memmap 被多进程继承造成奇怪问题
        self._X = None
        self._Y = None

    def _ensure_open(self):
        if self._X is None:
            self._X = np.load(self.x_path, mmap_mode="r")
        if self._Y is None:
            self._Y = np.load(self.y_path, mmap_mode="r")

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        self._ensure_open()
        real_idx = int(self.indices[idx])
        # copy=True：避免 torch.from_numpy 读只读 memmap 时的 non-writable warning
        x = np.array(self._X[real_idx], copy=True)
        y = np.array(self._Y[real_idx], copy=True)
        if self.return_float32:
            x = x.astype("float32", copy=False)
            y = y.astype("float32", copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)


def _loader_kwargs(nw: int):
    kwargs = {
        "num_workers": nw,
        "pin_memory": bool(getattr(Config, "PIN_MEMORY", True)),
        "drop_last": False,
    }
    if nw > 0:
        kwargs["persistent_workers"] = bool(getattr(Config, "PERSISTENT_WORKERS", True))
        kwargs["prefetch_factor"] = int(getattr(Config, "PREFETCH_FACTOR", 2))
    return kwargs


def create_dataloaders():
    # 先构建一次全量索引，保持与旧 split 逻辑一致
    cache_dir = get_cache_dir()
    with open(cache_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    N = int(meta["N"])
    tr, va, _ = split_indices(N)

    train_ds = TemporalMemmapDataset(split="train", indices=tr)
    val_ds = TemporalMemmapDataset(split="val", indices=va)

    bs = int(getattr(Config, "BATCH_SIZE", 8))
    nw = int(getattr(Config, "NUM_WORKERS", 4))
    train_shuffle = bool(getattr(Config, "CACHE_TRAIN_SHUFFLE", True))

    print("[TemporalMemmapDataset] 使用预生成 memmap 缓存，不再训练时读取 NetCDF")
    print(f"  cache_dir      = {cache_dir}")
    print(f"  split sizes    = train {len(train_ds)}, val {len(val_ds)}")
    print(f"  dtype          = {meta.get('dtype')}, return_float32={getattr(Config, 'CACHE_RETURN_FLOAT32', False)}")
    print(f"  train shuffle  = {train_shuffle}")
    print(f"  channel first  = {meta.get('channel_names', [])[:12]}")
    print(f"  channel last   = {meta.get('channel_names', [])[-5:]}")

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=train_shuffle,
        **_loader_kwargs(nw),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        **_loader_kwargs(nw),
    )
    return train_loader, val_loader


def create_test_loader():
    cache_dir = get_cache_dir()
    with open(cache_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    N = int(meta["N"])
    _, _, te = split_indices(N)

    test_ds = TemporalMemmapDataset(split="test", indices=te)
    bs = int(getattr(Config, "BATCH_SIZE", 8))
    nw = int(getattr(Config, "NUM_WORKERS", 4))
    return DataLoader(test_ds, batch_size=bs, shuffle=False, **_loader_kwargs(nw))
