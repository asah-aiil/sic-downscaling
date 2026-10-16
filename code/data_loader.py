#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_loader.py —— 时间窗口 memmap 缓存版
支持直接从已有的十变量 X.npy/Y.npy 缓存抽取并重排目标通道，
也支持读取 build_temporal_memmap_cache.py 新生成的专用缓存。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

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


def _alias_candidates(feature_name: str):
    aliases = getattr(Config, "INPUT_VAR_ALIASES", {})
    configured = aliases.get(feature_name, [feature_name])
    if isinstance(configured, str):
        configured = [configured]
    return list(dict.fromkeys([feature_name, *configured]))


def requested_channel_specs():
    """返回 [(模型逻辑通道名, 缓存候选通道名), ...]。"""
    specs = []
    for off in get_time_offsets():
        tag = "d" if off == 0 else f"d{off}"
        for var in getattr(Config, "INPUT_VARS"):
            logical = f"{var}_{tag}"
            candidates = [f"{source}_{tag}" for source in _alias_candidates(var)]
            specs.append((logical, list(dict.fromkeys(candidates))))
    if bool(getattr(Config, "ADD_MASK_CHANNEL", True)):
        specs.append(("mask_d", ["mask_d", "sic_mask_d"]))
    return specs


def expected_channel_names():
    return [logical for logical, _ in requested_channel_specs()]


def resolve_cache_channels(meta: dict, cache_dir: Path):
    cache_names = list(meta.get("channel_names", []))
    if not cache_names:
        raise RuntimeError(f"{cache_dir / 'meta.json'} 缺少 channel_names，无法安全选择通道")

    indices = []
    selected_names = []
    for logical, candidates in requested_channel_specs():
        selected = next((name for name in candidates if name in cache_names), None)
        if selected is None:
            raise RuntimeError(
                f"缓存缺少模型通道 {logical!r}；可接受 {candidates}，"
                f"缓存实际通道为 {cache_names}"
            )
        indices.append(cache_names.index(selected))
        selected_names.append(selected)

    if len(set(indices)) != len(indices):
        raise RuntimeError(f"缓存通道映射发生重复：indices={indices}, names={selected_names}")

    allow_subset = bool(getattr(Config, "CACHE_ALLOW_CHANNEL_SUBSET", False))
    if not allow_subset and (
        len(indices) != len(cache_names) or indices != list(range(len(cache_names)))
    ):
        raise RuntimeError(
            "当前配置禁止从缓存抽取/重排通道；"
            f"需要 {expected_channel_names()}，缓存为 {cache_names}"
        )
    return indices, selected_names


def validate_cache_meta(meta: dict, cache_dir: Path):
    """阻止把同通道数、不同变量顺序的旧缓存误送进模型。"""
    if "N" not in meta or "x_file" not in meta or "y_file" not in meta:
        raise RuntimeError(
            f"{cache_dir / 'meta.json'} 不是当前训练器支持的 X.npy/Y.npy 缓存格式。\n"
            "请用 build_temporal_memmap_cache.py 重新生成缓存。"
        )

    expected_vars = list(getattr(Config, "INPUT_VARS"))
    expected_offsets = get_time_offsets()
    expected_mask = bool(getattr(Config, "ADD_MASK_CHANNEL", True))
    expected_c = int(getattr(Config, "INPUT_CHANNELS"))
    expected_years = list(getattr(Config, "YEARS"))
    expected_ice_thr = float(getattr(Config, "ICE_MASK_THR", 0.15))
    configured_ranges = getattr(Config, "FEATURE_NORM_RANGES", {})
    expected_ranges = {
        var: [float(x) for x in configured_ranges[var]]
        for var in expected_vars
        if var in configured_ranges
    }
    channel_indices, selected_names = resolve_cache_channels(meta, cache_dir)

    errors = []
    allow_legacy = bool(getattr(Config, "CACHE_ALLOW_LEGACY_META", False))
    if "complete" in meta and meta.get("complete") is not True:
        errors.append("complete: cache 构建未完成")
    elif "complete" not in meta and not allow_legacy:
        errors.append("complete: 旧缓存没有完整构建标记")
    if len(channel_indices) != expected_c:
        errors.append(
            f"selected channels: cache={len(channel_indices)} config INPUT_CHANNELS={expected_c}"
        )
    if list(meta.get("time_offsets", [])) != expected_offsets:
        errors.append(f"time_offsets: cache={meta.get('time_offsets')} config={expected_offsets}")
    if bool(meta.get("add_mask_channel", False)) != expected_mask:
        errors.append(
            f"add_mask_channel: cache={meta.get('add_mask_channel')} config={expected_mask}"
        )
    cache_names = list(meta.get("channel_names", []))
    if int(meta.get("C", -1)) != len(cache_names):
        errors.append(
            f"cache C={meta.get('C')}，但 channel_names 数量={len(cache_names)}"
        )
    if list(meta.get("years", [])) != expected_years:
        errors.append(f"years: cache={meta.get('years')} config={expected_years}")
    if not np.isclose(float(meta.get("ice_mask_thr", -1.0)), expected_ice_thr):
        errors.append(
            f"ice_mask_thr: cache={meta.get('ice_mask_thr')} config={expected_ice_thr}"
        )
    if meta.get("feature_norm_ranges"):
        for logical, selected in zip(expected_vars, selected_names):
            # mask 不在 expected_vars 中，因此 zip 只检查三个物理变量。
            source_feature = selected.rsplit("_", 1)[0]
            cached = meta["feature_norm_ranges"].get(
                logical, meta["feature_norm_ranges"].get(source_feature)
            )
            expected = expected_ranges.get(logical)
            if cached is None or [float(x) for x in cached] != expected:
                errors.append(
                    f"{logical} norm range: cache={cached} config={expected}"
                )
    elif not allow_legacy:
        errors.append("feature_norm_ranges: 旧缓存未记录归一化范围")

    if errors and bool(getattr(Config, "CACHE_STRICT_META", True)):
        details = "\n  - ".join(errors)
        raise RuntimeError(
            f"缓存与当前模型配置不一致，拒绝训练：{cache_dir}\n  - {details}\n"
            "请核对 config.py 后重新运行 build_temporal_memmap_cache.py。"
        )
    return channel_indices, selected_names


def get_fixed_year_splits():
    train_years = [int(y) for y in getattr(Config, "TRAIN_YEARS", [])]
    val_years = [int(y) for y in getattr(Config, "VAL_YEARS", [])]
    test_years = [int(y) for y in getattr(Config, "TEST_YEARS", [])]
    all_years = [int(y) for y in getattr(Config, "YEARS")]

    groups = {
        "train": train_years,
        "val": val_years,
        "test": test_years,
    }
    for name, years in groups.items():
        if not years:
            raise RuntimeError(f"Config.{name.upper()}_YEARS 不能为空")
        if len(years) != len(set(years)):
            raise RuntimeError(f"Config.{name.upper()}_YEARS 存在重复年份: {years}")

    train_set, val_set, test_set = set(train_years), set(val_years), set(test_years)
    overlaps = (train_set & val_set) | (train_set & test_set) | (val_set & test_set)
    if overlaps:
        raise RuntimeError(f"年份划分发生重叠: {sorted(overlaps)}")
    if train_set | val_set | test_set != set(all_years):
        raise RuntimeError(
            "年份划分没有完整覆盖 Config.YEARS: "
            f"config={all_years}, split={sorted(train_set | val_set | test_set)}"
        )
    return train_years, val_years, test_years


def _read_sample_years(N: int, cache_dir: Path) -> np.ndarray:
    index_path = cache_dir / "sample_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"按年份划分需要缓存索引文件: {index_path}\n"
            "该文件应包含 sample_id 和 year 两列。"
        )

    sample_years = np.full(N, -1, dtype=np.int32)
    seen = set()
    with open(index_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "sample_id" not in fields or "year" not in fields:
            raise RuntimeError(
                f"{index_path} 缺少 sample_id/year 列，实际列为 {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                sample_id = int(row["sample_id"])
                year = int(row["year"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{index_path}:{row_number} 的 sample_id/year 无效") from exc
            if not 0 <= sample_id < N:
                raise RuntimeError(
                    f"{index_path}:{row_number} sample_id={sample_id} 超出 [0, {N})"
                )
            if sample_id in seen:
                raise RuntimeError(f"{index_path}:{row_number} sample_id={sample_id} 重复")
            seen.add(sample_id)
            sample_years[sample_id] = year

    missing = np.flatnonzero(sample_years < 0)
    if len(missing) > 0:
        preview = missing[:10].tolist()
        raise RuntimeError(
            f"{index_path} 只覆盖 {len(seen)}/{N} 个样本，缺失示例={preview}"
        )
    return sample_years


def split_indices(N: int, seed: Optional[int] = None, cache_dir: Optional[Path] = None):
    split_mode = str(getattr(Config, "SPLIT_MODE", "sample_random")).lower()
    if split_mode == "year_fixed":
        train_years, val_years, test_years = get_fixed_year_splits()
        sample_years = _read_sample_years(N, cache_dir or get_cache_dir())
        idx_train = np.flatnonzero(np.isin(sample_years, train_years)).astype(np.int64)
        idx_val = np.flatnonzero(np.isin(sample_years, val_years)).astype(np.int64)
        idx_test = np.flatnonzero(np.isin(sample_years, test_years)).astype(np.int64)
        if len(idx_train) + len(idx_val) + len(idx_test) != N:
            raise RuntimeError(
                "按年份划分后的样本数量不等于缓存总数: "
                f"train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}, N={N}"
            )
        return idx_train, idx_val, idx_test

    if split_mode != "sample_random":
        raise ValueError(
            f"不支持的 SPLIT_MODE={split_mode!r}；可选 year_fixed/sample_random"
        )
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
        self.channel_indices, self.cache_channel_names = validate_cache_meta(
            self.meta, self.cache_dir
        )

        self.x_path = self.cache_dir / self.meta.get("x_file", "X.npy")
        self.y_path = self.cache_dir / self.meta.get("y_file", "Y.npy")
        self.N = int(self.meta["N"])
        self.channel_names = expected_channel_names()

        if indices is None:
            tr, va, te = split_indices(self.N, cache_dir=self.cache_dir)
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
        expected_x = tuple(self.meta.get("x_shape", ()))
        expected_y = tuple(self.meta.get("y_shape", ()))
        if expected_x and tuple(self._X.shape) != expected_x:
            raise RuntimeError(f"X.npy shape={self._X.shape}，meta 记录为 {expected_x}")
        if expected_y and tuple(self._Y.shape) != expected_y:
            raise RuntimeError(f"Y.npy shape={self._Y.shape}，meta 记录为 {expected_y}")

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        self._ensure_open()
        real_idx = int(self.indices[idx])
        # copy=True：避免 torch.from_numpy 读只读 memmap 时的 non-writable warning
        x = np.array(self._X[real_idx, self.channel_indices, :, :], copy=True)
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
    validate_cache_meta(meta, cache_dir)
    N = int(meta["N"])
    tr, va, _ = split_indices(N, cache_dir=cache_dir)

    train_ds = TemporalMemmapDataset(split="train", indices=tr)
    val_ds = TemporalMemmapDataset(split="val", indices=va)

    bs = int(getattr(Config, "BATCH_SIZE", 8))
    nw = int(getattr(Config, "NUM_WORKERS", 4))
    train_shuffle = bool(getattr(Config, "CACHE_TRAIN_SHUFFLE", True))

    print("[TemporalMemmapDataset] 使用预生成 memmap 缓存，不再训练时读取 NetCDF")
    print(f"  cache_dir      = {cache_dir}")
    print(f"  split sizes    = train {len(train_ds)}, val {len(val_ds)}")
    print(f"  split mode     = {getattr(Config, 'SPLIT_MODE', 'sample_random')}")
    if str(getattr(Config, "SPLIT_MODE", "sample_random")).lower() == "year_fixed":
        train_years, val_years, test_years = get_fixed_year_splits()
        print(f"  train years    = {train_years}")
        print(f"  val years      = {val_years}")
        print(f"  test years     = {test_years}")
    print(f"  dtype          = {meta.get('dtype')}, return_float32={getattr(Config, 'CACHE_RETURN_FLOAT32', False)}")
    print(f"  train shuffle  = {train_shuffle}")
    print(f"  cache channels = {train_ds.cache_channel_names}")
    print(f"  model channels = {train_ds.channel_names}")
    print(f"  channel index  = {train_ds.channel_indices}")
    if "complete" not in meta or "feature_norm_ranges" not in meta:
        print("  legacy meta    = True（已按备份构建脚本兼容）")

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
    validate_cache_meta(meta, cache_dir)
    N = int(meta["N"])
    _, _, te = split_indices(N, cache_dir=cache_dir)

    test_ds = TemporalMemmapDataset(split="test", indices=te)
    bs = int(getattr(Config, "BATCH_SIZE", 8))
    nw = int(getattr(Config, "NUM_WORKERS", 4))
    return DataLoader(test_ds, batch_size=bs, shuffle=False, **_loader_kwargs(nw))
