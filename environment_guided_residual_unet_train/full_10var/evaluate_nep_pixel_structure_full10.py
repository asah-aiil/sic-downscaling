#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""环境引导 10 变量模型在东北航道的像元级空间一致性诊断。

使用未参与训练的 2024--2025 Bremen target 缓存。只统计静态东北航道海域
掩膜内、预测和真值一一对应的像元：
  * SIC=0.15 二分类混淆矩阵（开放水域 / 冰区）；
  * 0.15、0.30、0.70 航行阈值对应的四等级混淆矩阵；
  * 真值--预测二维对数频数图和一维对数频数分布。

这不是 A* 通航日统计；它评价相同航道像元上的 SIC 空间类别是否与 Bremen
真值一致，因此 FN/FP 能直接反映冰缘或局地阻塞位置的错分。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_envgated_full10_arctic_nep_2024_2025 import (
    DEFAULT_CACHE,
    DEFAULT_CHECKPOINT,
    DEFAULT_MASK_FILE,
    LOGICAL_VARS,
    CacheDataset,
    channel_index,
    configure_and_build_model,
    load_masks,
    load_meta,
    read_records,
    safe_torch_load,
)


BIN_EDGES = np.array([0.0, 0.15, 0.30, 0.70, 1.000001], dtype=np.float32)
BIN_LABELS = ["0–0.15", "0.15–0.30", "0.30–0.70", "0.70–1.00"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10变量环境引导模型：东北航道像元空间一致性诊断")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    p.add_argument("--mask-file", default=DEFAULT_MASK_FILE)
    p.add_argument("--mask-var", default="mask_all_shipping_seas")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bins", type=int, default=100, help="连续 SIC 分布的每轴分箱数")
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=0, help="0 表示全部，仅用于冒烟测试")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--out-dir",
        default="local-path/nep_pixel_structure_full10_2024_2025",
    )
    a = p.parse_args()
    if a.batch_size < 1 or a.num_workers < 0 or a.bins < 10 or a.max_samples < 0:
        p.error("batch-size>=1，num-workers/max-samples>=0，bins>=10")
    return a


def class_index(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, BIN_EDGES[1:-1], right=False).astype(np.int64)


def update_cm(cm: np.ndarray, truth_class: np.ndarray, pred_class: np.ndarray) -> None:
    k = cm.shape[0]
    flat = truth_class * k + pred_class
    cm += np.bincount(flat, minlength=k * k).reshape(k, k)


def normalise_rows(cm: np.ndarray) -> np.ndarray:
    return np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0)


def save_matrix(path: Path, cm: np.ndarray, labels: list[str], normalised: bool) -> None:
    values = normalise_rows(cm) if normalised else cm
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["truth\\prediction", *labels])
        for label, row in zip(labels, values):
            w.writerow([label, *row.tolist()])


def plot_cm(cm: np.ndarray, labels: list[str], title: str, path: Path, normalised: bool) -> None:
    data = normalise_rows(cm) if normalised else cm.astype(float)
    fig, ax = plt.subplots(figsize=(6.7, 5.5), constrained_layout=True)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1 if normalised else max(1.0, float(data.max())))
    fig.colorbar(im, ax=ax, label="行归一化比例" if normalised else "像元数")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("预测 SIC 类别")
    ax.set_ylabel("Bremen 真值 SIC 类别")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.3f}" if normalised else f"{int(data[i, j]):,}"
            color = "white" if data[i, j] > data.max() * 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def binary_metrics(cm: np.ndarray) -> dict[str, float | int]:
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    total = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "TN_open_water": tn, "FP_false_ice": fp, "FN_missed_ice": fn, "TP_ice": tp,
        "pixel_count": total,
        "accuracy": (tn + tp) / total if total else float("nan"),
        "ice_precision": precision,
        "ice_recall": recall,
        "ice_f1": 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and precision + recall else float("nan"),
        "ice_iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
    }


def plot_distributions(hist2d: np.ndarray, hist_truth: np.ndarray, hist_pred: np.ndarray, bins: int, path: Path) -> None:
    centers = (np.arange(bins) + 0.5) / bins
    positive = hist2d[hist2d > 0]
    norm = LogNorm(vmin=max(1, float(positive.min())) if positive.size else 1, vmax=max(1, float(hist2d.max())))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    image = axes[0].imshow(hist2d.T, origin="lower", extent=(0, 1, 0, 1), cmap="YlGnBu", norm=norm, aspect="equal")
    axes[0].plot([0, 1], [0, 1], "--", color="#d95f02", linewidth=1.1, label="y=x")
    for value in (0.15, 0.30, 0.70):
        axes[0].axvline(value, color="0.35", linewidth=0.6, alpha=0.7)
        axes[0].axhline(value, color="0.35", linewidth=0.6, alpha=0.7)
    axes[0].set(xlabel="Bremen 真值 SIC", ylabel="10变量模型预测 SIC", title="东北航道对应像元二维频数（对数色标）")
    axes[0].legend(loc="upper left", frameon=True)
    fig.colorbar(image, ax=axes[0], label="像元数（log）")
    axes[1].plot(centers, hist_truth, color="#2b8cbe", linewidth=1.6, label="Bremen 真值")
    axes[1].plot(centers, hist_pred, color="#238b45", linewidth=1.6, label="10变量降尺度")
    for value in (0.15, 0.30, 0.70):
        axes[1].axvline(value, color="0.35", linewidth=0.7, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set(xlim=(0, 1), xlabel="SIC", ylabel="像元数（log）", title="东北航道 SIC 一维频数分布（对数纵轴）")
    axes[1].legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    a = parse_args()
    checkpoint_path = Path(a.checkpoint).expanduser().resolve()
    cache = Path(a.cache).expanduser().resolve()
    out = Path(a.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(cache)
    names = list(meta.get("channel_names", []))
    channels = [channel_index(names, key) for key in LOGICAL_VARS]
    channels.append(channel_index(names, "mask"))
    x = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    records = read_records(cache, len(x), a.years)
    if a.max_samples:
        records = records[:a.max_samples]
    if not records or y.shape[1] != 1:
        raise RuntimeError(f"缓存记录或 Y 形状异常：records={len(records)}, Y={y.shape}")
    _, nep = load_masks(Path(a.mask_file).expanduser().resolve(), "mask_arctic_ocean", a.mask_var, tuple(y.shape[-2:]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = safe_torch_load(checkpoint_path)
    model, _ = configure_and_build_model(checkpoint, device)
    use_amp = device.type == "cuda" and not a.no_amp
    loader = DataLoader(CacheDataset(cache, meta, records, channels), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=a.num_workers > 0)
    binary = np.zeros((2, 2), dtype=np.int64)
    multi = np.zeros((4, 4), dtype=np.int64)
    hist2d = np.zeros((a.bins, a.bins), dtype=np.int64)
    hist_truth = np.zeros(a.bins, dtype=np.int64)
    hist_pred = np.zeros(a.bins, dtype=np.int64)
    print(f"checkpoint={checkpoint_path}\ncache={cache}\nsamples={len(records)}, NEP pixels/day={int(nep.sum())}\ndevice={device}, AMP={use_amp}", flush=True)
    with torch.inference_mode():
        for bi, (inputs, target, _, _) in enumerate(loader, 1):
            inputs = inputs.to(device, non_blocking=True)
            if not use_amp:
                inputs = inputs.float()
            target = target.to(device, non_blocking=True).float()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(inputs)
            prediction = output.float().clamp(0.0, 1.0).cpu().numpy()[:, 0]
            truth = target.cpu().numpy()[:, 0]
            for pred_one, truth_one in zip(prediction, truth):
                valid = nep & np.isfinite(pred_one) & np.isfinite(truth_one)
                p = pred_one[valid].astype(np.float32, copy=False)
                t = truth_one[valid].astype(np.float32, copy=False)
                update_cm(binary, (t >= 0.15).astype(np.int64), (p >= 0.15).astype(np.int64))
                update_cm(multi, class_index(t), class_index(p))
                h2, _, _ = np.histogram2d(t, p, bins=a.bins, range=((0, 1), (0, 1)))
                ht, _ = np.histogram(t, bins=a.bins, range=(0, 1))
                hp, _ = np.histogram(p, bins=a.bins, range=(0, 1))
                hist2d += h2.astype(np.int64); hist_truth += ht.astype(np.int64); hist_pred += hp.astype(np.int64)
            if bi == 1 or bi % a.progress_every == 0:
                print(f"{bi}/{len(loader)} batches", flush=True)

    save_matrix(out / "binary_confusion_counts.csv", binary, ["open <0.15", "ice ≥0.15"], False)
    save_matrix(out / "binary_confusion_row_normalized.csv", binary, ["open <0.15", "ice ≥0.15"], True)
    save_matrix(out / "multiclass_confusion_counts.csv", multi, BIN_LABELS, False)
    save_matrix(out / "multiclass_confusion_row_normalized.csv", multi, BIN_LABELS, True)
    plot_cm(binary, ["open <0.15", "ice ≥0.15"], "东北航道二分类混淆矩阵（10变量降尺度）", out / "binary_confusion_counts.png", False)
    plot_cm(binary, ["open <0.15", "ice ≥0.15"], "东北航道二分类混淆矩阵（按 Bremen 真值行归一化）", out / "binary_confusion_row_normalized.png", True)
    plot_cm(multi, BIN_LABELS, "东北航道 SIC 四等级混淆矩阵（10变量降尺度）", out / "multiclass_confusion_counts.png", False)
    plot_cm(multi, BIN_LABELS, "东北航道 SIC 四等级混淆矩阵（按 Bremen 真值行归一化）", out / "multiclass_confusion_row_normalized.png", True)
    plot_distributions(hist2d, hist_truth, hist_pred, a.bins, out / "sic_truth_prediction_log_distribution.png")
    metrics = binary_metrics(binary)
    metrics.update({"samples": len(records), "nep_pixels_per_sample": int(nep.sum()), "checkpoint": str(checkpoint_path), "cache": str(cache), "years": ",".join(map(str, a.years))})
    with (out / "binary_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics)); w.writeheader(); w.writerow(metrics)
    print("\n=== Done ===", flush=True)
    for key, value in metrics.items(): print(f"{key}={value}", flush=True)
    print(f"out_dir={out}", flush=True)


if __name__ == "__main__":
    main()
