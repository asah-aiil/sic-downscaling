#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""10变量降尺度模型：东北航道 Bremen 冰区内五分位 SIC 混淆矩阵。

空间评价范围为：静态东北航道海域掩膜 ∩ 当日 Bremen 真值 SIC >= 0.15。
以该范围内所有真值 SIC 的全样本分位数划分 5 个等级，预测 SIC 使用完全相同的
分界值分类。每个真值等级约占 20%，避免开阔水域或高 SIC 类别压倒混淆矩阵。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_envgated_full10_arctic_nep_2024_2025 import (
    DEFAULT_CACHE, DEFAULT_CHECKPOINT, DEFAULT_MASK_FILE, LOGICAL_VARS,
    CacheDataset, channel_index, configure_and_build_model, load_masks,
    load_meta, read_records, safe_torch_load,
)


ICE_THRESHOLD = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="东北航道 Bremen 冰区内五分位 SIC 混淆矩阵")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--years", type=int, nargs="+", default=[2024, 2025])
    p.add_argument("--mask-file", default=DEFAULT_MASK_FILE)
    p.add_argument("--mask-var", default="mask_all_shipping_seas")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--quantile-hist-bins", type=int, default=10000,
                   help="真值分位数计算的 SIC 直方图分箱数；10000 对应约 1e-4 精度")
    p.add_argument("--plot-bins", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out-dir", default="local-path/nep_icezone_quantile5_full10_2024_2025")
    a = p.parse_args()
    if a.batch_size < 1 or a.num_workers < 0 or a.max_samples < 0 or a.quantile_hist_bins < 100:
        p.error("batch-size>=1，num-workers/max-samples>=0，quantile-hist-bins>=100")
    return a


def quantile_edges_from_target(y: np.ndarray, records: list[dict], mask: np.ndarray, hist_bins: int) -> tuple[np.ndarray, int]:
    """仅扫描 target memmap，避免为分位数把全部冰区像元读入内存。"""
    hist = np.zeros(hist_bins, dtype=np.int64)
    n = 0
    for i, record in enumerate(records, 1):
        target = np.asarray(y[int(record["sample_id"]), 0], dtype=np.float32)
        valid = mask & np.isfinite(target) & (target >= ICE_THRESHOLD)
        values = np.clip(target[valid], ICE_THRESHOLD, 1.0)
        h, _ = np.histogram(values, bins=hist_bins, range=(ICE_THRESHOLD, 1.0))
        hist += h.astype(np.int64); n += int(values.size)
        if i % 100 == 0:
            print(f"quantile scan {i}/{len(records)}", flush=True)
    if n == 0:
        raise RuntimeError("东北航道 Bremen 冰区内没有 SIC >= 0.15 的有效像元")
    cdf = np.cumsum(hist)
    width = (1.0 - ICE_THRESHOLD) / hist_bins
    internal = []
    for q in (0.2, 0.4, 0.6, 0.8):
        idx = int(np.searchsorted(cdf, q * n, side="left"))
        internal.append(ICE_THRESHOLD + min(idx + 1, hist_bins) * width)
    edges = np.array([ICE_THRESHOLD, *internal, 1.000001], dtype=np.float32)
    # 对离散 SIC 值可能出现相邻分位数相等；np.digitize 可安全处理，仍保留原始边界供报告。
    return edges, n


def classify(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(values, edges[1:-1], right=False).astype(np.int64)


def update(cm: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> None:
    k = cm.shape[0]
    cm += np.bincount(truth * k + pred, minlength=k * k).reshape(k, k)


def row_normalize(cm: np.ndarray) -> np.ndarray:
    return np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0)


def labels_from_edges(edges: np.ndarray) -> list[str]:
    return [f"Q{i + 1}\n{edges[i]:.3f}–{min(edges[i + 1], 1.0):.3f}" for i in range(5)]


def save_csv(path: Path, cm: np.ndarray, labels: list[str], normalized: bool) -> None:
    values = row_normalize(cm) if normalized else cm
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f); w.writerow(["truth\\prediction", *labels])
        for label, row in zip(labels, values): w.writerow([label, *row.tolist()])


def plot_cm(cm: np.ndarray, labels: list[str], path: Path, normalized: bool) -> None:
    data = row_normalize(cm) if normalized else cm.astype(float)
    fig, ax = plt.subplots(figsize=(8.0, 6.6), constrained_layout=True)
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1 if normalized else max(1, float(data.max())))
    fig.colorbar(im, ax=ax, label="真值等级内比例" if normalized else "像元数")
    ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels)
    ax.set_xlabel("10变量模型预测 SIC 等级"); ax.set_ylabel("Bremen 真值 SIC 等级")
    ax.set_title("东北航道 Bremen 冰区：五分位 SIC 混淆矩阵" + "（行归一化）" if normalized else "东北航道 Bremen 冰区：五分位 SIC 混淆矩阵")
    for i in range(5):
        for j in range(5):
            text = f"{data[i,j]:.3f}" if normalized else f"{int(data[i,j]):,}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if data[i,j] > data.max() * .55 else "black")
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def plot_log_distribution(h2: np.ndarray, ht: np.ndarray, hp: np.ndarray, edges: np.ndarray, path: Path) -> None:
    bins = len(ht); centers = (np.arange(bins) + .5) / bins
    positive = h2[h2 > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)
    image = axes[0].imshow(h2.T, origin="lower", extent=(0, 1, 0, 1), aspect="equal", cmap="YlGnBu",
                           norm=LogNorm(vmin=max(1, float(positive.min())) if positive.size else 1, vmax=max(1, float(h2.max()))))
    axes[0].plot([0,1],[0,1],"--",color="#d95f02",linewidth=1.1)
    for edge in edges[:-1]:
        axes[0].axvline(edge, color="0.35", linewidth=.55); axes[0].axhline(edge, color="0.35", linewidth=.55)
    axes[0].set(xlabel="Bremen 真值 SIC", ylabel="10变量预测 SIC", title="冰区对应像元二维频数（log）")
    fig.colorbar(image, ax=axes[0], label="像元数（log）")
    axes[1].plot(centers, ht, label="Bremen 真值", color="#2b8cbe")
    axes[1].plot(centers, hp, label="10变量降尺度", color="#238b45")
    for edge in edges[:-1]: axes[1].axvline(edge, color="0.35", linewidth=.6)
    axes[1].set_yscale("log"); axes[1].set(xlim=(ICE_THRESHOLD,1), xlabel="SIC", ylabel="像元数（log）", title="冰区 SIC 分布（log）")
    axes[1].legend()
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    a = parse_args()
    cache = Path(a.cache).expanduser().resolve(); out = Path(a.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    meta = load_meta(cache); names = list(meta.get("channel_names", []))
    channels = [channel_index(names, key) for key in LOGICAL_VARS] + [channel_index(names, "mask")]
    x = np.load(cache / meta.get("x_file", "X.npy"), mmap_mode="r")
    y = np.load(cache / meta.get("y_file", "Y.npy"), mmap_mode="r")
    records = read_records(cache, len(x), a.years)
    if a.max_samples: records = records[:a.max_samples]
    if not records: raise RuntimeError("请求年份在缓存中没有样本")
    _, nep = load_masks(Path(a.mask_file).expanduser().resolve(), "mask_arctic_ocean", a.mask_var, tuple(y.shape[-2:]))
    print(f"quantile target domain: NEP static mask & Bremen SIC >= {ICE_THRESHOLD}", flush=True)
    edges, ice_pixels = quantile_edges_from_target(y, records, nep, a.quantile_hist_bins)
    labels = labels_from_edges(edges)
    print(f"samples={len(records)}, ice pixels={ice_pixels}, quantile edges={edges.tolist()}", flush=True)
    with (out / "quantile_edges.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.writer(f); w.writerow(["class","lower_sic","upper_sic","definition"])
        for i in range(5): w.writerow([f"Q{i+1}",float(edges[i]),float(min(edges[i+1],1.0)),"Bremen target quantile bin within NEP ice zone"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = safe_torch_load(Path(a.checkpoint).expanduser().resolve())
    model, _ = configure_and_build_model(checkpoint, device)
    use_amp = device.type == "cuda" and not a.no_amp
    loader = DataLoader(CacheDataset(cache, meta, records, channels), batch_size=a.batch_size, shuffle=False,
                        num_workers=a.num_workers, pin_memory=device.type=="cuda", persistent_workers=a.num_workers>0)
    cm = np.zeros((5,5), dtype=np.int64); h2=np.zeros((a.plot_bins,a.plot_bins),dtype=np.int64); ht=np.zeros(a.plot_bins,dtype=np.int64); hp=np.zeros(a.plot_bins,dtype=np.int64)
    print(f"device={device}, AMP={use_amp}", flush=True)
    with torch.inference_mode():
        for bi,(inputs,target,_,_) in enumerate(loader,1):
            inputs=inputs.to(device,non_blocking=True)
            if not use_amp: inputs=inputs.float()
            target=target.to(device,non_blocking=True).float()
            with torch.amp.autocast(device_type=device.type,enabled=use_amp): output=model(inputs)
            pred=output.float().clamp(0,1).cpu().numpy()[:,0]; truth=target.cpu().numpy()[:,0]
            for p,t in zip(pred,truth):
                valid=nep & np.isfinite(t) & np.isfinite(p) & (t>=ICE_THRESHOLD)
                tv=t[valid].astype(np.float32,copy=False); pv=p[valid].astype(np.float32,copy=False)
                update(cm,classify(tv,edges),classify(pv,edges))
                a2,_,_=np.histogram2d(tv,pv,bins=a.plot_bins,range=((0,1),(0,1))); at,_=np.histogram(tv,bins=a.plot_bins,range=(0,1)); ap,_=np.histogram(pv,bins=a.plot_bins,range=(0,1))
                h2 += a2.astype(np.int64); ht += at.astype(np.int64); hp += ap.astype(np.int64)
            if bi==1 or bi%a.progress_every==0: print(f"{bi}/{len(loader)} batches",flush=True)
    save_csv(out / "quantile5_confusion_counts.csv",cm,labels,False); save_csv(out / "quantile5_confusion_row_normalized.csv",cm,labels,True)
    plot_cm(cm,labels,out / "quantile5_confusion_counts.png",False); plot_cm(cm,labels,out / "quantile5_confusion_row_normalized.png",True)
    plot_log_distribution(h2,ht,hp,edges,out / "icezone_sic_truth_prediction_log_distribution.png")
    total=int(cm.sum()); accuracy=float(np.trace(cm)/total) if total else float("nan")
    with (out / "summary.txt").open("w",encoding="utf-8") as f:
        f.write("NEP Bremen ice-zone quantile-5 evaluation\\n")
        f.write(f"domain=mask_all_shipping_seas AND Bremen_target_SIC>={ICE_THRESHOLD}\\n")
        f.write(f"years={a.years}; samples={len(records)}; pixels={total}\\n")
        f.write(f"quantile_edges={edges.tolist()}\\n")
        f.write(f"five_class_accuracy={accuracy:.6f}\\n")
    print(f"done: pixels={total}, five_class_accuracy={accuracy:.6f}, out={out}",flush=True)


if __name__ == "__main__": main()
