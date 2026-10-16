# train_utils.py
# 稳定训练 + 结构化日志（CSV/JSONL）版
# - 无 warmup
# - 无冻结 loss 权重
# - 验证计算 SSIM/MS-SSIM：强制 float32 + autocast 关闭，避免 half/float mismatch
# - 每个 epoch 记录到 train_metrics.csv 便于画图

from __future__ import annotations

import os
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config


# -----------------------------
# AMP helpers
# -----------------------------
def _autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.amp.autocast("cpu", enabled=False)


def _make_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.amp.GradScaler(enabled=False)


# -----------------------------
# Metric helpers (float32, no autocast)
# -----------------------------
@torch.no_grad()
def _rmse01(pred01: torch.Tensor, target01: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(pred01, target01)
    return float(torch.sqrt(mse + eps).item())


@torch.no_grad()
def _ssim_msssim01(pred01: torch.Tensor, target01: torch.Tensor, win_size: int, win_sigma: float) -> Tuple[float, float]:
    """
    pred01/target01: float32, in [0,1]
    Returns (ssim, msssim). If pytorch_msssim unavailable -> (nan, nan)
    """
    try:
        from pytorch_msssim import ssim as _ssim_fn, ms_ssim as _msssim_fn  # type: ignore
    except Exception:
        return float("nan"), float("nan")

    s = float(_ssim_fn(pred01, target01, data_range=1.0, size_average=True, win_size=int(win_size), win_sigma=float(win_sigma)).item())
    m = float(_msssim_fn(pred01, target01, data_range=1.0, size_average=True, win_size=int(win_size), win_sigma=float(win_sigma)).item())
    return s, m


# -----------------------------
# Loss builder (兼容 LossFactory / get_loss 两种)
# -----------------------------
def _build_criterion(device: torch.device):
    loss_name = getattr(Config, "LOSS_NAME", "rmse")
    data_range = float(getattr(Config, "SIC_DATA_RANGE", 1.0))
    win = int(getattr(Config, "SSIM_WIN_SIZE", 11))
    sig = float(getattr(Config, "SSIM_WIN_SIGMA", 1.5))

    # 优先使用 LossFactory（你新版 loss_and_metrics）
    try:
        from loss_and_metrics import LossConfig, LossFactory  # type: ignore

        cfg = LossConfig(
            name=str(loss_name),
            data_range=data_range,
            ssim_window=win,
            ssim_sigma=sig,
            warmup_epochs=0,  # ✅ 强制无 warmup
            init_bias_to_rmse=bool(getattr(Config, "LOSS_WEIGHT_INIT_BIAS_TO_RMSE", True)),
            weights=None,
            schedule=None,
            smoothl1_beta=float(getattr(Config, "SMOOTHL1_BETA", 0.1)),
        )
        crit = LossFactory(cfg).to(device)
        return crit, "factory"

    except Exception:
        # 回退到 get_loss（旧接口）
        from loss_and_metrics import get_loss  # type: ignore

        crit = get_loss(str(loss_name), data_range=data_range, win_size=win, win_sigma=sig).to(device)
        return crit, "get_loss"


def _criterion_forward(criterion, pred: torch.Tensor, target: torch.Tensor, epoch: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    统一兼容：
    - LossFactory: criterion(pred, target, epoch=..., mask=None) -> (loss, comps)
    - 普通 nn.Module: criterion(pred, target) -> loss
    """
    try:
        out = criterion(pred, target, epoch=epoch, mask=None)
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], out[1]
    except TypeError:
        pass

    loss = criterion(pred, target)
    return loss, {}


# -----------------------------
# Logging (CSV / JSONL)
# -----------------------------
def _init_log_files(result_dir: Path) -> Tuple[Path, Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "train_metrics.csv"
    jsonl_path = result_dir / "train_metrics.jsonl"

    header = [
        "epoch",
        "time_sec",
        "train_loss",
        "val_loss",
        "val_rmse",
        "val_ssim",
        "val_msssim",
        "train_seen",
        "train_skip",
        "val_seen",
        "val_skip",
        "lr_model",
        "lr_loss",
        "w_rmse",
        "w_ssim",
        "w_msssim",
    ]

    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    return csv_path, jsonl_path


def _append_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    header = [
        "epoch",
        "time_sec",
        "train_loss",
        "val_loss",
        "val_rmse",
        "val_ssim",
        "val_msssim",
        "train_seen",
        "train_skip",
        "val_seen",
        "val_skip",
        "lr_model",
        "lr_loss",
        "w_rmse",
        "w_ssim",
        "w_msssim",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(k, "") for k in header])


def _append_jsonl(jsonl_path: Path, row: Dict[str, Any]) -> None:
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------
# Main train loop
# -----------------------------
def train_model(model, train_loader, val_loader):
    device = torch.device(getattr(Config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(getattr(Config, "AMP", True))
    epochs = int(getattr(Config, "NUM_EPOCHS", getattr(Config, "EPOCHS", 40)))

    data_range = float(getattr(Config, "SIC_DATA_RANGE", 1.0))
    win = int(getattr(Config, "SSIM_WIN_SIZE", 11))
    sig = float(getattr(Config, "SSIM_WIN_SIGMA", 1.5))

    expected_in = int(getattr(Config, "INPUT_CHANNELS", 10))
    grad_clip = float(getattr(Config, "GRAD_CLIP_NORM", 0.0))

    # dirs / paths
    result_dir = Path(getattr(Config, "RESULT_DIR", "./results"))
    best_path = Path(getattr(Config, "BEST_MODEL_PATH", "./checkpoints/best_model.pth"))
    last_path = Path(getattr(Config, "CHECKPOINT_PATH", "./checkpoints/last_checkpoint.pth"))
    best_path.parent.mkdir(parents=True, exist_ok=True)
    last_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path, jsonl_path = _init_log_files(result_dir)

    # build loss
    criterion, crit_mode = _build_criterion(device)

    # optimizer
    base_lr = float(getattr(Config, "LEARNING_RATE", getattr(Config, "BASE_LR", 1e-4)))
    wd = float(getattr(Config, "WEIGHT_DECAY", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=wd)

    # add learnable loss weights if exist
    learn_loss_w = bool(getattr(Config, "LOSS_LEARNABLE_WEIGHTS", True))
    crit_params = list(criterion.parameters()) if hasattr(criterion, "parameters") else []
    lr_loss = None

    if learn_loss_w and len(crit_params) > 0:
        mult = float(getattr(Config, "LOSS_WEIGHT_LR_MULT", 0.1))
        lr_loss = base_lr * mult
        wd_loss = float(getattr(Config, "LOSS_WEIGHT_WD", 0.0))
        for p in crit_params:
            p.requires_grad = True
        optimizer.add_param_group({"params": crit_params, "lr": lr_loss, "weight_decay": wd_loss})

    scaler = _make_scaler(device, enabled=use_amp)

    print(f"[Train] criterion mode = {crit_mode}")
    print(f"[Train] criterion 可学习参数数量 = {sum(p.numel() for p in crit_params)}")
    print(f"[Train] model lr = {base_lr:.2e}, weight_decay={wd}")
    if lr_loss is not None:
        print(f"[Train] loss lr  = {lr_loss:.2e} (mult={getattr(Config, 'LOSS_WEIGHT_LR_MULT', 0.1)}), weight_decay={getattr(Config, 'LOSS_WEIGHT_WD', 0.0)}")
    print(f"[Train] 使用损失: {getattr(Config, 'LOSS_NAME', 'rmse')}")
    print(f"[Train] device={device.type}, amp={use_amp}")
    print(f"[Train] log csv -> {csv_path}")
    print(f"[Train] log jsonl -> {jsonl_path}")

    best_val = float("inf")

    for epoch in range(epochs):
        ep = epoch + 1
        t0 = time.time()

        # ---------------- Train ----------------
        model.train()
        sum_loss = 0.0
        n_seen = 0
        n_skip = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {ep}/{epochs} [训练]")
        for step, (inputs, targets) in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # channel check (你明确说 mask 当作一个波段 -> expected_in 应该是 10)
            if inputs.ndim != 4 or inputs.shape[1] != expected_in:
                raise RuntimeError(f"[Train] 输入通道不匹配：inputs={tuple(inputs.shape)}，期望 C={expected_in}。请检查 data_loader 是否追加了 mask 波段。")

            optimizer.zero_grad(set_to_none=True)

            with _autocast(device, enabled=use_amp):
                pred = model(inputs)

                # safety: skip bad numeric
                if (not torch.isfinite(pred).all()) or (not torch.isfinite(targets).all()):
                    n_skip += 1
                    pbar.set_postfix({"loss": "nan_skip(pred/gt)", "skip": n_skip})
                    continue

                loss, comps = _criterion_forward(criterion, pred, targets, epoch=epoch)

            if not torch.isfinite(loss):
                n_skip += 1
                pbar.set_postfix({"loss": "nan_skip(loss)", "skip": n_skip})
                continue

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = int(inputs.shape[0])
            sum_loss += float(loss.detach().item()) * bs
            n_seen += bs

            # show some weights if available
            postfix = {
                "loss": f"{float(loss.detach().item()):.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                "skip": n_skip,
            }
            # common names from LossFactory
            for k in ("w_rmse", "w_ssim", "w_msssim", "w1", "w2"):
                if k in comps:
                    try:
                        postfix[k] = f"{float(comps[k]):.3f}"
                    except Exception:
                        pass
            pbar.set_postfix(postfix)

        train_loss = sum_loss / max(1, n_seen)

        # ---------------- Val ----------------
        model.eval()
        sum_vloss = 0.0
        v_seen = 0
        v_skip = 0

        sum_rmse = 0.0
        sum_ssim = 0.0
        sum_mss = 0.0
        n_metric = 0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {ep}/{epochs} [验证]", leave=False)
            for inputs, targets in vbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if inputs.ndim != 4 or inputs.shape[1] != expected_in:
                    raise RuntimeError(f"[Val] 输入通道不匹配：inputs={tuple(inputs.shape)}，期望 C={expected_in}。")

                with _autocast(device, enabled=use_amp):
                    pred = model(inputs)

                    if (not torch.isfinite(pred).all()) or (not torch.isfinite(targets).all()):
                        v_skip += 1
                        continue

                    vloss, comps_v = _criterion_forward(criterion, pred, targets, epoch=epoch)

                if not torch.isfinite(vloss):
                    v_skip += 1
                    continue

                bs = int(inputs.shape[0])
                sum_vloss += float(vloss.detach().item()) * bs
                v_seen += bs

                # ✅ 指标：强制 float32 + autocast 关闭（避免 half/float mismatch）
                with _autocast(device, enabled=False):
                    pr01 = (pred / data_range).clamp(0.0, 1.0).float()
                    gt01 = (targets / data_range).clamp(0.0, 1.0).float()

                    sum_rmse += _rmse01(pr01, gt01) * bs
                    ssim_v, mss_v = _ssim_msssim01(pr01, gt01, win_size=win, win_sigma=sig)
                    if math.isfinite(ssim_v):
                        sum_ssim += ssim_v * bs
                    if math.isfinite(mss_v):
                        sum_mss += mss_v * bs
                    n_metric += bs

        val_loss = sum_vloss / max(1, v_seen)
        val_rmse = sum_rmse / max(1, n_metric)
        val_ssim = sum_ssim / max(1, n_metric)
        val_mss = sum_mss / max(1, n_metric)

        t1 = time.time()
        elapsed = t1 - t0

        # extract weights if possible
        w_rmse = w_ssim = w_msssim = ""
        try:
            # LossFactory exposes get_weights sometimes
            if hasattr(criterion, "get_weights"):
                w = criterion.get_weights()
                if isinstance(w, dict):
                    # may be w1/w2 only
                    if "w_rmse" in w: w_rmse = w["w_rmse"]
                    if "w_ssim" in w: w_ssim = w["w_ssim"]
                    if "w_msssim" in w: w_msssim = w["w_msssim"]
        except Exception:
            pass

        # Or read from last comps if available (best effort)
        # (we keep blank if not found)

        # lr in optimizer groups: group0=model, group1=loss weights (if exist)
        lr_model = optimizer.param_groups[0]["lr"]
        lr_loss_group = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else ""

        print(
            f"\n[Epoch {ep}] "
            f"train_loss={train_loss:.6f} (seen={n_seen}, skip={n_skip}) | "
            f"val_loss={val_loss:.6f} (seen={v_seen}, skip={v_skip}) | "
            f"val_rmse={val_rmse:.6f} | val_ssim={val_ssim:.6f} | val_msssim={val_mss:.6f}"
        )

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "val_loss": best_val,
                    "config": {k: getattr(Config, k) for k in dir(Config) if k.isupper()},
                },
                str(best_path),
            )
            print(f"[Train] ✅ save BEST -> {best_path} (val_loss={best_val:.6f})")

        # save last
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": ep,
                "val_loss": val_loss,
                "config": {k: getattr(Config, k) for k in dir(Config) if k.isupper()},
            },
            str(last_path),
        )

        # write structured logs (CSV + JSONL)
        row = {
            "epoch": ep,
            "time_sec": round(elapsed, 3),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse),
            "val_ssim": float(val_ssim),
            "val_msssim": float(val_mss),
            "train_seen": int(n_seen),
            "train_skip": int(n_skip),
            "val_seen": int(v_seen),
            "val_skip": int(v_skip),
            "lr_model": float(lr_model),
            "lr_loss": (float(lr_loss_group) if isinstance(lr_loss_group, (float, int)) else lr_loss_group),
            "w_rmse": w_rmse,
            "w_ssim": w_ssim,
            "w_msssim": w_msssim,
            "best_val_loss_so_far": float(best_val),
        }
        _append_csv(csv_path, row)
        _append_jsonl(jsonl_path, row)

    print("[Train] DONE.")
    return best_val
