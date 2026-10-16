# loss_and_metrics.py
# 统一管理：训练损失（含可学习权重复合损失）+ 验证指标（含 Evaluator，兼容旧 eval_val.py）
# 关键点：
# - SSIM/MS-SSIM 作为“loss”用 (1 - ssim) / (1 - ms-ssim)，方向与 RMSE 一致（越小越好）
# - 为避免 AMP 下 half/float mismatch：SSIM/MS-SSIM 计算统一 float32 且关闭 autocast

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Optional: pytorch_msssim
# -----------------------------
try:
    from pytorch_msssim import ssim as _ssim_fn, ms_ssim as _msssim_fn
    HAS_MSSSIM = True
except Exception:
    _ssim_fn = None
    _msssim_fn = None
    HAS_MSSSIM = False
    print("[MSSSIM] 警告: 未安装 pytorch_msssim, SSIM/MS-SSIM 相关损失或指标将不可用。")


# ============================================================
# 基础 Loss（越小越好）
# ============================================================

class RMSELoss(nn.Module):
    """RMSE = sqrt(mean((pred-target)^2) + eps)"""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        return torch.sqrt(mse + self.eps)


class SSIMLoss(nn.Module):
    """
    SSIM loss = 1 - SSIM
    pred/target: (B,1,H,W) or (B,C,H,W), expected in [0, data_range]
    """
    def __init__(self, data_range: float = 1.0, win_size: int = 11, win_sigma: float = 1.5) -> None:
        super().__init__()
        if not HAS_MSSSIM or _ssim_fn is None:
            raise RuntimeError("SSIMLoss 需要安装 pytorch_msssim。")
        self.data_range = float(data_range)
        self.win_size = int(win_size)
        self.win_sigma = float(win_sigma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # ✅ 防止 AMP 半精度导致 window dtype 不一致：用 float32 且关闭 autocast
        with torch.amp.autocast("cuda", enabled=False):
            pr = pred.float()
            gt = target.float()
            s = _ssim_fn(
                pr, gt,
                data_range=self.data_range,
                size_average=True,
                win_size=self.win_size,
                win_sigma=self.win_sigma,
            )
            return 1.0 - s


class MSSSIMLoss(nn.Module):
    """MS-SSIM loss = 1 - MS-SSIM"""

    def __init__(self, data_range: float = 1.0, win_size: int = 11, win_sigma: float = 1.5) -> None:
        super().__init__()
        if not HAS_MSSSIM or _msssim_fn is None:
            raise RuntimeError("MSSSIMLoss 需要安装 pytorch_msssim。")
        self.data_range = float(data_range)
        self.win_size = int(win_size)
        self.win_sigma = float(win_sigma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            pr = pred.float()
            gt = target.float()
            m = _msssim_fn(
                pr, gt,
                data_range=self.data_range,
                size_average=True,
                win_size=self.win_size,
                win_sigma=self.win_sigma,
            )
            return 1.0 - m


# ============================================================
# 复合损失（可学习权重）
#   loss = w1 * RMSE + w2 * (1 - SSIM/MS-SSIM)
# ============================================================

class LearnableCompositeLoss(nn.Module):
    """
    可学习权重复合损失：
      w = softmax(alpha)
      loss = w1 * L1 + w2 * L2
    """

    def __init__(
        self,
        loss1: nn.Module,
        loss2: nn.Module,
        init_bias_to_loss1: bool = True,
    ) -> None:
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2

        # alpha -> softmax -> weights
        if init_bias_to_loss1:
            init = torch.tensor([2.0, 0.0], dtype=torch.float32)  # 初始偏向 loss1（通常是 RMSE）
        else:
            init = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.raw_weights = nn.Parameter(init)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        l1 = self.loss1(pred, target)
        l2 = self.loss2(pred, target)

        w = self.weights()
        w1, w2 = w[0], w[1]
        loss = w1 * l1 + w2 * l2

        comps = {
            "w1": float(w1.detach().item()),
            "w2": float(w2.detach().item()),
            "l1": float(l1.detach().item()),
            "l2": float(l2.detach().item()),
        }
        return loss, comps


# ============================================================
# 指标（metrics）：验证用，不参与反向传播
# ============================================================

@torch.no_grad()
def rmse_metric(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return torch.sqrt(mse + eps)


@torch.no_grad()
def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if not HAS_MSSSIM or _ssim_fn is None:
        return torch.tensor(float("nan"), device=pred.device)

    with torch.amp.autocast("cuda", enabled=False):
        pr = pred.float()
        gt = target.float()
        v = _ssim_fn(
            pr, gt,
            data_range=float(data_range),
            size_average=True,
            win_size=int(window),
            win_sigma=float(sigma),
        )
        return v


@torch.no_grad()
def msssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    if not HAS_MSSSIM or _msssim_fn is None:
        return torch.tensor(float("nan"), device=pred.device)

    with torch.amp.autocast("cuda", enabled=False):
        pr = pred.float()
        gt = target.float()
        v = _msssim_fn(
            pr, gt,
            data_range=float(data_range),
            size_average=True,
            win_size=int(window),
            win_sigma=float(sigma),
        )
        return v


@torch.no_grad()
def fss_metric(
    pred01: torch.Tensor,
    target01: torch.Tensor,
    threshold: float = 0.15,
    radius: int = 3,
) -> torch.Tensor:
    """
    Fraction Skill Score (FSS)
    pred01/target01: (B,1,H,W) in [0,1]
    """
    device = pred01.device
    pred_bin = (pred01 >= threshold).float()
    targ_bin = (target01 >= threshold).float()

    k = 2 * radius + 1
    kernel = torch.ones(1, 1, k, k, device=device) / float(k * k)

    pred_mean = F.conv2d(pred_bin, kernel, padding=radius)
    targ_mean = F.conv2d(targ_bin, kernel, padding=radius)

    mse_num = torch.mean((pred_mean - targ_mean) ** 2)
    mse_den = torch.mean(pred_mean ** 2 + targ_mean ** 2)
    return 1.0 - mse_num / (mse_den + 1e-6)


@torch.no_grad()
def ice_edge_distance_metric(
    pred01: torch.Tensor,
    target01: torch.Tensor,
    threshold: float = 0.15,
    pixel_km: float = 6.25,
    max_radius: int = 20,
) -> torch.Tensor:
    """
    粗略冰缘距离（km）。pred01/target01: (B,1,H,W) in [0,1]
    用膨胀层数近似距离，适合趋势对比。
    """
    device = pred01.device
    pred_bin = (pred01 >= threshold).float()
    targ_bin = (target01 >= threshold).float()

    lap_kernel = torch.tensor(
        [[0., 1., 0.],
         [1., -4., 1.],
         [0., 1., 0.]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 3, 3)

    def edge(mask: torch.Tensor) -> torch.Tensor:
        g = F.conv2d(mask, lap_kernel, padding=1)
        return (g.abs() > 0).float()

    pred_edge = edge(pred_bin)
    targ_edge = edge(targ_bin)

    kernel = torch.ones(1, 1, 3, 3, device=device)

    dist_map = torch.full_like(targ_edge, fill_value=float(max_radius), dtype=torch.float32)
    cur = targ_edge.clone()
    for r in range(1, max_radius + 1):
        cur = (F.conv2d(cur, kernel, padding=1) > 0).float()
        dist_map = torch.where((dist_map == float(max_radius)) & (cur > 0), float(r), dist_map)

    mask_edge = pred_edge > 0
    if mask_edge.sum() == 0:
        return torch.tensor(float("nan"), device=device)

    return dist_map[mask_edge].mean() * float(pixel_km)


# ============================================================
# LossConfig / LossFactory：给 train_utils 用
# ============================================================

@dataclass
class LossConfig:
    name: str = "rmse"  # "rmse" / "ssim" / "msssim" / "rmse+ssim" / "rmse+msssim"
    data_range: float = 1.0
    ssim_window: int = 11
    ssim_sigma: float = 1.5
    warmup_epochs: int = 0  # 你目前不用 warmup
    init_bias_to_rmse: bool = True  # 复合损失初始偏向 RMSE
    weights: Optional[Dict[str, float]] = None
    schedule: Optional[Dict[str, Any]] = None
    smoothl1_beta: float = 0.1


class LossFactory(nn.Module):
    """
    统一损失入口：
      - forward 返回 (loss, comps_dict)
      - comps_dict 可用于 tqdm 展示权重等信息
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = cfg.name.lower()

        if self.name == "rmse":
            self.core: nn.Module = RMSELoss()
        elif self.name == "ssim":
            self.core = SSIMLoss(data_range=cfg.data_range, win_size=cfg.ssim_window, win_sigma=cfg.ssim_sigma)
        elif self.name == "msssim":
            self.core = MSSSIMLoss(data_range=cfg.data_range, win_size=cfg.ssim_window, win_sigma=cfg.ssim_sigma)
        elif self.name == "rmse+ssim":
            l1 = RMSELoss()
            l2 = SSIMLoss(data_range=cfg.data_range, win_size=cfg.ssim_window, win_sigma=cfg.ssim_sigma)
            self.core = LearnableCompositeLoss(l1, l2, init_bias_to_loss1=cfg.init_bias_to_rmse)
        elif self.name == "rmse+msssim":
            l1 = RMSELoss()
            l2 = MSSSIMLoss(data_range=cfg.data_range, win_size=cfg.ssim_window, win_sigma=cfg.ssim_sigma)
            self.core = LearnableCompositeLoss(l1, l2, init_bias_to_loss1=cfg.init_bias_to_rmse)
        else:
            raise ValueError(f"未知损失类型: {cfg.name}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        epoch: int = 0,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # mask 预留接口（你目前不用于 loss）
        if isinstance(self.core, LearnableCompositeLoss):
            loss, comps = self.core(pred, target)
            # 语义化键名（便于打印/记录）
            if self.name == "rmse+ssim":
                comps.update({"w_rmse": comps["w1"], "w_ssim": comps["w2"]})
            elif self.name == "rmse+msssim":
                comps.update({"w_rmse": comps["w1"], "w_msssim": comps["w2"]})
            return loss, comps

        loss = self.core(pred, target)
        return loss, {}

    def get_weights(self) -> Optional[Dict[str, float]]:
        if isinstance(self.core, LearnableCompositeLoss):
            w = self.core.weights().detach().cpu().numpy().tolist()
            return {"w1": float(w[0]), "w2": float(w[1])}
        return None


# ============================================================
# 旧接口：get_loss（可选）
# ============================================================

def get_loss(name: str, data_range: float = 1.0, win_size: int = 11, win_sigma: float = 1.5) -> nn.Module:
    n = name.lower()
    if n == "rmse":
        return RMSELoss()
    if n == "ssim":
        return SSIMLoss(data_range=data_range, win_size=win_size, win_sigma=win_sigma)
    if n == "msssim":
        return MSSSIMLoss(data_range=data_range, win_size=win_size, win_sigma=win_sigma)
    if n == "rmse+ssim":
        return LearnableCompositeLoss(
            RMSELoss(),
            SSIMLoss(data_range=data_range, win_size=win_size, win_sigma=win_sigma),
            init_bias_to_loss1=True,
        )
    if n == "rmse+msssim":
        return LearnableCompositeLoss(
            RMSELoss(),
            MSSSIMLoss(data_range=data_range, win_size=win_size, win_sigma=win_sigma),
            init_bias_to_loss1=True,
        )
    raise ValueError(f"未知损失类型: {name}")


# ============================================================
# Backward-compatible Evaluator（兼容 eval_val.py）
# ============================================================

class Evaluator:
    """
    验证阶段指标累加器（兼容旧 eval_val.py 用法）
    用法:
        ev = Evaluator(data_range=1.0, fss_radii=[3,5,9,17], ...)
        ev.update(pred, target)
        metrics = ev.compute()
    """

    def __init__(
        self,
        data_range: float = 1.0,
        fss_threshold: float = 0.15,
        fss_radii: Optional[List[int]] = None,
        ice_edge_thr: float = 0.15,
        ice_edge_pixel_km: float = 6.25,
        ssim_win_size: int = 11,
        ssim_win_sigma: float = 1.5,
    ) -> None:
        self.data_range = float(data_range)
        self.fss_threshold = float(fss_threshold)
        self.fss_radii = fss_radii or []
        self.ice_edge_thr = float(ice_edge_thr)
        self.ice_edge_pixel_km = float(ice_edge_pixel_km)

        self.ssim_win_size = int(ssim_win_size)
        self.ssim_win_sigma = float(ssim_win_sigma)

        self.n_samples = 0
        self.sum_rmse = 0.0
        self.sum_ssim = 0.0
        self.sum_msssim = 0.0
        self.sum_fss: Dict[int, float] = {int(r): 0.0 for r in self.fss_radii}
        self.sum_ice_edge_dist = 0.0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """
        pred/target: (B,1,H,W)，数值在 [0, data_range]
        """
        B = int(pred.shape[0])
        self.n_samples += B

        # ✅ 统一 [0,1] + float32 + 关闭 autocast
        with torch.amp.autocast("cuda", enabled=False):
            pr = (pred / self.data_range).clamp(0.0, 1.0).float()
            gt = (target / self.data_range).clamp(0.0, 1.0).float()

            rmse_val = float(rmse_metric(pr, gt).item())
            self.sum_rmse += rmse_val * B

            ssim_val = float(ssim_metric(pr, gt, data_range=1.0, window=self.ssim_win_size, sigma=self.ssim_win_sigma).item())
            if not math.isnan(ssim_val):
                self.sum_ssim += ssim_val * B

            mss_val = float(msssim_metric(pr, gt, data_range=1.0, window=self.ssim_win_size, sigma=self.ssim_win_sigma).item())
            if not math.isnan(mss_val):
                self.sum_msssim += mss_val * B

            for r in self.fss_radii:
                fss_val = float(fss_metric(pr, gt, threshold=self.fss_threshold, radius=int(r)).item())
                self.sum_fss[int(r)] += fss_val * B

            if self.ice_edge_pixel_km and self.ice_edge_pixel_km > 0:
                dist = ice_edge_distance_metric(pr, gt, threshold=self.ice_edge_thr, pixel_km=self.ice_edge_pixel_km)
                dval = float(dist.item())
                if not math.isnan(dval):
                    self.sum_ice_edge_dist += dval * B

    def compute(self) -> Dict[str, float]:
        if self.n_samples == 0:
            return {}

        n = float(self.n_samples)
        out: Dict[str, float] = {
            "rmse": self.sum_rmse / n,
            "ssim": self.sum_ssim / n,
            "msssim": self.sum_msssim / n,
        }
        for r in self.fss_radii:
            out[f"fss_r{int(r)}"] = self.sum_fss[int(r)] / n

        if self.ice_edge_pixel_km and self.ice_edge_pixel_km > 0:
            out["ice_edge_dist_km"] = self.sum_ice_edge_dist / n

        return out
