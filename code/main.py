# main.py —— 多年份海冰降尺度训练入口（时间窗口版实验目录）

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime

import numpy as np
import torch

from config import Config
from data_loader import create_dataloaders
from model import UNetWithResidual
from train_utils import train_model


def setup_experiment_dir():
    """每次运行自动创建一个新的实验目录，避免覆盖旧实验。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    years = getattr(Config, "YEARS", None)
    if isinstance(years, (list, tuple)) and len(years) > 0:
        years_tag = f"{years[0]}_{years[-1]}"
    else:
        years_tag = "years"

    loss_tag = getattr(Config, "LOSS_NAME", "loss").replace("/", "_")
    learnable = getattr(Config, "LOSS_LEARNABLE_WEIGHTS", False)
    learn_tag = "learnable" if learnable else "fixed"

    offsets = getattr(Config, "TIME_OFFSETS", [0])
    offset_tag = "_".join([("d" if o == 0 else f"d{o}" if o < 0 else f"d+{o}") for o in offsets])
    time_tag = f"T{len(offsets)}_{offset_tag}"

    exp_tag = f"{ts}_multi_year_{years_tag}_{time_tag}_{loss_tag}_{learn_tag}"
    Config.EXP_TAG = exp_tag

    exp_dir = os.path.join(Config.CHECKPOINT_ROOT, Config.EXP_TAG)
    os.makedirs(exp_dir, exist_ok=True)

    Config.BEST_MODEL_PATH = os.path.join(exp_dir, "best_model.pth")
    Config.CHECKPOINT_PATH = os.path.join(exp_dir, "last_checkpoint.pth")
    Config.LOG_DIR = os.path.join(exp_dir, "logs")
    Config.RESULT_DIR = os.path.join(exp_dir, "results")
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)

    print(f"[Exp] EXP_TAG        = {Config.EXP_TAG}")
    print(f"[Exp] TIME_OFFSETS   = {getattr(Config, 'TIME_OFFSETS', [0])}")
    print(f"[Exp] INPUT_CHANNELS = {Config.INPUT_CHANNELS}")
    print(f"[Exp] BEST_MODEL     = {Config.BEST_MODEL_PATH}")
    print(f"[Exp] CHECKPOINT     = {Config.CHECKPOINT_PATH}")
    print(f"[Exp] LOG_DIR        = {Config.LOG_DIR}")
    print(f"[Exp] RESULT_DIR     = {Config.RESULT_DIR}")


def main():
    seed = Config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    setup_experiment_dir()

    train_loader, val_loader = create_dataloaders()

    model = UNetWithResidual(
        in_channels=Config.INPUT_CHANNELS,
        out_channels=Config.OUTPUT_CHANNELS,
    ).to(Config.DEVICE)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    _ = train_model(model, train_loader, val_loader)

    print("训练完成！")


if __name__ == "__main__":
    main()
