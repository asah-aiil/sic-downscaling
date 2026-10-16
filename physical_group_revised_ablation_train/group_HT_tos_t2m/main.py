# main.py -- Environment-guided Residual U-Net training
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    years_tag = f"{Config.YEARS[0]}_{Config.YEARS[-1]}"
    offsets = Config.TIME_OFFSETS
    offset_tag = "_".join("d" if o == 0 else f"d{o}" for o in offsets)
    feature_tag = "_".join(Config.INPUT_VARS) + ("_mask" if Config.ADD_MASK_CHANNEL else "")
    exp_tag = (
        f"{ts}_envgated_{feature_tag}_{Config.SPLIT_MODE}_{Config.MODEL_OUTPUT_MODE}_"
        f"{years_tag}_T{len(offsets)}_{offset_tag}_{Config.LOSS_NAME}_"
        f"{'learnable' if Config.LOSS_LEARNABLE_WEIGHTS else 'fixed'}"
    )
    Config.EXP_TAG = exp_tag
    exp_dir = os.path.join(Config.CHECKPOINT_ROOT, exp_tag)
    Config.BEST_MODEL_PATH = os.path.join(exp_dir, "best_model.pth")
    Config.CHECKPOINT_PATH = os.path.join(exp_dir, "last_checkpoint.pth")
    Config.LOG_DIR = os.path.join(exp_dir, "logs")
    Config.RESULT_DIR = os.path.join(exp_dir, "results")
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_DIR, exist_ok=True)
    print(f"[Exp] EXP_TAG        = {exp_tag}")
    print(f"[Exp] ARCHITECTURE   = {Config.ARCHITECTURE_NAME}")
    print(f"[Exp] INPUT_VARS     = {Config.INPUT_VARS}")
    print(f"[Exp] ENV_VARS       = {Config.ENV_VARS}")
    print(f"[Exp] BRANCH INDICES = env {Config.ENV_CHANNEL_INDICES}, sic {Config.SIC_CHANNEL_INDEX}, mask {Config.MASK_CHANNEL_INDEX}")
    print(f"[Exp] BEST_MODEL     = {Config.BEST_MODEL_PATH}")


def main():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    setup_experiment_dir()
    train_loader, val_loader = create_dataloaders()
    model = UNetWithResidual(Config.INPUT_CHANNELS, Config.OUTPUT_CHANNELS).to(Config.DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    train_model(model, train_loader, val_loader)
    print("训练完成！")


if __name__ == "__main__":
    main()



