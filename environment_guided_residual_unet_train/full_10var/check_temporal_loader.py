# check_temporal_loader.py
from config import Config
from data_loader import TemporalMemmapDataset, create_dataloaders
from model import UNetWithResidual
import torch
import torch.nn.functional as F


def main():
    print("ARCHITECTURE =", Config.ARCHITECTURE_NAME)
    print("INPUT_VARS =", Config.INPUT_VARS)
    print("ENV_VARS =", Config.ENV_VARS)
    print("INPUT_CHANNELS =", Config.INPUT_CHANNELS)
    print("ENV_CHANNEL_INDICES =", Config.ENV_CHANNEL_INDICES)
    print("SIC_CHANNEL_INDEX =", Config.SIC_CHANNEL_INDEX)
    print("MASK_CHANNEL_INDEX =", Config.MASK_CHANNEL_INDEX)
    ds = TemporalMemmapDataset(split="all")
    x, y = ds[0]
    print("x.shape =", tuple(x.shape), "y.shape =", tuple(y.shape))
    print("finite =", bool(x.isfinite().all()), bool(y.isfinite().all()))
    print("cache channel indices =", ds.channel_indices)
    print("cache channel names =", ds.cache_channel_names)
    expected = Config.ENV_CHANNEL_INDICES + [Config.SIC_CHANNEL_INDEX, Config.MASK_CHANNEL_INDEX]
    if sorted(expected) != list(range(Config.INPUT_CHANNELS)):
        raise RuntimeError(f"Invalid branch mapping: {expected}")
    train, val = create_dataloaders()
    xb, yb = next(iter(train))
    print("train/val sizes =", len(train.dataset), len(val.dataset))
    print("batch =", tuple(xb.shape), tuple(yb.shape))
    if xb.shape[1] != Config.INPUT_CHANNELS:
        raise RuntimeError("Batch channel count mismatch")
    device = torch.device(Config.DEVICE)
    model = UNetWithResidual(Config.INPUT_CHANNELS, Config.OUTPUT_CHANNELS).to(device).eval()
    dummy = torch.rand(1, Config.INPUT_CHANNELS, 32, 32, device=device)
    with torch.inference_mode():
        prediction = model(dummy)
    baseline = F.interpolate(
        dummy[:, Config.SIC_CHANNEL_INDEX:Config.SIC_CHANNEL_INDEX + 1],
        scale_factor=4,
        mode=Config.SIC_BASELINE_INTERPOLATION,
        align_corners=False,
    )
    difference = float((prediction - baseline).abs().max())
    print("model smoke output =", tuple(prediction.shape))
    print("initial output vs SIC baseline max abs diff =", difference)
    if prediction.shape != (1, 1, 128, 128) or difference > 1e-6:
        raise RuntimeError("Environment-guided model smoke check failed")


if __name__ == "__main__":
    main()
