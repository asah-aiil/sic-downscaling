# check_temporal_loader.py —— 检查缓存通道、数值和 batch 是否正确

from config import Config
from data_loader import TemporalMemmapDataset, create_dataloaders


def main():
    print("=== Config ===")
    print("INPUT_VARS     =", Config.INPUT_VARS)
    print("TIME_OFFSETS   =", Config.TIME_OFFSETS)
    print("INPUT_CHANNELS =", Config.INPUT_CHANNELS)

    ds = TemporalMemmapDataset(split="all")
    x, y = ds[0]
    print("\n=== Single sample ===")
    print("x.shape =", tuple(x.shape))
    print("y.shape =", tuple(y.shape))
    print("x finite =", bool(x.isfinite().all()))
    print("y finite =", bool(y.isfinite().all()))
    print("cache channel indices =", ds.channel_indices)
    print("cache channel names   =", ds.cache_channel_names)
    if hasattr(ds, "channel_names"):
        print("channel_names =")
        for i, name in enumerate(ds.channel_names):
            print(f"  {i:02d}: {name}")
            ch = x[i].float()
            print(
                f"      min={ch.min().item():.6f}, "
                f"max={ch.max().item():.6f}, mean={ch.mean().item():.6f}"
            )

    train_loader, val_loader = create_dataloaders()
    xb, yb = next(iter(train_loader))
    print("\n=== One train batch ===")
    print("xb.shape =", tuple(xb.shape))
    print("yb.shape =", tuple(yb.shape))
    print("xb finite =", bool(xb.isfinite().all()))
    print("yb finite =", bool(yb.isfinite().all()))


if __name__ == "__main__":
    main()
