# model.py
import torch
import torch.nn as nn
from config import Config




class ResidualBlock(nn.Module):
    """带残差连接的双卷积块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential()   # 如果输入输出通道数不同，使用1x1卷积调整
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)  # 残差连接
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    """带残差连接的下采样块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.residual_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        return self.residual_block(x)


class UpBlock(nn.Module):
    """带残差连接的上采样块"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # 亚像素卷积上采样
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 残差块
        self.residual_block = ResidualBlock(in_channels + out_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 来自上层的特征
        # x2: 跳跃连接的特征
        x1 = self.upsample(x1)

        # 处理尺寸差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 填充以匹配尺寸
        x1 = nn.functional.pad(x1, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2
        ])

        # 拼接跳跃连接
        x = torch.cat([x2, x1], dim=1)
        return self.residual_block(x)


class UNetWithResidual(nn.Module):
    """带编码残差块的 U-Net；支持显式粗分辨率 SIC 基线残差输出。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.output_mode = str(
            getattr(Config, "MODEL_OUTPUT_MODE", "legacy_mixed_sigmoid")
        ).lower()
        self.sic_channel_index = int(getattr(Config, "SIC_CHANNEL_INDEX", 2))
        self.sic_residual_scale = float(getattr(Config, "SIC_RESIDUAL_SCALE", 1.0))
        self.sic_interpolation = str(
            getattr(Config, "SIC_BASELINE_INTERPOLATION", "bilinear")
        ).lower()

        if not 0 <= self.sic_channel_index < in_channels:
            raise ValueError(
                f"SIC_CHANNEL_INDEX={self.sic_channel_index} 超出输入通道范围 [0, {in_channels})"
            )
        if self.output_mode not in {"sic_residual", "legacy_mixed_sigmoid"}:
            raise ValueError(
                f"未知 MODEL_OUTPUT_MODE={self.output_mode!r}；"
                "可选 sic_residual/legacy_mixed_sigmoid"
            )
        if self.output_mode == "sic_residual" and out_channels != 1:
            raise ValueError("sic_residual 当前只支持 OUTPUT_CHANNELS=1")

        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 编码器路径
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)  # 确保初始化了 down4

        # 解码器路径
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

        # 亚像素卷积
        self.final_upscale = nn.Sequential(
            nn.Conv2d(64, 64 * 16, kernel_size=3, padding=1),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

        if self.output_mode == "legacy_mixed_sigmoid":
            # 仅用于同一年份划分下复现旧模型，以便做严格消融。
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None
            if bool(getattr(Config, "SIC_RESIDUAL_ZERO_INIT", True)):
                final_conv = self.final_upscale[-1]
                nn.init.zeros_(final_conv.weight)
                if final_conv.bias is not None:
                    nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        identity = x

        # 初始卷积
        x = self.initial_conv(x)

        # 编码路径
        enc1 = x
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)
        bottle = self.down4(enc4)  # 确保这里可以调用 down4

        # 解码路径
        dec1 = self.up1(bottle, enc4)
        dec2 = self.up2(dec1, enc3)
        dec3 = self.up3(dec2, enc2)
        dec4 = self.up4(dec3, enc1)

        # 上采样
        out = self.final_upscale(dec4)

        if self.output_mode == "sic_residual":
            # 只用 coarse SIC 构造物理基线，PR/TOS/MASK 仅用于预测修正量。
            sic_coarse = identity[
                :, self.sic_channel_index : self.sic_channel_index + 1, :, :
            ]
            align_corners = False if self.sic_interpolation in {"bilinear", "bicubic"} else None
            sic_baseline = nn.functional.interpolate(
                sic_coarse,
                size=(out.size(2), out.size(3)),
                mode=self.sic_interpolation,
                align_corners=align_corners,
            )
            correction = torch.tanh(out) * self.sic_residual_scale
            # 训练时不硬裁剪，避免在 0/1 边界产生零梯度；验证脚本会裁剪到 [0,1]。
            return sic_baseline + correction

        # 旧模型：四个通道经 1x1 卷积混合作为残差，再经过 sigmoid。
        residual = self.residual_conv(identity)
        residual = nn.functional.interpolate(
            residual,
            size=(out.size(2), out.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        return torch.sigmoid(out + residual)

