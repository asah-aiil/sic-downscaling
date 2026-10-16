# model.py -- Environment-guided Residual U-Net
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stem = ResidualBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(512, 1024))

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class EnvironmentGate(nn.Module):
    """gate=sigmoid(conv(environment)); SIC=SIC*(1+scale*gate)."""
    def __init__(self, channels, initial_bias=-4.0, scale=1.0):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        self.scale = float(scale)
        # Initial gate is sigmoid(-4)=0.018, so the SIC path begins near identity.
        nn.init.zeros_(self.conv.weight)
        nn.init.constant_(self.conv.bias, float(initial_bias))

    def forward(self, sic_feature, environment_feature):
        gate = torch.sigmoid(self.conv(environment_feature))
        return sic_feature * (1.0 + self.scale * gate)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, decoder_feature, skip_feature):
        decoder_feature = self.upsample(decoder_feature)
        diff_y = skip_feature.size(2) - decoder_feature.size(2)
        diff_x = skip_feature.size(3) - decoder_feature.size(3)
        decoder_feature = F.pad(
            decoder_feature,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        return self.block(torch.cat([skip_feature, decoder_feature], dim=1))


class UNetWithResidual(nn.Module):
    """Two encoders; environment features gate SIC+mask features at five scales."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        if str(getattr(Config, "MODEL_OUTPUT_MODE", "")).lower() != "sic_residual":
            raise ValueError("Environment-guided model only supports sic_residual")
        if out_channels != 1:
            raise ValueError("Only one SIC output channel is supported")

        self.sic_channel_index = int(Config.SIC_CHANNEL_INDEX)
        self.mask_channel_index = int(Config.MASK_CHANNEL_INDEX)
        self.environment_channel_indices = [int(i) for i in Config.ENV_CHANNEL_INDICES]
        expected = self.environment_channel_indices + [
            self.sic_channel_index, self.mask_channel_index
        ]
        if len(set(expected)) != len(expected) or sorted(expected) != list(range(in_channels)):
            raise ValueError(
                f"Branch channel mapping must cover input exactly once: {expected}, in={in_channels}"
            )
        if not self.environment_channel_indices:
            raise ValueError("Environment encoder requires at least one environment channel")

        self.sic_residual_scale = float(getattr(Config, "SIC_RESIDUAL_SCALE", 1.0))
        self.sic_interpolation = str(
            getattr(Config, "SIC_BASELINE_INTERPOLATION", "bilinear")
        ).lower()

        self.sic_encoder = Encoder(2)
        self.environment_encoder = Encoder(len(self.environment_channel_indices))
        bias = float(getattr(Config, "ENV_GATE_INITIAL_BIAS", -4.0))
        scale = float(getattr(Config, "ENV_GATE_SCALE", 1.0))
        self.environment_gates = nn.ModuleList(
            [EnvironmentGate(c, bias, scale) for c in (64, 128, 256, 512, 1024)]
        )

        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)
        self.final_upscale = nn.Sequential(
            nn.Conv2d(64, 64 * 16, 3, padding=1),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )
        if bool(getattr(Config, "SIC_RESIDUAL_ZERO_INIT", True)):
            nn.init.zeros_(self.final_upscale[-1].weight)
            nn.init.zeros_(self.final_upscale[-1].bias)

    def forward(self, x):
        sic = x[:, self.sic_channel_index:self.sic_channel_index + 1]
        mask = x[:, self.mask_channel_index:self.mask_channel_index + 1]
        environment = x[:, self.environment_channel_indices]
        sic_features = self.sic_encoder(torch.cat([sic, mask], dim=1))
        environment_features = self.environment_encoder(environment)
        fused = [
            gate(sic_feature, environment_feature)
            for gate, sic_feature, environment_feature in zip(
                self.environment_gates, sic_features, environment_features
            )
        ]
        x1, x2, x3, x4, bottle = fused
        decoder = self.up1(bottle, x4)
        decoder = self.up2(decoder, x3)
        decoder = self.up3(decoder, x2)
        decoder = self.up4(decoder, x1)
        correction_logits = self.final_upscale(decoder)

        align_corners = (
            False if self.sic_interpolation in {"bilinear", "bicubic"} else None
        )
        baseline = F.interpolate(
            sic,
            size=correction_logits.shape[-2:],
            mode=self.sic_interpolation,
            align_corners=align_corners,
        )
        delta_sic = torch.tanh(correction_logits) * self.sic_residual_scale
        return baseline + delta_sic



