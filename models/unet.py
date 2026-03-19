import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding as used in the original Transformer paper.
    Maps a timestep t to a high-dimensional embedding vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: shape [B]
        return: shape [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]   # [B, half_dim]

        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, dim]
        return emb


class ConvBlock(nn.Module):
    """
    Conv -> GroupNorm -> SiLU
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # skip connection
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        """
        x: [B, C, H, W]
        t_emb: [B, time_emb_dim]
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        # add time embedding
        time_feat = self.time_mlp(t_emb)[:, :, None, None]  # [B, C, 1, 1]
        h = h + time_feat

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)

        return h + self.res_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block:
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block1 = ConvBlock(in_channels, out_channels, time_emb_dim)
        self.block2 = ConvBlock(out_channels, out_channels, time_emb_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    """
    Upsampling block:
    """
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block1 = ConvBlock(out_channels + skip_channels, out_channels, time_emb_dim)
        self.block2 = ConvBlock(out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, t_emb):
        x = self.upsample(x)

        # if skip and x have different spatial sizes due to rounding, 
        # we can center-crop the skip connection to match x
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")

        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t_emb)
        x = self.block2(x, t_emb)
        return x


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net
    input:
        xt: [B, 3, H, W]
        cond: [B, 3, H, W]   # degraded image
        t: [B]               # diffusion timestep
    output:
        predicted noise: [B, 3, H, W]
    """
    def __init__(self, in_channels=6, out_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # input layer
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # downsampling
        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)          # 64
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)      # 128
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)  # 256

        # bottleneck
        self.mid1 = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ConvBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # upsampling
        self.up3 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2, time_emb_dim)  # 256 -> 128
        self.up2 = UpBlock(base_channels * 2, base_channels * 2, base_channels, time_emb_dim)       # 128 -> 64
        self.up1 = UpBlock(base_channels, base_channels, base_channels, time_emb_dim)                # 64 -> 64

        # output layer
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, xt, cond, t):
        """
        xt:   [B, 3, H, W]
        cond: [B, 3, H, W]
        t:    [B]
        """
        # cat
        x = torch.cat([xt, cond], dim=1)   # [B, 6, H, W]

        # time embedding
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # conv input
        x = self.input_conv(x)

        # encoder
        x, skip1 = self.down1(x, t_emb)
        x, skip2 = self.down2(x, t_emb)
        x, skip3 = self.down3(x, t_emb)

        # bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        # decoder
        x = self.up3(x, skip3, t_emb)
        x = self.up2(x, skip2, t_emb)
        x = self.up1(x, skip1, t_emb)

        # output
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_conv(x)

        return x