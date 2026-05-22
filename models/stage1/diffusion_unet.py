"""Stage 1 diffusion UNet components used across training and inference."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Embed diffusion timesteps with sinusoidal features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timestep.device
        emb_scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device) * -emb_scale)
        embedding = timestep.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([embedding.sin(), embedding.cos()], dim=1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResBlock(nn.Module):
    """Residual block conditioned on the diffusion timestep embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = F.silu(hidden)
        hidden = hidden + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = F.silu(hidden)
        return hidden + self.skip(x)


class Stage1DUNet(nn.Module):
    """Compact diffusion UNet with separately saveable encoder weights."""

    def __init__(self, in_ch: int = 1, base_ch: int = 32, time_dim: int = 128) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.enc1 = ResBlock(in_ch, base_ch, time_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, stride=2, padding=1)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, stride=2, padding=1)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1)

        self.mid = ResBlock(base_ch * 4, base_ch * 8, time_dim)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1)
        self.dec3 = ResBlock(base_ch * 8, base_ch * 4, time_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 4, base_ch * 2, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_dim)
        self.out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def encoder(
        self, x: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode one noisy image into multi-scale features."""
        time_emb = self.time_mlp(timestep)
        e1 = self.enc1(x, time_emb)
        e2 = self.enc2(self.down1(e1), time_emb)
        e3 = self.enc3(self.down2(e2), time_emb)
        bottleneck = self.mid(self.down3(e3), time_emb)
        return e1, e2, e3, bottleneck, time_emb

    def feature_pyramid(self, x: torch.Tensor, timestep: torch.Tensor) -> list[torch.Tensor]:
        """Return the Stage 2-compatible 15-slot feature pyramid."""
        e1, e2, e3, bottleneck, _ = self.encoder(x, timestep)
        deepest = F.avg_pool2d(bottleneck, kernel_size=2, stride=2)
        return [
            e1,
            e1,
            e1,
            e2,
            e2,
            e2,
            e3,
            e3,
            e3,
            bottleneck,
            bottleneck,
            bottleneck,
            deepest,
            deepest,
            deepest,
        ]

    def extract_features(
        self, x: torch.Tensor, t: torch.Tensor | int, feat_type: str = "dec"
    ) -> list[torch.Tensor]:
        """Compatibility wrapper for legacy Stage 2 feature extraction."""
        del feat_type
        if isinstance(t, int):
            timestep = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        else:
            timestep = t.to(device=x.device, dtype=torch.long)
        return self.feature_pyramid(x, timestep)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, bottleneck, time_emb = self.encoder(x, timestep)

        hidden = self.up3(bottleneck)
        hidden = self.dec3(torch.cat([hidden, e3], dim=1), time_emb)
        hidden = self.up2(hidden)
        hidden = self.dec2(torch.cat([hidden, e2], dim=1), time_emb)
        hidden = self.up1(hidden)
        hidden = self.dec1(torch.cat([hidden, e1], dim=1), time_emb)
        return self.out(hidden)

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """Return the encoder-only weights for reuse in Stage 2."""
        keys = ("time_mlp", "enc1", "down1", "enc2", "down2", "enc3", "down3", "mid")
        return {
            key: value
            for key, value in self.state_dict().items()
            if key.split(".")[0] in keys
        }


class GaussianDiffusion:
    """Minimal forward diffusion process used by the local training pipeline."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.timesteps = timesteps
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    def q_sample(self, clean_image: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample a noisy image at timestep ``t`` from the clean input."""
        a = self.sqrt_alpha_cumprod[timestep].view(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_cumprod[timestep].view(-1, 1, 1, 1)
        return a * clean_image + b * noise
