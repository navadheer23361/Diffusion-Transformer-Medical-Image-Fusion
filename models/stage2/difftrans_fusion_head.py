"""
DiffTransFuse - Cross-Modal Transformer Fusion Head
Replaces the AMFF (ThreeAttBlock) in DM-FNet with a cross-modal transformer
that computes self-attention within each modality and cross-attention between
MRI and CT, enabling global long-range dependency modeling.

Original AMFF: local convolution-based spatial+channel+pixel attention
New CMT:       transformer self-attention + cross-attention across modalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────
# 1.  Positional Encoding (2-D sinusoidal, appended to features)
# ─────────────────────────────────────────────────────────────
class PositionalEncoding2D(nn.Module):
    """Adds 2-D sinusoidal positional encoding to a flattened spatial feature map."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B, N, C)  where N = H*W
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        device = x.device

        pe = torch.zeros(H, W, C, device=device)
        d_model_half = C // 2
        div_term = torch.exp(
            torch.arange(0, d_model_half, 2, device=device).float()
            * (-math.log(10000.0) / d_model_half)
        )
        pos_h = torch.arange(H, device=device).unsqueeze(1).float()
        pos_w = torch.arange(W, device=device).unsqueeze(1).float()

        pe[:, :, 0:d_model_half:2] = torch.sin(pos_h * div_term).unsqueeze(1).expand(H, W, -1)
        pe[:, :, 1:d_model_half:2] = torch.cos(pos_h * div_term).unsqueeze(1).expand(H, W, -1)
        pe[:, :, d_model_half::2]  = torch.sin(pos_w * div_term).unsqueeze(0).expand(H, W, -1)
        pe[:, :, d_model_half+1::2] = torch.cos(pos_w * div_term).unsqueeze(0).expand(H, W, -1)

        pe = pe.view(N, C).unsqueeze(0)  # (1, N, C)
        return x + pe


# ─────────────────────────────────────────────────────────────
# 2.  Multi-Head Self-Attention  (standard, within one modality)
# ─────────────────────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv  = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ─────────────────────────────────────────────────────────────
# 3.  Multi-Head Cross-Attention  (MRI queries CT keys/values)
# ─────────────────────────────────────────────────────────────
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.proj    = nn.Linear(dim, dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, query, context):
        # query:   (B, N, C)  — MRI features
        # context: (B, N, C)  — CT  features
        B, N, C = query.shape
        q  = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(context).reshape(B, N, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.drop(attn.softmax(dim=-1))
        out  = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


# ─────────────────────────────────────────────────────────────
# 4.  Feed-Forward Network (standard transformer FFN)
# ─────────────────────────────────────────────────────────────
class FFN(nn.Module):
    def __init__(self, dim, mlp_ratio=2, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# 5.  Cross-Modal Transformer Block
#     Replaces ThreeAttBlock (AMFF) from the original DM-FNet
#     Flow per modality:
#       Self-Attn → Cross-Attn (attends to other modality) → FFN
#     Both modalities are updated symmetrically.
# ─────────────────────────────────────────────────────────────
class CrossModalTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2, dropout=0.0):
        super().__init__()
        # Layer norms for MRI stream
        self.norm_mri_sa  = nn.LayerNorm(dim)
        self.norm_mri_ca  = nn.LayerNorm(dim)
        self.norm_mri_ffn = nn.LayerNorm(dim)
        # Layer norms for CT stream
        self.norm_ct_sa   = nn.LayerNorm(dim)
        self.norm_ct_ca   = nn.LayerNorm(dim)
        self.norm_ct_ffn  = nn.LayerNorm(dim)

        # Shared self-attention weights per stream
        self.sa_mri = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.sa_ct  = MultiHeadSelfAttention(dim, num_heads, dropout)

        # Cross-attention: MRI ← CT  and  CT ← MRI
        self.ca_mri = MultiHeadCrossAttention(dim, num_heads, dropout)  # MRI queries CT
        self.ca_ct  = MultiHeadCrossAttention(dim, num_heads, dropout)  # CT queries MRI

        # FFN per stream
        self.ffn_mri = FFN(dim, mlp_ratio, dropout)
        self.ffn_ct  = FFN(dim, mlp_ratio, dropout)

        self.pos = PositionalEncoding2D(dim)

    def forward(self, f_mri, f_ct):
        """
        f_mri, f_ct: (B, C, H, W)
        returns updated f_mri, f_ct: (B, C, H, W)
        """
        B, C, H, W = f_mri.shape
        # Flatten spatial dims → token sequence
        mri = f_mri.flatten(2).transpose(1, 2)   # (B, H*W, C)
        ct  = f_ct.flatten(2).transpose(1, 2)

        # Add positional encoding
        mri = self.pos(mri)
        ct  = self.pos(ct)

        # ── MRI stream ──────────────────────────────────────────
        mri = mri + self.sa_mri(self.norm_mri_sa(mri))       # self-attn
        mri = mri + self.ca_mri(self.norm_mri_ca(mri), ct)   # cross-attn ← CT
        mri = mri + self.ffn_mri(self.norm_mri_ffn(mri))     # FFN

        # ── CT stream ───────────────────────────────────────────
        ct  = ct  + self.sa_ct(self.norm_ct_sa(ct))          # self-attn
        ct  = ct  + self.ca_ct(self.norm_ct_ca(ct), mri)     # cross-attn ← MRI
        ct  = ct  + self.ffn_ct(self.norm_ct_ffn(ct))        # FFN

        # Reshape back to spatial
        f_mri = mri.transpose(1, 2).reshape(B, C, H, W)
        f_ct  = ct.transpose(1, 2).reshape(B, C, H, W)
        return f_mri, f_ct


# ─────────────────────────────────────────────────────────────
# 6.  Feature merging after transformer  (replaces w*x+(1-w)*y)
#     Learnable weighted merge + residual
# ─────────────────────────────────────────────────────────────
class LearnableMerge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))   # learnable blend weight

    def forward(self, f_mri, f_ct):
        a = torch.sigmoid(self.alpha)
        return a * f_mri + (1 - a) * f_ct + f_mri + f_ct  # weighted + residual


# ─────────────────────────────────────────────────────────────
# 7.  Block1 — same as original (multi-timestep aggregation)
# ─────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        c_num = max(len(time_steps), 1)
        self.block = nn.Sequential(
            nn.Conv2d(dim * c_num, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────────────────────
# 8.  Projection conv  (channel-to-transformer-dim adapter)
#     Needed because diffusion features may have large channels
#     (e.g. 512) that are expensive for full attention.
#     We project down, run transformer, project back.
# ─────────────────────────────────────────────────────────────
class ChannelAdapter(nn.Module):
    def __init__(self, in_dim, attn_dim):
        super().__init__()
        self.down = nn.Conv2d(in_dim, attn_dim, 1)
        self.up   = nn.Conv2d(attn_dim, in_dim, 1)
        self.norm = nn.GroupNorm(8, attn_dim)

    def down_proj(self, x):
        return F.relu(self.norm(self.down(x)))

    def up_proj(self, x):
        return self.up(x)


# ─────────────────────────────────────────────────────────────
# 9.  CMT Attention Block  (drop-in replacement for AttentionBlock2)
#     Has identical interface:  forward(f_mri, f_ct, MSFs, lvl)
# ─────────────────────────────────────────────────────────────
class CMTAttentionBlock(nn.Module):
    def __init__(self, dim, dim_out, dims, layer_num,
                 attn_dim=128, num_heads=4):
        super().__init__()
        # Channel adapter: project to attn_dim for transformer
        self.adapter   = ChannelAdapter(dim, min(attn_dim, dim))
        actual_attn_dim = min(attn_dim, dim)

        # Make sure attn_dim divisible by num_heads
        while actual_attn_dim % num_heads != 0:
            num_heads -= 1

        self.cmt_block = CrossModalTransformerBlock(
            dim=actual_attn_dim, num_heads=num_heads
        )
        self.merge = LearnableMerge(dim)

        # Output conv block (same role as original)
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.ReLU(),
        )

        # Optional skip from previous MSF scale (same as original)
        if layer_num >= 2:
            self.block1 = nn.Sequential(
                nn.Conv2d(dims[layer_num - 1], dim, 1),
                nn.ReLU(),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.ReLU(),
            )

        self.layer_num = layer_num
        self.dims = dims

    def forward(self, f_mri, f_ct, MSFs, lvl):
        # Project down for transformer
        f_mri_low = self.adapter.down_proj(f_mri)
        f_ct_low  = self.adapter.down_proj(f_ct)

        # Cross-modal transformer (core novelty)
        f_mri_low, f_ct_low = self.cmt_block(f_mri_low, f_ct_low)

        # Project back up
        f_mri = self.adapter.up_proj(f_mri_low)
        f_ct  = self.adapter.up_proj(f_ct_low)

        # Learnable weighted merge
        fea = self.merge(f_mri, f_ct)

        # Multi-scale skip connections (identical to original logic)
        if lvl == 1:
            fea = self.block(fea)
        elif lvl > 2:
            be_fea = F.interpolate(
                self.block1(MSFs[lvl - 3]),
                scale_factor=2, mode="bilinear", align_corners=True
            )
            fea = self.block(fea + MSFs[lvl - 2] + be_fea)
        else:
            fea = self.block(fea + MSFs[lvl - 2])

        return fea


# ─────────────────────────────────────────────────────────────
# 10. Final head convs  (unchanged from original)
# ─────────────────────────────────────────────────────────────
class HeadTanh2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class HeadLeakyRelu2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


# ─────────────────────────────────────────────────────────────
# Helper: same channel-count logic as original
# ─────────────────────────────────────────────────────────────
def get_in_channels(feat_scales, inner_channel, channel_multiplier):
    in_channels = 0
    for scale in feat_scales:
        if scale < 3:
            in_channels += inner_channel * channel_multiplier[0]
        elif scale < 6:
            in_channels += inner_channel * channel_multiplier[1]
        elif scale < 9:
            in_channels += inner_channel * channel_multiplier[2]
        elif scale < 12:
            in_channels += inner_channel * channel_multiplier[3]
        elif scale < 15:
            in_channels += inner_channel * channel_multiplier[4]
    return in_channels


# ─────────────────────────────────────────────────────────────
# 11. DiffTransFuse_Head  — full fusion head, drop-in replacement
#     for Fusion_Head in fs_head.py
#     API is IDENTICAL: __init__ and forward signatures unchanged.
# ─────────────────────────────────────────────────────────────
class DiffTransFuse_Head(nn.Module):
    """
    Drop-in replacement for Fusion_Head.
    Identical constructor and forward API.
    Only the internal attention mechanism changes:
        ThreeAttBlock (conv attention) → CrossModalTransformerBlock
    """
    def __init__(self, feat_scales, out_channels=1, inner_channel=None,
                 channel_multiplier=None, img_size=256,
                 time_steps=None, hard_gate=False,
                 attn_dim=128, num_heads=4):
        super().__init__()
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.time_steps  = time_steps
        self.img_size    = img_size

        dims = [
            get_in_channels([s], inner_channel, channel_multiplier)
            for s in feat_scales
        ]

        self.decoder = nn.ModuleList()
        for i in range(len(feat_scales)):
            dim = dims[i]
            # Multi-timestep aggregation block (unchanged)
            self.decoder.append(Block(dim=dim, dim_out=dim, time_steps=time_steps))
            # CMT attention block (NEW — replaces AttentionBlock2)
            dim_out = dims[i + 1] if i != len(feat_scales) - 1 else dims[i]
            self.decoder.append(
                CMTAttentionBlock(
                    dim=dim, dim_out=dim_out, dims=dims,
                    layer_num=i, attn_dim=attn_dim, num_heads=num_heads
                )
            )

        self.rgb_decode2 = HeadLeakyRelu2d(128, 64)
        self.rgb_decode1 = HeadTanh2d(64, out_channels)

    def forward(self, y1, y2, feats, hard_gate=False):
        lvl   = 0
        MSFs  = []
        f_s   = None

        for layer in self.decoder:
            if isinstance(layer, Block):
                # Stack multi-timestep features for each modality
                if 'CT' not in feats or 'MR' not in feats:
                    raise KeyError(
                        "DiffTransFuse_Head expects feats['CT'] from the CT "
                        "encoder and feats['MR'] from the MR encoder.")
                f_s_1 = feats['MR'][0][self.feat_scales[lvl]]
                f_s_2 = feats['CT'][0][self.feat_scales[lvl]]
                if len(self.time_steps) > 1:
                    for i in range(1, len(self.time_steps)):
                        f_s_1 = torch.cat(
                            (f_s_1, feats['MR'][i][self.feat_scales[lvl]]), dim=1
                        )
                        f_s_2 = torch.cat(
                            (f_s_2, feats['CT'][i][self.feat_scales[lvl]]), dim=1
                        )
                    f_s_1 = layer(f_s_1)
                    f_s_2 = layer(f_s_2)
                lvl += 1
            else:
                # CMTAttentionBlock — cross-modal transformer fusion
                f_s = layer(f_s_1, f_s_2, MSFs, lvl)
                if lvl != len(self.feat_scales):
                    MSFs.append(
                        F.interpolate(f_s, scale_factor=2,
                                      mode="bilinear", align_corners=True)
                    )

        x       = self.rgb_decode2(f_s)
        rgb_img = self.rgb_decode1(x)
        return rgb_img
