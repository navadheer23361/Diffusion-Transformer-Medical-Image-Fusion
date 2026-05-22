"""
DiffTransFuse — fs_head.py
Replaces DM-FNet's convolution-based AMFF with a Cross-Modal Transformer.
Everything else (Block, MSFF upsample path, HeadTanh2d) is kept identical
so the rest of the codebase requires zero changes.

Key additions (from MedFusion-TransNet paper):
  1. CrossModalTransformerBlock  — self-attention per modality + cross-attention
                                   between MRI and CT/PET/SPECT
  2. AtrousMSFF                  — multi-scale atrous convolution fusion
  3. BoundaryRefinementLoss      — distance-weighted boundary loss (used in fs_loss.py)
  4. AttentionBlock2 is replaced by CrossModalAttBlock which wraps the transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Kept from original DM-FNet (unchanged) — spatial / channel / pixel att
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        return self.sa(torch.cat([x_avg, x_max], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden_dim = max(1, dim // reduction)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, 1, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 3, padding=1,
                             padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x2 = torch.cat([x.unsqueeze(2), pattn1.unsqueeze(2)], dim=2)
        b, c, t, h, w = x2.shape
        x2 = x2.reshape(b, c * t, h, w)
        return self.sigmoid(self.pa2(x2))


class ThreeAttBlock(nn.Module):
    """Original AMFF pixel-level attention — kept and used inside CrossModalAttBlock."""
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)

    def forward(self, x):
        cattn = self.ca(x)
        sattn = self.sa(x)
        pattn1 = sattn + cattn
        pattn2 = self.pa(x, pattn1)
        return x * pattn2


# ---------------------------------------------------------------------------
# 2.  NEW — Cross-Modal Transformer Block
#     Self-attention within each modality → cross-attention between modalities
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        residual = x
        x = self.norm(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                    # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return residual + self.proj(x)


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: query from modality A, key/value from modality B.
    This lets MRI features attend to CT features and vice-versa.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, query_feat, context_feat):
        # query_feat, context_feat: (B, N, C)
        B, N, C = query_feat.shape
        q = self.q_proj(self.norm_q(query_feat))
        kv = self.kv_proj(self.norm_kv(context_feat))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return query_feat + self.out_proj(out)


class TransformerFFN(nn.Module):
    """Feed-forward network after attention, with pre-norm."""
    def __init__(self, dim, expansion=4, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class CrossModalTransformerBlock(nn.Module):
    """
    Full cross-modal block for one spatial scale:
      1. Self-attention on MRI features
      2. Self-attention on CT features
      3. Cross-attention: MRI queries CT
      4. Cross-attention: CT queries MRI
      5. FFN on each
      6. Residual pixel attention (original AMFF) on the merged result
    Input / output: (B, C, H, W) tensors.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0, max_tokens=1024):
        super().__init__()
        # Ensure num_heads divides dim
        while dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.self_attn_mri = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.self_attn_ct  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.cross_mri2ct  = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.cross_ct2mri  = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.ffn_mri = TransformerFFN(dim, dropout=dropout)
        self.ffn_ct  = TransformerFFN(dim, dropout=dropout)
        self.max_tokens = max_tokens
        # Pixel attention for final fusion (kept from original AMFF)
        self.pixel_att = ThreeAttBlock(dim)
        # Merge fused tokens back to spatial
        self.merge = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.ReLU(inplace=True),
        )

    def _token_safe_features(self, mri_feat, ct_feat):
        _, _, H, W = mri_feat.shape
        tokens = H * W
        if tokens <= self.max_tokens:
            return mri_feat, ct_feat, (H, W)

        scale = (self.max_tokens / float(tokens)) ** 0.5
        out_h = max(1, int(H * scale))
        out_w = max(1, int(W * scale))
        mri_small = F.interpolate(
            mri_feat, size=(out_h, out_w), mode='bilinear', align_corners=False)
        ct_small = F.interpolate(
            ct_feat, size=(out_h, out_w), mode='bilinear', align_corners=False)
        return mri_small, ct_small, (H, W)

    def forward(self, mri_feat, ct_feat):
        # mri_feat, ct_feat: (B, C, H, W)
        mri_base, ct_base = mri_feat, ct_feat
        mri_feat, ct_feat, original_size = self._token_safe_features(mri_feat, ct_feat)
        B, C, H, W = mri_feat.shape

        # Flatten to sequence
        mri_seq = mri_feat.flatten(2).transpose(1, 2)   # (B, N, C)
        ct_seq  = ct_feat.flatten(2).transpose(1, 2)    # (B, N, C)

        # Within-modality self-attention
        mri_seq = self.self_attn_mri(mri_seq)
        ct_seq  = self.self_attn_ct(ct_seq)

        # Cross-modal attention
        mri_cross = self.cross_mri2ct(mri_seq, ct_seq)  # MRI attends to CT
        ct_cross  = self.cross_ct2mri(ct_seq, mri_seq)  # CT attends to MRI

        # FFN
        mri_out = self.ffn_mri(mri_cross)
        ct_out  = self.ffn_ct(ct_cross)

        # Back to spatial
        mri_sp = mri_out.transpose(1, 2).reshape(B, C, H, W)
        ct_sp  = ct_out.transpose(1, 2).reshape(B, C, H, W)

        # Merge + pixel attention (original AMFF pixel path preserved)
        merged = self.merge(torch.cat([mri_sp, ct_sp], dim=1))
        if merged.shape[-2:] != original_size:
            merged = F.interpolate(
                merged, size=original_size, mode='bilinear', align_corners=False)
            merged = merged + 0.5 * (mri_base + ct_base)
        merged = self.pixel_att(merged)
        return merged


# ---------------------------------------------------------------------------
# 3.  NEW — Atrous Multi-Scale Feature Fusion (from MedFusion-TransNet)
#     Applied after transformer fusion to capture multi-scale context
# ---------------------------------------------------------------------------

class AtrousMSFF(nn.Module):
    """
    Parallel atrous convolutions at dilation rates {1, 2, 4} with
    learnable scale weights — inspired by MedFusion-TransNet Section 3.3.3.
    """
    def __init__(self, dim):
        super().__init__()
        self.dilations = [1, 2, 4]
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, 3, padding=d, dilation=d, bias=True)
            for d in self.dilations
        ])
        self.weights = nn.Parameter(torch.ones(len(self.dilations)) / len(self.dilations))
        self.bottleneck = nn.Conv2d(dim, dim, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        w = torch.softmax(self.weights, dim=0)
        out = sum(w[i] * self.convs[i](x) for i in range(len(self.dilations)))
        return self.relu(self.bottleneck(out))


# ---------------------------------------------------------------------------
# 4.  CrossModalAttBlock — drop-in replacement for original AttentionBlock2
#     Keeps the same forward signature: forward(x, y, MSFs, lvl)
# ---------------------------------------------------------------------------

class CrossModalAttBlock(nn.Module):
    """
    Replaces AttentionBlock2.
    Uses CrossModalTransformerBlock instead of ThreeAttBlock convolution.
    Spatial downscaling handles large feature maps efficiently.
    """
    def __init__(self, dim, dim_out, dims, layer_num, num_heads=8, max_tokens=1024):
        super().__init__()
        self.transformer = CrossModalTransformerBlock(
            dim, num_heads=num_heads, max_tokens=max_tokens)
        self.atrous = AtrousMSFF(dim)

        if layer_num >= 2:
            self.block1 = nn.Sequential(
                nn.Conv2d(dims[layer_num - 1], dim, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.layer_num = layer_num

    def forward(self, x, y, MSFs, lvl):
        # x = MRI feature, y = CT/PET/SPECT feature
        # CrossModal transformer replaces (w*x + (1-w)*y + x + y)
        fea = self.transformer(x, y)       # (B, dim, H, W)
        fea = self.atrous(fea)             # multi-scale atrous context

        if lvl == 1:
            fea = self.block(fea)
        elif lvl > 2:
            be_fea = F.interpolate(
                self.block1(MSFs[lvl - 3]), scale_factor=2,
                mode='bilinear', align_corners=True)
            fea = self.block(fea + MSFs[lvl - 2] + be_fea)
        else:
            fea = self.block(fea + MSFs[lvl - 2])
        return fea


# ---------------------------------------------------------------------------
# 5.  Kept from original DM-FNet — Block, head layers, helper
# ---------------------------------------------------------------------------

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
        else:
            print('Unbounded number for feat_scales. 0<=feat_scales<=14')
    return in_channels


class Block(nn.Module):
    def __init__(self, dim, dim_out, time_steps):
        super().__init__()
        c_num = max(len(time_steps), 1)
        self.block = nn.Sequential(
            nn.Conv2d(dim * c_num, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HeadTanh2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class HeadLeakyRelu2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.2)


# ---------------------------------------------------------------------------
# 6.  DiffTransFuse_Head — drop-in replacement for Fusion_Head
#     Only change: AttentionBlock2 → CrossModalAttBlock
# ---------------------------------------------------------------------------

class Fusion_Head(nn.Module):
    """
    DiffTransFuse fusion head for grayscale CT and MR slices.

    The Stage II caller must pass feature dictionaries extracted by the trained
    dual diffusion UNets: {'CT': ct_encoder_features, 'MR': mr_encoder_features}.
    """
    def __init__(self, feat_scales, out_channels=1, inner_channel=None,
                 channel_multiplier=None, img_size=(224, 256),
                 time_steps=None, num_heads=8, max_tokens=1024):
        super().__init__()
        feat_scales.sort(reverse=True)
        self.feat_scales = feat_scales
        self.in_channels = get_in_channels(feat_scales, inner_channel, channel_multiplier)
        self.img_size = img_size
        if isinstance(img_size, int):
            self.output_size = (img_size, img_size)
        else:
            self.output_size = (int(img_size[0]), int(img_size[1]))
        self.time_steps = time_steps or [0]

        dims = [
            get_in_channels([self.feat_scales[i]], inner_channel, channel_multiplier)
            for i in range(len(self.feat_scales))
        ]

        self.decoder = nn.ModuleList()
        for i in range(len(self.feat_scales)):
            dim = dims[i]
            # Time-step feature aggregation block
            self.decoder.append(Block(dim=dim, dim_out=dim, time_steps=self.time_steps))

            dim_out = dims[i + 1] if i != len(self.feat_scales) - 1 else dims[i]

            # CrossModalAttBlock for CT and MR feature fusion
            self.decoder.append(
                CrossModalAttBlock(dim=dim, dim_out=dim_out,
                                   dims=dims, layer_num=i,
                                   num_heads=num_heads,
                                   max_tokens=max_tokens)
            )

        self.fuse_decode = HeadLeakyRelu2d(dims[-1], 64)
        self.out_decode = HeadTanh2d(64, out_channels)

    def forward(self, ct_img, mr_img, feats, hard_gate=False):
        """
        Fuse CT/MR diffusion features into one grayscale image.

        ct_img and mr_img are kept in the signature for compatibility with the
        original fusion model. Feature extraction must already have been done by
        the Stage II pipeline using the trained dual UNets.
        """
        lvl = 0
        MSFs = []
        f_s = None
        if 'CT' not in feats or 'MR' not in feats:
            raise KeyError(
                "Fusion_Head expects feats['CT'] from the CT encoder and "
                "feats['MR'] from the MR encoder.")
        ct_stream = feats['CT']
        mr_stream = feats['MR']
        if ct_stream is mr_stream:
            raise ValueError(
                'CT and MR feature streams are identical. Stage II must use '
                'the corresponding trained CT and MR encoders.')

        for layer in self.decoder:
            if isinstance(layer, Block):
                f_s_ct = ct_stream[0][self.feat_scales[lvl]]
                f_s_mr = mr_stream[0][self.feat_scales[lvl]]
                for i in range(1, len(self.time_steps)):
                    f_s_ct = torch.cat(
                        (f_s_ct, ct_stream[i][self.feat_scales[lvl]]), dim=1)
                    f_s_mr = torch.cat(
                        (f_s_mr, mr_stream[i][self.feat_scales[lvl]]), dim=1)
                f_s_ct = layer(f_s_ct)
                f_s_mr = layer(f_s_mr)
                lvl += 1
            else:
                # CrossModalAttBlock: pass CT and MR features separately
                f_s = layer(f_s_ct, f_s_mr, MSFs, lvl)
                if lvl != len(self.feat_scales):
                    MSFs.append(
                        F.interpolate(f_s, scale_factor=2,
                                      mode='bilinear', align_corners=True))

        fused = self.fuse_decode(f_s)
        fused = self.out_decode(fused)
        if fused.shape[-2:] != self.output_size:
            fused = F.interpolate(
                fused, size=self.output_size, mode='bilinear', align_corners=False)
        return fused
