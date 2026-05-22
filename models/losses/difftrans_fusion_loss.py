"""
DiffTransFuse Loss Function
Extends DM-FNet's Fusionloss with:
  1. Boundary refinement loss  (from MedFusion-TransNet DRGO idea)
  2. Adaptive region weighting (uncertainty-based, from DRGO)
  3. Keeps all original terms: L_int (1.5x), L_ssim, L_grad (Sobel)

Total loss:
    L = alpha * L_int + beta * (L_ssim + L_grad) + gamma_b * L_boundary + gamma_a * L_adaptive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# ─────────────────────────────────────────────────────────────
# SSIM helpers  (kept identical from original fs_loss.py)
# ─────────────────────────────────────────────────────────────
def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_std(img1, img2, img3, window, window_size, channel, size_average=True):
    """Max-std SSIM from original: picks the source patch with larger std."""
    def _mu_sigma(img):
        mu    = F.conv2d(img,       window, padding=window_size // 2, groups=channel)
        sigma = F.conv2d(img * img, window, padding=window_size // 2, groups=channel) - mu.pow(2)
        return mu, sigma

    mu1, s1 = _mu_sigma(img1)
    mu2, s2 = _mu_sigma(img2)
    mu3, s3 = _mu_sigma(img3)

    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1 * mu2
    sigma13 = F.conv2d(img1 * img3, window, padding=window_size // 2, groups=channel) - mu1 * mu3

    C2 = 0.03 ** 2
    ssim12 = (2 * sigma12 + C2) / (s1 + s2 + C2)
    ssim13 = (2 * sigma13 + C2) / (s1 + s3 + C2)

    chosen = torch.where(torch.abs(torch.sqrt(s2.clamp(0))) >= torch.abs(torch.sqrt(s3.clamp(0))),
                         ssim12, ssim13)
    return chosen.mean() if size_average else chosen.mean(1).mean(1).mean(1)


def ssim_max_std(img1, img2, img3, window_size=11):
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    return _ssim_std(img1, img2, img3, window, window_size, channel)


# ─────────────────────────────────────────────────────────────
# Sobel gradient operator  (identical to original)
# ─────────────────────────────────────────────────────────────
class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        ky = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('weightx', kx)
        self.register_buffer('weighty', ky)

    def forward(self, x):
        b, c, _, _ = x.shape
        out = []
        for i in range(b):
            ch_list = []
            for j in range(c):
                patch = x[i, j:j+1].unsqueeze(0)
                gx = F.conv2d(patch, self.weightx, padding=1)
                gy = F.conv2d(patch, self.weighty, padding=1)
                ch_list.append(torch.abs(gx) + torch.abs(gy))
            out.append(torch.cat(ch_list, dim=1))
        return torch.cat(out, dim=0)


# ─────────────────────────────────────────────────────────────
# Boundary Refinement Loss  (from MedFusion DRGO idea)
# Penalises disagreement at predicted high-gradient (boundary)
# regions, weighted by inverse distance to boundary.
# ─────────────────────────────────────────────────────────────
class BoundaryRefinementLoss(nn.Module):
    """
    Computes a boundary-aware loss that upweights pixels near edges.

    Steps:
      1. Detect edges in both source images with Sobel.
      2. Create a joint boundary map B (union of strong edges).
      3. Build weight map w_ij = 1 / (1 + distance_from_boundary).
         Approximated efficiently with a max-pooling dilation.
      4. L_boundary = mean( w * |fused_grad - max_source_grad|^2 )
    """
    def __init__(self, edge_thresh=0.1):
        super().__init__()
        self.sobel      = Sobelxy()
        self.edge_thresh = edge_thresh

    def _dilate_mask(self, mask, radius=3):
        """Approximate distance weighting via max-pool dilation."""
        kernel = 2 * radius + 1
        dilated = F.max_pool2d(mask.float(), kernel, stride=1, padding=radius)
        # weight decays from boundary: pixels inside dilation get value < 1
        weight = 1.0 - (dilated - mask.float()).clamp(0, 1) * 0.5
        return weight.clamp(0.1, 1.0)

    def forward(self, fused, ir, vis):
        # Gradients
        fused_g = self.sobel(fused)
        ir_g    = self.sobel(ir)
        vis_g   = self.sobel(vis)

        # Joint boundary map from sources
        source_max_g = torch.max(ir_g, vis_g)
        edge_mask    = (source_max_g > self.edge_thresh).float()

        # Boundary weight map
        w = self._dilate_mask(edge_mask.mean(dim=1, keepdim=True))

        # Weighted L2 between fused gradients and max-source gradients
        diff   = (fused_g - source_max_g) ** 2
        l_bnd  = (w * diff).mean()
        return l_bnd


# ─────────────────────────────────────────────────────────────
# Adaptive Region Loss  (from DRGO uncertainty weighting idea)
# Re-weights patches by prediction uncertainty (entropy) so
# the optimizer focuses on hard / rare anatomical regions.
# ─────────────────────────────────────────────────────────────
class AdaptiveRegionLoss(nn.Module):
    """
    Patch-level uncertainty weighting.
    High-entropy (uncertain) patches get higher loss weight.
    This mimics DRGO dynamic region sampling in loss space.

    Note: "uncertainty" here is approximated from the fused image
    itself — patches where fused values are close to 0.5 (uncertain)
    or where local standard deviation is high are harder.
    """
    def __init__(self, patch_size=16, gamma=0.5):
        super().__init__()
        self.patch_size = patch_size
        self.gamma      = gamma

    def _patch_uncertainty(self, fused):
        """Compute local std as uncertainty proxy per spatial patch."""
        p  = self.patch_size
        B, C, H, W = fused.shape
        # Pad to make H, W divisible by p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        x = F.pad(fused, (0, pad_w, 0, pad_h))
        _, _, H2, W2 = x.shape
        # Unfold into patches
        patches = x.unfold(2, p, p).unfold(3, p, p)  # B, C, nH, nW, p, p
        # Local std
        std = patches.reshape(B, C, -1, p * p).std(dim=-1)  # B, C, nH*nW
        uncertainty = std.mean(dim=1)                         # B, nH*nW
        # Normalise to [0,1]
        u_min = uncertainty.min(dim=-1, keepdim=True)[0]
        u_max = uncertainty.max(dim=-1, keepdim=True)[0]
        uncertainty = (uncertainty - u_min) / (u_max - u_min + 1e-8)
        # Expand back to spatial map
        nH, nW = H2 // p, W2 // p
        uncertainty = uncertainty.reshape(B, 1, nH, nW)
        weight = F.interpolate(uncertainty, size=(H, W), mode='nearest')
        return weight  # B, 1, H, W

    def forward(self, fused, ir, vis):
        w       = self._patch_uncertainty(fused).detach()  # no grad through weights
        w       = 1.0 + self.gamma * w                     # base 1 + uncertainty boost
        source_max = torch.max(ir, vis)
        loss    = (w * torch.abs(fused - source_max)).mean()
        return loss


# ─────────────────────────────────────────────────────────────
# DiffTransFuse Loss  (main class, use this in Fusion_model.py)
# ─────────────────────────────────────────────────────────────
class DiffTransFuseLoss(nn.Module):
    """
    Combined loss = alpha * L_int
                  + beta  * (L_ssim + L_grad)
                  + gamma_b * L_boundary
                  + gamma_a * L_adaptive

    Default weights chosen from:
      alpha=1.5, beta=0.5  — same as DM-FNet paper (Table I best config)
      gamma_b=0.3          — boundary loss contribution
      gamma_a=0.2          — adaptive region loss contribution
    """
    def __init__(self, alpha=1.5, beta=0.5, gamma_b=0.3, gamma_a=0.2):
        super().__init__()
        self.alpha   = alpha
        self.beta    = beta
        self.gamma_b = gamma_b
        self.gamma_a = gamma_a

        self.sobel    = Sobelxy()
        self.bnd_loss = BoundaryRefinementLoss(edge_thresh=0.1)
        self.adp_loss = AdaptiveRegionLoss(patch_size=16, gamma=0.5)

    def forward(self, data, generate_img):
        """
        Same signature as original Fusionloss.forward(data, generate_img)
        Returns: (loss_in, loss_grad_combined)
        so Fusion_model.py needs no changes.
        """
        vis = data["vis"]
        ir  = data["ir"]

        B, C, W, H = vis.shape
        ir = ir.expand(B, C, W, H)

        # ── 1. Intensity loss (L_int)  ──────────────────────────
        x_in_max   = torch.max(vis, ir)
        loss_int   = F.l1_loss(generate_img, x_in_max)
        loss_int   = self.alpha * loss_int

        # ── 2. SSIM loss  ───────────────────────────────────────
        loss_ssim  = 0.5 * (1 - ssim_max_std(generate_img, ir, vis))

        # ── 3. Gradient loss (Sobel)  ───────────────────────────
        vis_g      = self.sobel(vis)
        ir_g       = self.sobel(ir)
        fused_g    = self.sobel(generate_img)
        x_grad_max = torch.maximum(vis_g, ir_g)
        loss_grad  = F.l1_loss(fused_g, x_grad_max)

        # ── 4. Boundary refinement loss (NEW)  ──────────────────
        loss_bnd   = self.bnd_loss(generate_img, ir, vis)

        # ── 5. Adaptive region loss (NEW)  ──────────────────────
        loss_adp   = self.adp_loss(generate_img, ir, vis)

        # ── Combine ─────────────────────────────────────────────
        # l_grad_combined carries ssim+grad+bnd+adp for logging clarity
        l_in   = loss_int
        l_grad = (self.beta * (loss_ssim + loss_grad)
                  + self.gamma_b * loss_bnd
                  + self.gamma_a * loss_adp)

        return l_in, l_grad
