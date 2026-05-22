"""
DiffTransFuse — fs_loss.py
Extends DM-FNet's Fusionloss with two additions from MedFusion-TransNet:
  1. BoundaryRefinementLoss  — distance-weighted boundary pixel loss
  2. AdaptiveLoss            — uncertainty-aware region weighting
The final combined loss is used in Fusion_model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


# ---------------------------------------------------------------------------
# Kept from original DM-FNet (unchanged)
# ---------------------------------------------------------------------------

def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D = gaussian(window_size, 1.5).unsqueeze(1)
    _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_three(img1, img2, img3, window, window_size, channel, size_average=True):
    def conv(img):
        return F.conv2d(img, window, padding=window_size // 2, groups=channel)

    mu1, mu2, mu3 = conv(img1), conv(img2), conv(img3)
    s1 = conv(img1 * img1) - mu1 ** 2
    s2 = conv(img2 * img2) - mu2 ** 2
    s3 = conv(img3 * img3) - mu3 ** 2
    s12 = conv(img1 * img2) - mu1 * mu2
    s13 = conv(img1 * img3) - mu1 * mu3
    C2 = 0.03 ** 2
    ssim_12 = (2 * s12 + C2) / (s1 + s2 + C2)
    ssim_13 = (2 * s13 + C2) / (s1 + s3 + C2)
    if size_average:
        map_std = torch.where(torch.abs(torch.sqrt(s2.clamp(min=0))) >=
                              torch.abs(torch.sqrt(s3.clamp(min=0))),
                              ssim_12, ssim_13)
        return map_std.mean()
    return torch.where(torch.sqrt(s2.clamp(min=0)) > torch.sqrt(s3.clamp(min=0)),
                       ssim_12.abs(), ssim_13.abs()).mean(1).mean(1).mean(1)


def ssim(img1, img2, img3, window_size=11, size_average=True):
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim_three(img1, img2, img3, window, window_size, channel, size_average)


class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        ky = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kx, requires_grad=False)
        self.weighty = nn.Parameter(data=ky, requires_grad=False)

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                xi = x[i, j:j+1, :, :].unsqueeze(0)
                sx = F.conv2d(xi, self.weightx.to(x.device), padding=1)
                sy = F.conv2d(xi, self.weighty.to(x.device), padding=1)
                tensor_list.append(torch.abs(sx) + torch.abs(sy))
            batch_list.append(torch.cat(tensor_list, dim=1))
        return torch.cat(batch_list, dim=0)


# ---------------------------------------------------------------------------
# NEW — Boundary Refinement Loss (MedFusion-TransNet Section 3.4.3)
# ---------------------------------------------------------------------------

class BoundaryRefinementLoss(nn.Module):
    """
    Computes a distance-weighted loss that penalises incorrect predictions
    at and near image boundaries.

    For fusion (no segmentation ground truth), we derive the boundary map
    from the maximum-gradient regions of the source images — wherever the
    edge contrast is highest, that is the boundary that matters most.

    Formula (paper Eq. 32-33):
        L_boundary = Σ_{(i,j)∈B}  w_ij * ||F(i) - max(A(i), B(i))||²
        w_ij = 1 / (1 + dist((i,j), B))
    """
    def __init__(self, edge_thresh=0.1):
        super().__init__()
        self.sobel = Sobelxy()
        self.edge_thresh = edge_thresh

    def _get_edge_map(self, img):
        """Binary edge mask from Sobel gradient magnitude."""
        grad = self.sobel(img)           # (B, C, H, W)
        mag = grad.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        # Normalise per image
        b = mag.shape[0]
        mag_flat = mag.view(b, -1)
        mn = mag_flat.min(1)[0].view(b, 1, 1, 1)
        mx = mag_flat.max(1)[0].view(b, 1, 1, 1)
        mag_norm = (mag - mn) / (mx - mn + 1e-6)
        return (mag_norm > self.edge_thresh).float()   # (B, 1, H, W)

    def _distance_weights(self, edge_mask):
        """
        Approximate distance transform: invert edge mask and use
        max-pool to dilate, producing soft distance weights.
        Higher weight = closer to edge boundary.
        """
        # Dilate edge mask to create near-boundary region
        dilated = F.max_pool2d(edge_mask, kernel_size=5, stride=1, padding=2)
        # Weight = 1 for edge pixels, decays for non-edge
        dist_approx = dilated * 0.9 + edge_mask * 0.1 + 0.01
        return dist_approx  # (B, 1, H, W)

    def forward(self, fused, mri, ct):
        """
        Args:
            fused : (B, C, H, W)  fused output
            mri   : (B, C, H, W)  MRI source
            ct    : (B, C, H, W)  CT/PET/SPECT source (broadcast if needed)
        """
        B, C, H, W = fused.shape
        ct_exp = ct.expand(B, C, H, W)

        # Target: max of source images (same as intensity loss target)
        target = torch.max(mri, ct_exp)

        # Derive boundary weights from source images
        edge_mri = self._get_edge_map(mri)
        edge_ct  = self._get_edge_map(ct_exp)
        edge_combined = torch.max(edge_mri, edge_ct)   # (B, 1, H, W)
        w = self._distance_weights(edge_combined)       # (B, 1, H, W)

        # Weighted squared error
        err = (fused - target) ** 2                    # (B, C, H, W)
        loss = (w * err).mean()
        return loss


# ---------------------------------------------------------------------------
# NEW — Adaptive Region Loss (MedFusion-TransNet Section 3.4.2)
# ---------------------------------------------------------------------------

class AdaptiveLoss(nn.Module):
    """
    Re-weights the per-pixel intensity loss by prediction uncertainty.
    Regions where the network is uncertain (high entropy in pixel values)
    get upweighted so the network is forced to resolve ambiguities.

    Formula (paper Eq. 27-29):
        L_adaptive = Σ_i  λ_i * L(R_i)
        λ_i = γ * Uncertainty(R_i) + (1-γ) * ClassImbalance(R_i)
    """
    def __init__(self, gamma=0.5, patch_size=16):
        super().__init__()
        self.gamma = gamma
        self.patch_size = patch_size

    def _patch_uncertainty(self, fused):
        """
        Entropy-based uncertainty computed on overlapping patches via
        unfold. High variance regions = high uncertainty.
        Returns: (B, 1, H, W) uncertainty map.
        """
        p = self.patch_size
        # Use local variance as a proxy for uncertainty
        avg = F.avg_pool2d(fused, p, stride=1, padding=p // 2)
        sq_avg = F.avg_pool2d(fused ** 2, p, stride=1, padding=p // 2)
        variance = (sq_avg - avg ** 2).clamp(min=0)
        unc = variance.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        if unc.shape[-2:] != fused.shape[-2:]:
            unc = F.interpolate(
                unc, size=fused.shape[-2:], mode='bilinear', align_corners=False)
        # Normalise
        b = unc.shape[0]
        flat = unc.view(b, -1)
        mn = flat.min(1)[0].view(b, 1, 1, 1)
        mx = flat.max(1)[0].view(b, 1, 1, 1)
        return (unc - mn) / (mx - mn + 1e-6)

    def _class_imbalance(self, mri, ct):
        """
        Low-intensity regions are underrepresented — upweight them.
        """
        p = self.patch_size
        ct_exp = ct.expand_as(mri)
        avg_intensity = F.avg_pool2d(
            (mri + ct_exp) / 2, p, stride=1, padding=p // 2)
        avg_intensity = avg_intensity.mean(dim=1, keepdim=True)
        if avg_intensity.shape[-2:] != mri.shape[-2:]:
            avg_intensity = F.interpolate(
                avg_intensity, size=mri.shape[-2:], mode='bilinear', align_corners=False)
        # Inverse: dark regions get high weight
        imbalance = 1.0 - avg_intensity.clamp(0, 1)
        return imbalance  # (B, 1, H, W)

    def forward(self, fused, mri, ct):
        B, C, H, W = fused.shape
        ct_exp = ct.expand(B, C, H, W)
        target = torch.max(mri, ct_exp)

        unc = self._patch_uncertainty(fused)
        imb = self._class_imbalance(mri, ct_exp)

        # Adaptive weight per pixel
        lam = self.gamma * unc + (1 - self.gamma) * imb
        # Normalise so total weight stays constant
        lam = lam / (lam.mean() + 1e-6)

        base_loss = F.l1_loss(fused, target, reduction='none')   # (B, C, H, W)
        loss = (lam * base_loss).mean()
        return loss


# ---------------------------------------------------------------------------
# Combined loss — DiffTransFuseLoss
# Wraps the three losses with the same forward() signature as original
# Fusionloss so Fusion_model.py only needs one line changed.
# ---------------------------------------------------------------------------

class Fusionloss(nn.Module):
    """
    DiffTransFuse hybrid loss:
      L_total = α·L_int + β·L_ssim + L_grad + λ_b·L_boundary + λ_a·L_adaptive
    Default weights match DM-FNet ablation best: α=1.5, β=0.5.
    Boundary and adaptive losses are added with small weights to start.
    """
    def __init__(self, alpha=1.5, beta=0.5, lambda_boundary=0.3,
                 lambda_adaptive=0.2, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_boundary = lambda_boundary
        self.lambda_adaptive = lambda_adaptive
        self.sobelconv = Sobelxy()
        self.boundary_loss = BoundaryRefinementLoss(edge_thresh=0.1)
        self.adaptive_loss = AdaptiveLoss(gamma=gamma, patch_size=16)

    def forward(self, data, generate_img):
        image_vis = data['vis']   # MRI (Y channel)
        image_ir  = data['ir']    # CT / PET / SPECT

        B, C, W, H = image_vis.shape
        image_ir_exp = image_ir.expand(B, C, W, H)

        # --- Intensity loss (original DM-FNet, α=1.5) ---
        x_in_max = torch.max(image_vis, image_ir_exp)
        loss_in = self.alpha * F.l1_loss(generate_img, x_in_max)

        # --- Gradient loss (Sobel) ---
        y_grad   = self.sobelconv(image_vis)
        ir_grad  = self.sobelconv(image_ir_exp)
        gen_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(gen_grad, x_grad_joint)

        # --- SSIM loss (original DM-FNet, β=0.5) ---
        loss_ssim = self.beta * (1 - ssim(generate_img, image_ir_exp, image_vis))

        # --- Boundary refinement loss (NEW from MedFusion-TransNet) ---
        loss_boundary = self.boundary_loss(generate_img, image_vis, image_ir)

        # --- Adaptive region loss (NEW from MedFusion-TransNet) ---
        loss_adaptive = self.adaptive_loss(generate_img, image_vis, image_ir)

        # Combined non-intensity term. Fusion_model.py adds loss_in separately,
        # so boundary/adaptive losses must be returned here to be optimized.
        loss_grad_total = (loss_ssim + loss_grad
                           + self.lambda_boundary * loss_boundary
                           + self.lambda_adaptive * loss_adaptive)

        # Return same two values as original for Fusion_model.py compatibility
        return loss_in, loss_grad_total
