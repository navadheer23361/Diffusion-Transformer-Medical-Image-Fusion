"""Loss functions used by DiffTransFuse."""

from .difftrans_fusion_loss import DiffTransFuseLoss
from .fusion_loss import Fusionloss

__all__ = [
    "DiffTransFuseLoss",
    "Fusionloss",
]
