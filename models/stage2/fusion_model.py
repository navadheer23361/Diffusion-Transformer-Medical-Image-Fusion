"""Legacy Stage 2 fusion model wrapper kept for package compatibility."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from models.losses.fusion_loss import Fusionloss
from models.stage2.feature_extraction import validate_dual_features
from models.stage2.network_builders import define_DFFM

logger = logging.getLogger("base")


class DFFM:
    """Small compatibility wrapper around the refactored Stage 2 fusion head."""

    def __init__(self, opt):
        self.opt = opt
        device_name = opt.get("device")
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_name)
        self.netDF = define_DFFM(opt).to(self.device)
        self.loss_func = Fusionloss(
            alpha=1.5,
            beta=0.5,
            lambda_boundary=opt["train"].get("lambda_boundary", 0.3),
            lambda_adaptive=opt["train"].get("lambda_adaptive", 0.2),
            gamma=opt["train"].get("drgo_gamma", 0.5),
        ).to(self.device)

        self.log_dict = OrderedDict()
        self.loss_all = []
        self.loss_in = []
        self.loss_grad = []
        self.alpha = 1.0
        self.pred_fused = None
        self.pred_rgb = None
        self.data = None
        self.feats = None

        if opt["phase"] == "train":
            self.netDF.train()
            params = list(self.netDF.parameters())
            opt_type = opt["train"]["optimizer"]["type"]
            lr = opt["train"]["optimizer"]["lr"]
            if opt_type == "adam":
                self.optDF = torch.optim.Adam(params, lr=lr)
            elif opt_type == "adamw":
                self.optDF = torch.optim.AdamW(params, lr=lr)
            else:
                raise NotImplementedError(f"Optimizer [{opt_type}] not implemented")
            self.exp_lr_scheduler_netDF = None
        else:
            self.netDF.eval()
            self.optDF = None
            self.exp_lr_scheduler_netDF = None

        self.load_network()
        self.print_network()

    def feed_data(self, feats, data):
        """Cache one batch of diffusion features and paired source data."""
        self.feats = validate_dual_features(feats, time_steps=self.opt["model_df"]["t"])
        self.data = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in data.items()
        }

    def optimize_parameters(self):
        """Run one optimization step."""
        if self.optDF is None:
            raise RuntimeError("optimize_parameters() requires phase='train'.")
        self.optDF.zero_grad(set_to_none=True)
        self.pred_fused = self.netDF(self.data["ir"], self.data["vis"], self.feats, hard_gate=False)
        self.pred_rgb = self.pred_fused
        loss_in, loss_grad = self.loss_func(data=self.data, generate_img=self.pred_fused)
        loss_fs = loss_in + self.alpha * loss_grad
        loss_fs.backward()
        self.optDF.step()
        self.loss_all.append(loss_fs.item())
        self.loss_in.append(loss_in.item())
        self.loss_grad.append(loss_grad.item())

    def update_loss(self):
        """Aggregate recent loss values into the current log dictionary."""
        self.log_dict["l_all"] = float(np.average(self.loss_all)) if self.loss_all else 0.0
        self.log_dict["l_in"] = float(np.average(self.loss_in)) if self.loss_in else 0.0
        self.log_dict["l_grad"] = float(np.average(self.loss_grad)) if self.loss_grad else 0.0
        self.loss_all = []
        self.loss_in = []
        self.loss_grad = []

    def test(self):
        """Run one evaluation forward pass."""
        self.netDF.eval()
        with torch.no_grad():
            self.pred_fused = self.netDF(self.data["ir"], self.data["vis"], self.feats, hard_gate=True)
            self.pred_rgb = self.pred_fused
            loss_in, loss_grad = self.loss_func(data=self.data, generate_img=self.pred_fused)
            loss_fs = loss_in + loss_grad
            self.loss_all.append(loss_fs.item())
            self.loss_in.append(loss_in.item())
            self.loss_grad.append(loss_grad.item())
        if self.opt["phase"] == "train":
            self.netDF.train()

    def get_current_log(self):
        """Return the latest aggregated loss dictionary."""
        return self.log_dict

    def get_current_visuals(self):
        """Return the latest fused output and source inputs."""
        out_dict = OrderedDict()
        out_dict["pred_fused"] = self.pred_fused
        out_dict["pred_rgb"] = self.pred_fused
        out_dict["gt_mr"] = self.data["vis"][:, 0:1, :, :]
        out_dict["gt_ct"] = self.data["ir"][:, 0:1, :, :]
        return out_dict

    def print_network(self):
        """Log the Stage 2 fusion head structure."""
        network = self.netDF.module if isinstance(self.netDF, nn.DataParallel) else self.netDF
        num_params = sum(param.numel() for param in network.parameters())
        logger.info("Network structure: %s, parameters: %s", network.__class__.__name__, f"{num_params:,}")

    def save_network(self, epoch, is_best_model=False):
        """Save Stage 2 weights to the configured checkpoint directory."""
        checkpoint_dir = Path(self.opt["path"]["checkpoint"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        gen_path = checkpoint_dir / f"df_model_E{epoch}_gen.pth"
        network = self.netDF.module if isinstance(self.netDF, nn.DataParallel) else self.netDF
        state_dict = {key: value.detach().cpu() for key, value in network.state_dict().items()}
        torch.save(state_dict, gen_path)
        if is_best_model:
            torch.save(state_dict, checkpoint_dir / "best_df_model_gen.pth")
        logger.info("Saved model -> [%s]", gen_path)

    def load_network(self):
        """Load optional pretrained Stage 2 weights."""
        load_path = self.opt["path_df"]["resume_state"]
        if load_path:
            logger.info("Loading pretrained model [%s]", load_path)
            network = self.netDF.module if isinstance(self.netDF, nn.DataParallel) else self.netDF
            state = torch.load(load_path, map_location=self.device)
            network.load_state_dict(state, strict=False)

    def _update_lr_schedulers(self):
        """Advance the optional learning-rate scheduler."""
        if self.exp_lr_scheduler_netDF is not None:
            self.exp_lr_scheduler_netDF.step()
