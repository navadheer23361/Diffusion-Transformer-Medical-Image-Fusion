"""Network builders for the refactored DiffTransFuse package."""

import functools
import logging

import torch
import torch.nn as nn
from torch.nn import init

from models.stage1.diffusion_unet import Stage1DUNet
from models.stage2.feature_extraction import extract_dual_unet_features
from models.stage2.fusion_head import Fusion_Head

logger = logging.getLogger("base")


# ---------------------------------------------------------------------------
# Weight initialisation helpers (unchanged from original)
# ---------------------------------------------------------------------------

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        net.apply(functools.partial(weights_init_normal, std=std))
    elif init_type == 'kaiming':
        net.apply(functools.partial(weights_init_kaiming, scale=scale))
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(f'Init [{init_type}] not implemented')


def _build_stage1_unet(unet_opt):
    """Build the local Stage 1 diffusion UNet from config values."""
    return Stage1DUNet(
        in_ch=unet_opt.get("in_channel", 1),
        base_ch=unet_opt.get("inner_channel", 32),
        time_dim=unet_opt.get("time_dim", 128),
    )


def _load_checkpoint_if_present(network, load_path, label):
    if load_path is None:
        return
    logger.info('Loading %s pretrained UNet [%s]', label, load_path)
    state = torch.load(load_path, map_location='cpu')
    if isinstance(state, dict):
        if "model_state" in state:
            state = state["model_state"]
        elif 'state_dict' in state:
            state = state['state_dict']
    network.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# UNet / Diffusion model builders
# ---------------------------------------------------------------------------

def define_UNet(opt):
    model_opt = opt['model']
    netG = _build_stage1_unet(model_opt['unet'])
    if opt['phase'] == 'train':
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

def define_Dual_UNet(opt):
    """Build independent CT and MR diffusion UNets for Stage I/II.

    Use this instead of one shared UNet when extracting Stage II diffusion
    features. The two networks share architecture by default, but the config can
    override either branch with model.unet_ct or model.unet_mr.
    """
    model_opt = opt['model']
    base_unet_opt = model_opt['unet']
    ct_unet_opt = dict(base_unet_opt)
    ct_unet_opt.update(model_opt.get('unet_ct', {}))
    mr_unet_opt = dict(base_unet_opt)
    mr_unet_opt.update(model_opt.get('unet_mr', {}))

    netG_CT = _build_stage1_unet(ct_unet_opt)
    netG_MR = _build_stage1_unet(mr_unet_opt)

    if opt['phase'] == 'train':
        init_weights(netG_CT, init_type='orthogonal')
        init_weights(netG_MR, init_type='orthogonal')

    path_opt = opt.get('path_dual_unet', {})
    _load_checkpoint_if_present(netG_CT, path_opt.get('ct_resume_state'), 'CT')
    _load_checkpoint_if_present(netG_MR, path_opt.get('mr_resume_state'), 'MR')

    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG_CT = nn.DataParallel(netG_CT)
        netG_MR = nn.DataParallel(netG_MR)

    return netG_CT, netG_MR


def extract_stage2_dual_features(netG_CT, netG_MR, data, opt):
    """Extract fusion features with modality-specific diffusion encoders.

    CT slices are stored under data['ir'] by the local dataset and MR slices
    under data['vis'] for compatibility with DM-FNet losses.
    """
    model_df_opt = opt['model_df']
    return extract_dual_unet_features(
        ct_encoder=netG_CT,
        mr_encoder=netG_MR,
        ct_image=data['ir'],
        mr_image=data['vis'],
        time_steps=model_df_opt['t'],
        feat_type=model_df_opt.get('feat_type', 'dec'),
    )



# ---------------------------------------------------------------------------
# Fusion network builder — now passes num_heads to DiffTransFuse head
# ---------------------------------------------------------------------------

def define_DFFM(opt):
    df_model_opt      = opt['model_df']
    diffusion_model_opt = opt['model']

    # num_heads from config (default 8, falls back gracefully inside the module)
    num_heads = df_model_opt.get('num_heads', 8)

    netDF = Fusion_Head(
        feat_scales=df_model_opt['feat_scales'],
        out_channels=df_model_opt['out_channels'],
        inner_channel=diffusion_model_opt['unet'].get('inner_channel', 32),
        channel_multiplier=diffusion_model_opt['unet']['channel_multiplier'],
        img_size=df_model_opt['output_cm_size'],
        time_steps=df_model_opt['t'],
        num_heads=num_heads,
        max_tokens=df_model_opt.get('max_attention_tokens', 1024),
    )

    if opt['phase'] == 'train':
        init_weights(netDF, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netDF = nn.DataParallel(netDF)
    return netDF
