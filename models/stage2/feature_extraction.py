"""
Utilities for Stage II dual-encoder feature flow.

Stage II must pass CT slices through the trained CT diffusion UNet and MR
slices through the trained MR diffusion UNet. The fusion head consumes the
resulting dictionary:

    {
        'CT': [features_at_t5, features_at_t10, features_at_t20],
        'MR': [features_at_t5, features_at_t10, features_at_t20],
    }

Each item in the lists is the multi-scale feature container returned by the
corresponding diffusion encoder.
"""

from __future__ import annotations

import torch


def pack_dual_features(ct_features, mr_features):
    """Pack separate CT/MR encoder outputs for the fusion model."""
    if ct_features is mr_features:
        raise ValueError(
            'CT and MR features are the same object. Stage II must use the '
            'trained CT encoder for CT and the trained MR encoder for MR.')
    return {'CT': ct_features, 'MR': mr_features}


def validate_dual_features(feats, time_steps=None):
    """Validate that Stage II received separate CT and MR feature streams."""
    if not isinstance(feats, dict):
        raise TypeError('feats must be a dict with keys CT and MR')
    if 'CT' not in feats or 'MR' not in feats:
        raise KeyError(
            "Stage II fusion expects feats['CT'] from the CT encoder and "
            "feats['MR'] from the MR encoder.")
    if feats['CT'] is feats['MR']:
        raise ValueError(
            "feats['CT'] and feats['MR'] point to the same object. Use "
            'separate CT/MR diffusion encoders before fusion.')
    if time_steps is not None:
        expected = len(time_steps)
        if len(feats['CT']) != expected or len(feats['MR']) != expected:
            raise ValueError(
                'Expected {} timestep feature sets, got CT={} MR={}'.format(
                    expected, len(feats['CT']), len(feats['MR'])))
    return feats


def _extract_one_timestep(encoder, image, t, feat_type='dec'):
    """Call the feature-extraction API exposed by the parent DM-FNet encoder."""
    module = encoder.module if hasattr(encoder, 'module') else encoder
    for method_name in ('extract_features', 'get_feats', 'get_features'):
        if hasattr(module, method_name):
            method = getattr(module, method_name)
            try:
                return method(image, t=t, feat_type=feat_type)
            except TypeError:
                try:
                    return method(image, t)
                except TypeError:
                    return method(image)

    try:
        return module(image, t=t, feat_type=feat_type, return_feats=True)
    except TypeError:
        pass

    raise AttributeError(
        'Could not extract diffusion features from {}. Add one of these methods '
        'to the parent DM-FNet encoder: extract_features(image, t, feat_type), '
        'get_feats(image, t), or get_features(image, t).'.format(
            module.__class__.__name__))


def extract_dual_unet_features(ct_encoder, mr_encoder, ct_image, mr_image,
                               time_steps, feat_type='dec'):
    """Extract Stage II features with the corresponding modality encoders."""
    ct_features = [
        _extract_one_timestep(ct_encoder, ct_image, t, feat_type=feat_type)
        for t in time_steps
    ]
    mr_features = [
        _extract_one_timestep(mr_encoder, mr_image, t, feat_type=feat_type)
        for t in time_steps
    ]
    return pack_dual_features(ct_features, mr_features)


def stage1_features_for_t(model, diffusion, image, timestep: int, device):
    """Extract one timestep's Stage 1 features from a clean image batch."""
    image = image.to(device)
    timestep_tensor = torch.full(
        (image.size(0),), int(timestep), device=device, dtype=torch.long
    )
    noise = torch.randn_like(image)
    noisy_image = diffusion.q_sample(image, timestep_tensor, noise)
    module = model.module if hasattr(model, "module") else model
    if hasattr(module, "feature_pyramid"):
        return module.feature_pyramid(noisy_image, timestep_tensor)
    return _extract_one_timestep(module, noisy_image, timestep_tensor)


def extract_stage1_dual_features(
    ct_model,
    mr_model,
    diffusion,
    batch: dict,
    time_steps,
    device,
):
    """Extract CT and MR feature streams from the frozen Stage 1 encoders."""
    ct_image = batch["ir"].to(device)
    mr_image = batch["vis"].to(device)
    ct_features = [
        stage1_features_for_t(ct_model, diffusion, ct_image, timestep, device)
        for timestep in time_steps
    ]
    mr_features = [
        stage1_features_for_t(mr_model, diffusion, mr_image, timestep, device)
        for timestep in time_steps
    ]
    return pack_dual_features(ct_features, mr_features)
