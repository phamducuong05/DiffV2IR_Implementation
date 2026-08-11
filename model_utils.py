"""Shared model helpers for DiffV2IR inference (requires torch).

Functions moved out of infer.py so both infer.py and eval_flir.py reuse them.
"""
from __future__ import annotations

import sys

sys.path.append("./stable_diffusion")

import einops
import k_diffusion as K
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from blip_models.blip import blip_decoder
from ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale, seg_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=4)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=4)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat1": [torch.cat([cond["c_concat1"][0], cond["c_concat1"][0], uncond["c_concat1"][0], uncond["c_concat1"][0]])],
            "c_concat2": [torch.cat([cond["c_concat2"][0], cond["c_concat2"][0], cond["c_concat2"][0], uncond["c_concat2"][0]])],
        }
        out_cond, out_img_cond, out_seg_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(4)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_seg_cond) + seg_cfg_scale * (out_seg_cond - out_uncond)


def _ema_weights_present(sd) -> bool:
    """True if `sd` carries LitEma shadow buffers for the UNet.

    LitEma registers buffers under flattened names (dots removed), so a loaded
    EMA weight shows up as e.g. ``model_ema.diffusion_modelinput_blocks00weight``
    (no dots after the ``model_ema.`` prefix). Dotted ``model_ema.*`` keys would
    not have matched the flattened buffers on load and are ignored here.
    """
    return any(
        k.startswith("model_ema.") and "." not in k[len("model_ema."):]
        for k in sd
    )


def apply_ema_for_inference(model, sd) -> bool:
    """Fold the checkpoint's EMA clone into the live denoiser, then drop it.

    DiffV2IR's ``LatentDiffusion`` keeps a full ``LitEma`` clone of the UNet
    (``model.model_ema``), and ``ema_scope()`` additionally deep-clones the live
    params for the duration of each sampling call. At inference we only need one
    copy of the weights: apply the EMA values once, disable ``ema_scope`` and
    delete the clone, saving ~2x the UNet's memory. Returns True when EMA
    weights were actually applied (else the base weights are kept).
    """
    if not (getattr(model, "use_ema", False) and hasattr(model, "model_ema")):
        return False
    if _ema_weights_present(sd):
        model.model_ema.copy_to(model.model)
        print("Applied EMA weights to denoiser; dropped EMA clone (saves ~1x UNet memory).")
        applied = True
    else:
        print("Checkpoint has no model_ema weights; keeping base denoiser weights.")
        applied = False
    model.use_ema = False
    del model.model_ema
    return applied


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    apply_ema_for_inference(model, sd)
    # Free the ~8 GB checkpoint tensors now that the weights are in the model.
    del pl_sd, sd
    return model


def load_demo_image(image_size, device, img_url):
    raw_image = Image.open(img_url).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0)
    return image.to(device)


def load_blip_model(blip_path, device="cuda"):
    model = blip_decoder(pretrained=str(blip_path), image_size=384, vit='base')
    model.eval()
    return model.to(device)
