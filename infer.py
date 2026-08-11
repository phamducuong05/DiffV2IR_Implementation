from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser
import os
import k_diffusion as K
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import shutil
import requests
import torch
import json
import os



sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config

import json

from model_utils import CFGDenoiser, load_model_from_config, load_demo_image, load_blip_model
from eval_utils import resolve_or_download, BLIP_URL, BLIP_BYTES


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--blip-ckpt", default="", type=str,
                        help="Path to BLIP caption model. Empty -> auto-download to --cache-dir")
    parser.add_argument("--cache-dir", default="./weights", type=str,
                        help="Where to download weights when --blip-ckpt is empty")
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", default="turn the RGB image into the infrared one",type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--cfg-seg", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    #os.makedirs('/home/jovyan/.cache/torch/hub/checkpoints/')
    #shutil.copy("checkpoint_liberty_with_aug.pth","/home/jovyan/.cache/torch/hub/checkpoints/")
    

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    blip_path = args.blip_ckpt
    if not blip_path or not os.path.exists(blip_path):
        blip_path = resolve_or_download(BLIP_URL, os.path.join(args.cache_dir, "model_base_caption_capfilt_large.pth"), BLIP_BYTES)
    blip_model = load_blip_model(blip_path)
    seed = random.randint(0, 100000) if args.seed is None else args.seed
    for root, dirs, files in os.walk(args.input):
        for file in files:
            image = load_demo_image(image_size=384, device='cuda',img_url=os.path.join(root,file))
            with torch.no_grad():
                caption = blip_model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
            args.edit = "turn the visible image of "+caption[0]+" into infrared"
            input_image = Image.open(os.path.join(args.input,file)).convert("RGB")
            input_seg = Image.open(os.path.join(args.input+"_seg",file.split(".")[0]+".png")).convert("RGB")
            width, height = input_image.size
            factor = args.resolution / max(width, height)
            factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
            width = int((width * factor) // 64) * 64
            height = int((height * factor) // 64) * 64
            input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
            input_seg = ImageOps.fit(input_seg, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

            if args.edit == "":
                input_image.save(os.path.join(args.output,file))
                return

            with torch.no_grad(), autocast("cuda"), model.ema_scope():
                cond = {}
                cond["c_crossattn"] = [model.get_learned_conditioning([args.edit])]
                input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                input_seg = 2 * torch.tensor(np.array(input_seg)).float() / 255 - 1
                input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                input_seg = rearrange(input_seg, "h w c -> 1 c h w").to(model.device)
                cond["c_concat1"] = [model.encode_first_stage(input_image).mode()]
                cond["c_concat2"] = [model.encode_first_stage(input_seg).mode()]

                uncond = {}
                uncond["c_crossattn"] = [null_token]
                uncond["c_concat1"] = [torch.zeros_like(cond["c_concat1"][0])]
                uncond["c_concat2"] = [torch.zeros_like(cond["c_concat2"][0])]

                sigmas = model_wrap.get_sigmas(args.steps)

                extra_args = {
                    "cond": cond,
                    "uncond": uncond,
                    "text_cfg_scale": args.cfg_text,
                    "image_cfg_scale": args.cfg_image,
                    "seg_cfg_scale": args.cfg_seg,
                }
                torch.manual_seed(seed)
                z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]
                z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                x = model.decode_first_stage(z)
                x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                x = 255.0 * rearrange(x, "1 c h w -> h w c")
                edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
            edited_image.save(os.path.join(args.output,file))
    


if __name__ == "__main__":
    main()
