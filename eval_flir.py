"""Eval DiffV2IR on a FLIR-aligned dataset (RGB -> IR) and report metrics.

Usage:
    python eval_flir.py --dataset /kaggle/input/flir-aligned --output /kaggle/output

Seg maps (--seg-mode) are generated once and cached under <output>/seg/.
Weights are auto-downloaded on first run into --cache-dir unless a local
path is passed (--ckpt / --blip-ckpt / --sam-checkpoint).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import k_diffusion as K
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps

sys.path.append(".")
sys.path.append("./stable_diffusion")

from eval_utils import (
    SAM_VIT_B_BYTES, SAM_VIT_B_URL, build_prompt, compute_resize_dims,
    hf_or_download_flir, load_split_stems, mean_std, parse_voc_xml,
    render_boxes_to_seg, resolve_align_root, resolve_or_download, stem_to_paths,
)
from metrics.clip_similarity import ClipSimilarity
from model_utils import CFGDenoiser, load_blip_model, load_demo_image, load_model_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Eval DiffV2IR on a FLIR-aligned dataset")
    p.add_argument("--dataset", required=True, type=str, help="Path to FLIR align dataset (auto-detects align/)")
    p.add_argument("--output", required=True, type=str)
    p.add_argument("--ckpt", default="", type=str, help="Local FLIR.ckpt; empty -> download from HF")
    p.add_argument("--blip-ckpt", default="", type=str)
    p.add_argument("--sam-checkpoint", default="", type=str)
    p.add_argument("--config", default="configs/generate.yaml", type=str)
    p.add_argument("--seg-mode", default="sam", choices=["sam", "xml", "zero", "deeplab"])
    p.add_argument("--split", default="validation", type=str)
    p.add_argument("--num-samples", default=0, type=int, help="0 = all stems in split")
    p.add_argument("--visualize", default=20, type=int)
    p.add_argument("--resolution", default=512, type=int)
    p.add_argument("--steps", default=50, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--cfg-text", default=7.5, type=float)
    p.add_argument("--cfg-image", default=1.5, type=float)
    p.add_argument("--cfg-seg", default=1.5, type=float)
    p.add_argument("--cache-dir", default="./weights", type=str)
    p.add_argument("--sam-model-type", default="vit_b", type=str)
    p.add_argument("--sam-points-per-side", default=16, type=int)
    p.add_argument("--sam-max-size", default=512, type=int)
    p.add_argument("--dry-run", action="store_true", help="Process one sample then exit")
    p.add_argument("--fp16", dest="fp16", action="store_true", default=None,
                   help="Cast the diffusion model to fp16 (auto-enabled when CUDA is available).")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="Keep the diffusion model in fp32 (default on CPU).")
    return p.parse_args()


def resolve_weights(args):
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    # DiffV2IR checkpoint: local path > huggingface_hub > direct URL
    out["ckpt"] = Path(args.ckpt) if args.ckpt and Path(args.ckpt).exists() \
        else hf_or_download_flir(cache / "FLIR.ckpt")
    # BLIP caption model
    if args.blip_ckpt and Path(args.blip_ckpt).exists():
        out["blip"] = Path(args.blip_ckpt)
    else:
        from eval_utils import BLIP_BYTES, BLIP_URL
        out["blip"] = resolve_or_download(BLIP_URL, cache / "model_base_caption_capfilt_large.pth", BLIP_BYTES)
    # SAM (only for sam mode)
    out["sam"] = None
    if args.seg_mode == "sam":
        if args.sam_checkpoint and Path(args.sam_checkpoint).exists():
            out["sam"] = Path(args.sam_checkpoint)
        else:
            out["sam"] = resolve_or_download(SAM_VIT_B_URL, cache / "sam_vit_b.pth", SAM_VIT_B_BYTES)
    return out


def load_sam(checkpoint: Path, model_type: str, device: str, points_per_side: int):
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(
            "segment-anything not installed. Run `pip install segment-anything` "
            "or use --seg-mode xml/zero.") from exc
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device)
    return SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)


def sam_seg(generator, pil_rgb: Image.Image, w: int, h: int, max_size: int) -> Image.Image:
    img = pil_rgb
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    masks = generator.generate(np.array(img))
    union = np.zeros((img.size[1], img.size[0]), dtype=bool)
    for m in masks:
        union |= m["segmentation"]
    arr = (union * 255).astype(np.uint8)
    arr = np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def deeplab_seg(pil_rgb: Image.Image, w: int, h: int) -> Image.Image:
    from torchvision import models, transforms as T
    model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    x = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])(pil_rgb)
    with torch.no_grad():
        cls = model(x.unsqueeze(0).to(dev))["out"].argmax(1).squeeze(0).cpu()  # H,W
    obj = torch.isin(cls, torch.tensor([2, 7, 15]))  # bicycle, car, person (VOC)
    arr = (obj.numpy() * 255).astype(np.uint8)
    arr = np.array(Image.fromarray(arr).resize((w, h), Image.NEAREST))
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def generate_seg_map(mode, stem, align_root, rgb_path, seg_out_dir, args, sam_gen):
    """Build (or return cached) seg PNG. White objects on black, RGB."""
    seg_out_dir = Path(seg_out_dir)
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    out = seg_out_dir / f"{stem}.png"
    if out.exists():
        return out
    img = Image.open(rgb_path).convert("RGB")
    w, h = img.size
    if mode == "zero":
        seg = Image.new("RGB", (w, h), (0, 0, 0))
    elif mode == "xml":
        boxes = parse_voc_xml(stem_to_paths(align_root, stem)["ann"])
        seg = render_boxes_to_seg(boxes, w, h)
    elif mode == "deeplab":
        seg = deeplab_seg(img, w, h)
    elif mode == "sam":
        seg = sam_seg(sam_gen, img, w, h, args.sam_max_size)
    else:
        raise ValueError(f"Unknown --seg-mode {mode}")
    seg.save(out)
    return out


def _pil_to_tensor(pil: Image.Image) -> torch.Tensor:
    """(1,3,H,W) float in [0,1]."""
    return torch.tensor(np.array(pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def _kornia_ssim(a, b) -> float:
    """Mean SSIM, tolerant of kornia 0.6/0.7 vs 1.x API differences.

    kornia <= 0.7: ``ssim(..., reduction="mean")`` returns the per-image mean.
    kornia >= 1.0 : ``reduction`` was dropped and ``ssim`` returns the full
    SSIM map ``(B,C,H,W)``, so the mean must be taken explicitly.
    """
    import kornia
    try:
        out = kornia.metrics.ssim(a, b, window_size=11, reduction="mean")
    except TypeError:
        out = kornia.metrics.ssim(a, b, window_size=11)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return float(out.mean())


def compute_sample_metrics(clip_sim, lpips_fn, rgb_img, gen_img, gt_img,
                           caption, device) -> dict[str, float]:
    """Per-image CLIP (4) + SSIM + PSNR + LPIPS metrics vs GT IR."""
    import kornia
    # align GT to generated size
    if gt_img.size != gen_img.size:
        gt_img = gt_img.resize(gen_img.size, Image.LANCZOS)
    gen_t = _pil_to_tensor(gen_img).to(device)
    gt_t = _pil_to_tensor(gt_img).to(device)
    rgb_t = _pil_to_tensor(rgb_img).to(device)

    m = {}
    m["psnr"] = float(kornia.metrics.psnr(gen_t, gt_t, max_val=1.0))
    m["ssim"] = _kornia_ssim(gen_t, gt_t)
    m["lpips"] = float(lpips_fn(gen_t * 2 - 1, gt_t * 2 - 1))  # LPIPS expects [-1,1]

    text_in = [caption]
    text_out = [f"infrared image of {caption}"]
    sim_0, sim_1, sim_direction, sim_image = clip_sim(
        rgb_t, gen_t, text_in, text_out)
    m["clip_sim_0"] = float(sim_0)
    m["clip_sim_1"] = float(sim_1)
    m["clip_sim_direction"] = float(sim_direction)
    m["clip_sim_image"] = float(sim_image)
    m["clip_sim_gt"] = float(F.cosine_similarity(
        clip_sim.encode_image(gen_t), clip_sim.encode_image(gt_t)).item())
    return m


def summarize(records: list[dict]) -> dict:
    """records: list of {stem, ...metric floats}; returns mean/std summary."""
    metric_keys = [k for k in records[0] if k not in ("stem", "caption", "prompt", "gen", "rgb", "ir", "seg")]
    summary = {"num_samples": len(records), "metrics": {}}
    for k in metric_keys:
        vals = [r[k] for r in records]
        mean, std = mean_std(vals)
        summary["metrics"][k] = {"mean": mean, "std": std}
    return summary


def _concat_horizontal(imgs, gap=6, bg=(255, 255, 255)):
    h = max(im.size[1] for im in imgs)
    width = sum(im.size[0] for im in imgs) + gap * (len(imgs) - 1)
    canvas = Image.new("RGB", (width, h), bg)
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.size[0] + gap
    return canvas


def build_visualization(samples, out_dir, n=20, include_seg=True):
    """Write per-sample triplets (RGB | GT-IR | gen-IR [| seg]) + a vertical grid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in samples[:n]:
        imgs = [s["rgb"], s["ir"], s["gen"]]
        if include_seg and s.get("seg") is not None:
            imgs.append(s["seg"])
        imgs = [Image.open(p).convert("RGB") for p in imgs]
        # uniform height so rows align
        h = 256
        imgs = [im.resize((int(im.size[0] * h / im.size[1]), h), Image.LANCZOS) for im in imgs]
        row = _concat_horizontal(imgs)
        row.save(out_dir / f"{Path(s['stem']).stem}_triplet.png")
        rows.append(row)
    if rows:
        gap = 8
        w = max(r.size[0] for r in rows)
        grid_h = sum(r.size[1] for r in rows) + gap * (len(rows) - 1)
        grid = Image.new("RGB", (w, grid_h), (255, 255, 255))
        y = 0
        for r in rows:
            grid.paste(r, (0, y))
            y += r.size[1] + gap
        grid_path = out_dir / f"grid_{min(n, len(rows))}.png"
        grid.save(grid_path)
        return grid_path
    return None


def run_inference(args, model, model_wrap, model_wrap_cfg, null_token,
                  blip_model, device, rgb_path, seg_path, out_path):
    # 1) BLIP caption -> prompt
    img384 = load_demo_image(384, device, str(rgb_path))
    with torch.no_grad():
        caption = blip_model.generate(img384, sample=True, top_p=0.9,
                                      max_length=20, min_length=5)[0]
    prompt = build_prompt(caption)

    # 2) load + resize input RGB and seg (multiples of 64, aspect preserved)
    input_image = Image.open(rgb_path).convert("RGB")
    input_seg = Image.open(seg_path).convert("RGB")
    width, height = input_image.size
    new_w, new_h = compute_resize_dims(width, height, args.resolution)
    input_image = ImageOps.fit(input_image, (new_w, new_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    input_seg = ImageOps.fit(input_seg, (new_w, new_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    # 3) sample
    with torch.no_grad(), torch.autocast(device), model.ema_scope():
        cond = {
            "c_crossattn": [model.get_learned_conditioning([prompt])],
            "c_concat1": [model.encode_first_stage(rearrange(
                2 * torch.tensor(np.array(input_image)).float() / 255 - 1,
                "h w c -> 1 c h w").to(model.device)).mode()],
            "c_concat2": [model.encode_first_stage(rearrange(
                2 * torch.tensor(np.array(input_seg)).float() / 255 - 1,
                "h w c -> 1 c h w").to(model.device)).mode()],
        }
        uncond = {
            "c_crossattn": [null_token],
            "c_concat1": [torch.zeros_like(cond["c_concat1"][0])],
            "c_concat2": [torch.zeros_like(cond["c_concat2"][0])],
        }
        sigmas = model_wrap.get_sigmas(args.steps)
        extra_args = {"cond": cond, "uncond": uncond,
                      "text_cfg_scale": args.cfg_text,
                      "image_cfg_scale": args.cfg_image,
                      "seg_cfg_scale": args.cfg_seg}
        torch.manual_seed(args.seed)
        z = torch.randn_like(cond["c_concat1"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        out_img = Image.fromarray(x.type(torch.uint8).cpu().numpy())

    # 4) save
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)
    return out_img, caption, prompt


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    align_root = resolve_align_root(args.dataset)
    stems = load_split_stems(align_root, args.split)
    if not stems:
        raise RuntimeError(f"No stems found in align_{args.split}.txt under {align_root}")
    if args.num_samples > 0:
        stems = stems[:args.num_samples]
    print(f"[0/1] Loading weights")
    weights = resolve_weights(args)
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, weights["ckpt"])
    model.eval().to(device)
    use_fp16 = (args.fp16 if args.fp16 is not None else torch.cuda.is_available())
    use_fp16 = use_fp16 and torch.cuda.is_available()
    if use_fp16:
        model.half()
        print("Diffusion model cast to fp16.")
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])
    blip_model = load_blip_model(weights["blip"], device=device)
    clip_sim = ClipSimilarity().to(device)
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    sam_gen = None
    if args.seg_mode == "sam":
        sam_gen = load_sam(weights["sam"], args.sam_model_type, device, args.sam_points_per_side)

    gen_dir = Path(args.output) / "generated"
    seg_dir = Path(args.output) / "seg"

    records = []
    for i, stem in enumerate(stems):
        paths = stem_to_paths(align_root, stem)
        if not (paths["rgb"].exists() and paths["ir"].exists()):
            print(f"skip {stem}: missing rgb/ir")
            continue
        seg_path = generate_seg_map(args.seg_mode, stem, align_root, paths["rgb"],
                                    seg_dir, args, sam_gen)
        out_path = gen_dir / f"{stem}.png"
        print(f"[{i + 1}/{len(stems)}] {stem}")
        gen_img, caption, prompt = run_inference(args, model, model_wrap, model_wrap_cfg,
                                                 null_token, blip_model, device,
                                                 paths["rgb"], seg_path, out_path)
        gt_img = Image.open(paths["ir"]).convert("RGB")
        rgb_img = Image.open(paths["rgb"]).convert("RGB")
        met = compute_sample_metrics(clip_sim, lpips_fn, rgb_img, gen_img, gt_img,
                                     caption, device)
        rec = {"stem": stem, "caption": caption, "prompt": prompt,
               "gen": str(out_path), "rgb": str(paths["rgb"]),
               "ir": str(paths["ir"]), "seg": str(seg_path)}
        rec.update(met)
        records.append(rec)
        torch.cuda.empty_cache()  # drop per-image fragmentation so a 15 GB T4 survives a long run
        if args.dry_run:
            break

    # persist run manifest
    with open(Path(args.output) / "run_manifest.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"Done. Generated {len(records)} images -> {gen_dir}")

    # aggregate report
    scored = [r for r in records if "psnr" in r]
    if not scored:
        print("No scored samples; skipping metrics report.")
        return
    metrics_path = Path(args.output) / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for r in scored:
            keep = {k: r[k] for k in r if isinstance(r[k], (int, float, str))}
            f.write(json.dumps(keep) + "\n")
    summary = summarize(scored)
    with open(Path(args.output) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("==== Metrics (mean +/- std) ====")
    for k, v in summary["metrics"].items():
        print(f"  {k:20s} {v['mean']:.4f} +/- {v['std']:.4f}")

    # visualization: RGB | GT-IR | gen-IR [| seg]
    viz_dir = Path(args.output) / "visualization"
    grid_path = build_visualization(records, viz_dir, n=args.visualize)
    print(f"Visualization -> {grid_path}")


if __name__ == "__main__":
    main()
