"""Torch-free helpers for the FLIR eval pipeline.

Kept dependency-light so it can be unit-tested without torch/kornia/clip.
"""
from __future__ import annotations

import math
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw

# Verified download sources (2026-08-11)
FLIR_CKPT_URL = ("https://huggingface.co/datasets/Lidong26/IR-500K/resolve/main/"
                 "IR-500k/finetuned_checkpoints/FLIR.ckpt")
FLIR_CKPT_BYTES = 7704016434
BLIP_URL = ("https://storage.googleapis.com/sfr-vision-language-research/BLIP/"
            "models/model_base_caption_capfilt_large.pth")
BLIP_BYTES = 896081425
SAM_VIT_B_URL = ("https://dl.fbaipublicfiles.com/segment_anything/"
                 "sam_vit_b_01ec64.pth")
SAM_VIT_B_BYTES = 375042383

# Noise labels present in FLIR VOC XMLs
_IGNORE_LABELS = {"FLIR", "dog"}


def build_prompt(caption: str) -> str:
    """Turn a BLIP caption into the DiffV2IR edit prompt."""
    return f"turn the visible image of {caption} into infrared"


def compute_resize_dims(width: int, height: int, resolution: int) -> tuple[int, int]:
    """Mirror infer.py: scale max dim to `resolution`, round both dims to multiples of 64."""
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    w = int((width * factor) // 64) * 64
    h = int((height * factor) // 64) * 64
    return w, h


def mean_std(vals: list[float]) -> tuple[float, float]:
    """Population mean/std; (0.0, 0.0) for empty input."""
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return mean, math.sqrt(var)


def parse_voc_xml(xml_path: Path) -> list[dict]:
    """Parse a FLIR VOC XML -> [{"label": str, "box": [xmin,ymin,xmax,ymax]}].

    Ignores the noise labels FLIR/dog and boxes with degenerate sizes.
    """
    try:
        tree = ET.parse(xml_path)
    except (ET.ParseError, FileNotFoundError):
        return []
    boxes = []
    for obj in tree.getroot().findall("object"):
        name = obj.findtext("name", default="").strip()
        if name in _IGNORE_LABELS:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if xmax > xmin and ymax > ymin:
            boxes.append({"label": name, "box": [xmin, ymin, xmax, ymax]})
    return boxes


def render_boxes_to_seg(boxes: list[dict], width: int, height: int) -> Image.Image:
    """Black RGB image with white filled boxes (white-on-black seg, like training)."""
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    for obj in boxes:
        draw.rectangle([int(v) for v in obj["box"]], fill=(255, 255, 255))
    return img


def resolve_align_root(path) -> Path:
    """If `<path>/align` is a dir, return it; otherwise treat `path` as the align dir."""
    p = Path(path)
    if (p / "align").is_dir():
        return p / "align"
    return p


def load_split_stems(align_root: Path, split: str) -> list[str]:
    """Read `align_{split}.txt` one stem per non-blank line.

    FLIR ADAS 1.3 aligned keeps the split files at the dataset root
    (`align_train.txt`, `align_validation.txt`); older layouts put them
    under `ImageSets/Main/`. Both are supported, root-first.
    """
    align_root = Path(align_root)
    candidates = [
        align_root / f"align_{split}.txt",
        align_root / "ImageSets" / "Main" / f"align_{split}.txt",
    ]
    for f in candidates:
        if f.exists():
            with open(f) as fh:
                return [line.strip() for line in fh if line.strip()]
    raise FileNotFoundError(
        f"split file align_{split}.txt not found under {align_root} "
        f"(tried {[str(c) for c in candidates]})")


def _preview_base(stem: str) -> tuple[str, str]:
    """Return (preview_stem, base_stem) whether or not stem carries the suffix."""
    if stem.endswith("_PreviewData"):
        return stem, stem[: -len("_PreviewData")]
    return stem + "_PreviewData", stem


def stem_to_paths(align_root: Path, stem: str) -> dict:
    """Map a split-file stem to FLIR file paths.

    Tolerates stems with/without the `_PreviewData` suffix and probes the
    real extension in `JPEGImages` (some FLIR builds use `.jpg`, others
    `.jpeg`); falls back to the canonical names when nothing is found.
    """
    align_root = Path(align_root)
    preview, base = _preview_base(stem)
    jpg_dir = align_root / "JPEGImages"
    ann_dir = align_root / "Annotations"
    rgb_glob = sorted(jpg_dir.glob(f"{base}_RGB.*"))
    ir_glob = sorted(jpg_dir.glob(f"{preview}.*"))
    ann_glob = sorted(ann_dir.glob(f"{preview}.*"))
    return {
        "rgb": rgb_glob[0] if rgb_glob else jpg_dir / f"{base}_RGB.jpg",
        "ir": ir_glob[0] if ir_glob else jpg_dir / f"{preview}.jpeg",
        "ann": ann_glob[0] if ann_glob else ann_dir / f"{preview}.xml",
        "stem": stem,
    }


def resolve_or_download(url: str, dest: Path,
                        expected_bytes: Optional[int] = None) -> Path:
    """Return `dest` if present and size-correct; else stream-download `url` to it.

    Uses a `.part` temp file and replaces atomically; raises if final size
    mismatches `expected_bytes`.
    """
    dest = Path(dest)
    if dest.exists():
        if expected_bytes is None or dest.stat().st_size == expected_bytes:
            return dest
        print(f"Warning: {dest.name} size mismatch, re-downloading")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    if expected_bytes is not None and dest.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"Downloaded {dest} has wrong size {dest.stat().st_size} != {expected_bytes}")
    return dest


def hf_or_download_flir(dest: Path) -> Path:
    """Download FLIR.ckpt: huggingface_hub (resume + cache) with requests fallback."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size == FLIR_CKPT_BYTES:
        return dest
    try:
        from huggingface_hub import hf_hub_download
        cached = Path(hf_hub_download(
            repo_id="Lidong26/IR-500K",
            filename="IR-500k/finetuned_checkpoints/FLIR.ckpt",
            repo_type="dataset", cache_dir=str(dest.parent)))
        if cached != dest:
            shutil.copyfile(cached, dest)
        return dest
    except Exception as exc:  # no huggingface_hub / network to HF API
        print(f"huggingface_hub failed ({exc}); falling back to direct download")
        return resolve_or_download(FLIR_CKPT_URL, dest, FLIR_CKPT_BYTES)
