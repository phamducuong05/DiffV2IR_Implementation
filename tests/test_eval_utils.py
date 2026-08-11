import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval_utils import (
    build_prompt, compute_resize_dims, mean_std, parse_voc_xml,
    render_boxes_to_seg, resolve_align_root, load_split_stems,
    stem_to_paths, resolve_or_download,
)


def test_build_prompt():
    assert build_prompt("a car") == "turn the visible image of a car into infrared"


def test_compute_resize_dims_multiples_of_64():
    w, h = compute_resize_dims(640, 512, 512)
    assert w % 64 == 0 and h % 64 == 0
    assert (w, h) == (512, 448)  # mirrors infer.py output for 640x512


def test_mean_std_empty():
    assert mean_std([]) == (0.0, 0.0)


def test_mean_std_values():
    mean, std = mean_std([2.0, 4.0, 6.0])
    assert mean == pytest.approx(4.0)
    assert std == pytest.approx(1.63299, rel=1e-3)


def test_parse_voc_xml_skips_ignored(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text(
        "<annotation><object><name>person</name><bndbox>"
        "<xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object>"
        "<object><name>FLIR</name></object>"
        "<object><name>dog</name><bndbox>"
        "<xmin>0</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>"
        "</annotation>"
    )
    boxes = parse_voc_xml(xml)
    assert len(boxes) == 1
    assert boxes[0]["label"] == "person"
    assert boxes[0]["box"] == [1.0, 2.0, 10.0, 20.0]


def test_render_boxes_to_seg_white_on_black():
    from PIL import Image
    img = render_boxes_to_seg([{"label": "person", "box": [10, 10, 30, 40]}], 64, 64)
    assert isinstance(img, Image.Image)
    assert img.getpixel((0, 0)) == (0, 0, 0)
    assert img.getpixel((20, 25)) == (255, 255, 255)


def test_resolve_align_root_detects_align(tmp_path):
    (tmp_path / "align").mkdir()
    assert resolve_align_root(tmp_path) == tmp_path / "align"
    assert resolve_align_root(tmp_path / "align") == tmp_path / "align"


def test_load_split_stems(tmp_path):
    split_dir = tmp_path / "ImageSets" / "Main"
    split_dir.mkdir(parents=True)
    split = split_dir / "align_validation.txt"
    split.write_text("FLIR_00001_PreviewData\n\nFLIR_00002_PreviewData\n")
    assert load_split_stems(tmp_path, "validation") == [
        "FLIR_00001_PreviewData", "FLIR_00002_PreviewData"]


def test_stem_to_paths(tmp_path):
    p = stem_to_paths(tmp_path, "FLIR_00001_PreviewData")
    assert p["rgb"] == tmp_path / "JPEGImages" / "FLIR_00001_RGB.jpg"
    assert p["ir"] == tmp_path / "JPEGImages" / "FLIR_00001_PreviewData.jpeg"
    assert p["ann"] == tmp_path / "Annotations" / "FLIR_00001_PreviewData.xml"


def test_resolve_or_download_correct_size_no_download(tmp_path):
    dest = tmp_path / "f.ckpt"
    dest.write_bytes(b"x" * 10)
    out = resolve_or_download("http://unused", dest, expected_bytes=10)
    assert out == dest


def test_resolve_or_download_wrong_size_replaces(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"y" * 20)
    dest = tmp_path / "f.ckpt"
    dest.write_bytes(b"x" * 5)  # wrong size -> triggers download
    out = resolve_or_download(src.as_uri(), dest, expected_bytes=20)
    assert out == dest
    assert dest.stat().st_size == 20
