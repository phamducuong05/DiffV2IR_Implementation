# Design: Eval DiffV2IR trên dataset FLIR (chạy trên Kaggle)

**Ngày:** 2026-08-11
**Repo đích:** `https://github.com/phamducuong05/DiffV2IR_Implementation.git` (branch `main`)

## 1. Mục tiêu

Sửa code trong repo DiffV2IR để:
1. Nhận input là **path dataset FLIR-aligned** (mount từ Kaggle, hoặc bất kỳ path nào).
2. Chạy **inference RGB → IR** bằng model DiffV2IR finetune trên FLIR.
3. **Eval** ảnh IR sinh ra bằng metrics: CLIP (trong repo) + SSIM + PSNR + LPIPS (so với ảnh IR thật).
4. **Visualize 20 ảnh** (RGB gốc | IR thật | IR sinh) để đánh giá bằng mắt.
5. Lưu ảnh IR sinh ra vào `--output` (trên Kaggle: `/kaggle/output`).

User chạy trên Kaggle: clone repo → `pip install -r requirements.txt` → `python eval_flir.py --dataset <path>`.

## 2. Dataset — FLIR ADAS Aligned (VOC format)

```
<dataset_root>/
└── align/
    ├── JPEGImages/
    │   ├── FLIR_XXXXX_PreviewData.jpeg   ← ảnh IR (ground truth, thermal, lưu dạng RGB)
    │   ├── FLIR_XXXXX_RGB.jpg            ← ảnh RGB (input, căn chỉnh với IR)
    ├── Annotations/
    │   ├── FLIR_XXXXX_PreviewData.xml    ← VOC XML (object: person/car/bicycle)
    └── ImageSets/Main/
        ├── align_train.txt
        └── align_validation.txt          ← split dùng cho eval (mặc định)
```

- `stem = "FLIR_XXXXX_PreviewData"`
- IR  = `JPEGImages/{stem}.jpeg`
- RGB = `JPEGImages/{stem.replace('_PreviewData','')}_RGB.jpg`
- Ann = `Annotations/{stem}.xml`

Script tự nhận diện: nếu path truyền vào chứa thư mục `align/` thì dùng `<path>/align`, ngược lại coi path đó chính là thư mục `align`.

## 3. Kiến trúc & files

**Approach A — script eval tự chứa.** Chỉ thêm/sửa 4 file:

| File | Loại | Nội dung |
|---|---|---|
| `model_utils.py` | mới | Code dùng chung: `load_model_from_config`, `CFGDenoiser` (3-way), `load_demo_image` (preprocess BLIP), `build_prompt(caption)`, hàm tải/download weight (`resolve_or_download`) |
| `eval_flir.py` | mới | Script eval chính (mọi thứ) |
| `infer.py` | patch nhỏ | Bỏ path BLIP cứng `/data/wld/...` → thêm `--blip-ckpt`; dùng `model_utils` |
| `requirements.txt` | patch | Thêm `lpips`, `segment-anything`, `huggingface_hub` |

Dùng lại (không sửa): `configs/generate.yaml`, `metrics/clip_similarity.py` (`ClipSimilarity`), `blip_models/blip.py` (`blip_decoder`).

## 4. CLI `eval_flir.py`

```bash
python eval_flir.py \
  --dataset /kaggle/input/flir-aligned \   # required; tự nhận diện align/
  --output /kaggle/output \                # required; nơi lưu ảnh + metrics + visualize
  --ckpt ""                                # path FLIR.ckpt; trống → tự tải từ HF
  --blip-ckpt ""                           # path BLIP; trống → tự tải từ URL đã verify
  --sam-checkpoint ""                      # path SAM; trống → tự tải (chỉ khi seg-mode=sam)
  --config configs/generate.yaml
  --seg-mode sam                           # sam | xml | zero | deeplab
  --split validation
  --num-samples 0                          # 0 = chạy hết split
  --visualize 20
  --resolution 512
  --steps 50
  --seed 0
  --cfg-text 7.5
  --cfg-image 1.5
  --cfg-seg 1.5
  --cache-dir ./weights                    # nơi tải weight
  --sam-model-type vit_b
  --sam-points-per-side 16
  --sam-max-size 512                       # downscale ảnh trước khi chạy SAM
  --dry-run                                # chỉ chạy 1 ảnh để smoke-test
```

**Cây output:**
```
<output>/
├── generated/FLIR_XXXXX.png      ← IR model sinh ra
├── seg/FLIR_XXXXX.png            ← seg map (cache, chạy 1 lần)
├── visualization/
│   ├── grid_20.png               ← 20 hàng × [RGB | IR thật | IR sinh] (+seg nếu bật)
│   └── FLIR_XXXXX_triplet.png
├── metrics.jsonl                 ← metric từng ảnh
└── summary.json                  ← mean ± std
```

## 5. Nguồn weights (đã verify URL)

Ưu tiên path local; nếu trống/không tồn tại → tự tải xuống `--cache-dir`.

| Weight | Kích thước | URL |
|---|---|---|
| FLIR.ckpt (DiffV2IR finetune FLIR) | 7.7 GB | `https://huggingface.co/datasets/Lidong26/IR-500K/resolve/main/IR-500k/finetuned_checkpoints/FLIR.ckpt` |
| BLIP caption | 896 MB | `https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth` |
| SAM ViT-B | 375 MB | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` |
| CLIP ViT-L/14 | ~1 GB | tự tải qua `clip.load("ViT-L/14", download_root=cache)` |

- FLIR.ckpt dùng `huggingface_hub.hf_hub_download(repo_id="Lidong26/IR-500K", filename="IR-500k/finetuned_checkpoints/FLIR.ckpt", repo_type="dataset")` (resume + cache), fallback `requests`.
- Check kích thước file sau tải (tránh file 0 byte/hỏng).
- FLIR.ckpt chứa UNet + VAE + CLIP text encoder → **không cần** `--vae-ckpt` riêng.
- Link BLIP cũ trong repo (404) được thay bằng link đã verify; `capfilt_large` tương thích `blip_decoder(image_size=384, vit='base')`.

## 6. Sinh seg map (mặc định `sam`)

- Load `SamAutomaticMaskGenerator` từ `segment_anything` (`vit_b`).
- Mỗi ảnh RGB → `masks = sam_generator.generate(rgb_np)` → hợp nhất: trắng nếu thuộc ≥1 mask, còn lại đen → RGB PNG lưu `<output>/seg/{stem}.png` (white-on-black, đúng format train).
- Cache: file seg tồn tại → bỏ qua (chạy lại nhanh).
- Tốc độ: downscale ảnh ≤ `--sam-max-size` (512), `--sam-points-per-side 16` → ~2–4s/ảnh trên T4; với full val (~1000 ảnh) seg chạy ~1h, chỉ 1 lần do cache.
- `--seg-mode xml`: tô trắng các bbox từ XML (fallback không cần thêm dep).
- `--seg-mode zero`: ảnh đen.
- `--seg-mode deeplab`: `torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)` → trắng nếu pixel thuộc class object (person=15, car=7, bicycle=2 trong VOC).
- `import segment_anything` lazy: nếu chưa cài + chọn `sam` → báo rõ hướng dẫn.

## 7. Pipeline inference (từng ảnh)

```
RGB  → BLIP caption → prompt "turn the visible image of {caption} into infrared"
RGB  → resize bội số 64 (giữ tỉ lệ, tái dùng công thức infer.py) → [-1,1] → VAE encode → c_concat1
seg  → resize cùng kích thước → [-1,1] → VAE encode → c_concat2
text → model.get_learned_conditioning(prompt) → c_crossattn
z    = randn_like(c_concat1) * sigmas[0]  (torch.manual_seed(seed))
x    = sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args)
IR   = decode_first_stage(x) → clamp [0,1] → *255 → lưu PNG
```

uncond: `c_crossattn = null_token`, `c_concat1/c2 = zeros_like`. Batch = 1 (giữ logic gốc). CFG scale: `--cfg-text 7.5`, `--cfg-image 1.5`, `--cfg-seg 1.5` (mặc định như infer.py).

## 8. Metrics (per-image, so với GT IR)

Resize ảnh IR thật về đúng kích thước ảnh sinh rồi tính.

| Metric | Nguồn | Hướng tốt |
|---|---|---|
| SSIM | `kornia.metrics.ssim` | ↑ |
| PSNR | `kornia.metrics.psnr` | ↑ |
| LPIPS | `lpips.LPIPS(net='alex')` (tự tải weight) | ↓ |
| CLIP sim_0, sim_1, sim_direction, sim_image | `metrics/clip_similarity.py` `ClipSimilarity(image_0=RGB, image_1=gen, text_0=caption, text_1="infrared image of {caption}")` | ↑ |
| CLIP sim_gt | cosine(gen, GT IR) qua `ClipSimilarity.encode_image` | ↑ |

Kết quả: `metrics.jsonl` (từng ảnh) + `summary.json` (mean ± std) + bảng in console.

## 9. Visualization (20 ảnh)

- `visualization/grid_20.png`: 20 hàng × [RGB | IR thật | IR sinh] (+cột seg).
- `visualization/{stem}_triplet.png`: từng mẫu.
- FLIR lưu IR dạng RGB JPEG → hiển thị trực tiếp.

## 10. Xử lý lỗi

- Thiếu weight → liệt kê file cần; thiếu internet + thiếu weight local → liệt kê 4 file cần upload lên Kaggle.
- `--num-samples > len(split)` → clamp + cảnh báo.
- Split trống → lỗi rõ ràng.
- `segment-anything` chưa cài khi `--seg-mode sam` → hướng dẫn `pip install segment-anything` hoặc đổi `--seg-mode xml/zero`.
- `--dry-run`: chạy 1 ảnh rồi dừng → smoke-test nhanh trên Kaggle.

## 11. Chạy trên Kaggle

1. Kaggle notebook bật **Internet** (cần tải weight + CLIP/BLIP).
2. `!git clone https://github.com/phamducuong05/DiffV2IR_Implementation.git && cd DiffV2IR_Implementation`
3. `!pip install -r requirements.txt`
4. Mount dataset FLIR-aligned → path `/kaggle/input/...`.
5. `!python eval_flir.py --dataset /kaggle/input/flir-aligned --output /kaggle/output`
   - Lần đầu tải ~9 GB weight (7.7 + 0.9 + 0.4 + CLIP) vào `./weights` (nằm trong `/kaggle/working`).
   - **Khuyên:** upload FLIR.ckpt + BLIP + SAM lên Kaggle dataset, truyền `--ckpt /kaggle/input/...` để không tải lại.
6. Download `/kaggle/output` (hoặc `/kaggle/working`) sau khi chạy xong.

## 12. Git push (theo yêu cầu user)

Sau khi code + test xong:
```bash
rm -rf .git
git init
git remote add origin https://github.com/phamducuong05/DiffV2IR_Implementation.git
git branch -M main
git add -A && git commit -m "..." 
git push -u origin main
```
(Thực hiện bằng script, không dùng interactive.)

## 13. Test kế hoạch

- **Local (không GPU):** import-check 2 file mới; test `--seg-mode xml` và `zero` trên fixture FLIR nhỏ tự tạo; test auto-detect `align/`.
- **`--dry-run` trên Kaggle:** chạy 1 ảnh end-to-end để smoke-test trước khi chạy full.
- **Full:** chạy `--num-samples 0` trên val set.
