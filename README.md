# Hybrid Diffusion-Based X-ray Image Enhancement

A hybrid **classical + deep-learning** image processing pipeline that cleans up noisy
portable X-ray images: it removes noise while keeping anatomical edges sharp, boosts
local contrast so faint structures become visible, and reports quantitative quality
metrics — all through a desktop GUI, a CLI, or a Google Colab notebook.

**Pipeline:** Anisotropic Diffusion → DnCNN Deep Denoiser → CLAHE → Optional Sharpening → Quality Evaluation

![Before and after enhancement](docs/images/before_after.png)

---

## Why this exists

Portable/handheld X-ray units are convenient but noisy — low photon counts (quantum
noise) and electronic sensor noise both show up as grain that can obscure subtle
anatomical detail. Naively blurring the image to remove that noise also blurs the
edges a clinician actually needs to see. This project combines a **classical,
edge-aware PDE filter** with a **learned CNN denoiser** and **adaptive contrast
enhancement**, in an order specifically chosen so each stage cleans up what the
previous one left behind, without one stage undoing another's work.

![Pipeline stage-by-stage output](docs/images/pipeline_stages.png)

---

## How it works — the pipeline

```
Input Image
   │
   ▼
Image Validation           →  rejects corrupt/unusable files early
   │
   ▼
Grayscale Conversion        →  X-rays are single-channel; keeps every stage consistent
   │
   ▼
Noise Analysis               →  estimates noise level (wavelet-based sigma estimator)
   │
   ▼
Anisotropic Diffusion        →  Perona-Malik PDE: smooths flat regions, protects edges
   │
   ▼
DnCNN Deep Denoiser           →  learned residual CNN cleans up what diffusion leaves
   │                              behind (skips gracefully if no pretrained weights)
   ▼
CLAHE Contrast Enhancement     →  boosts local contrast now that noise is suppressed
   │
   ▼
Optional Edge Enhancement       →  unsharp mask/Laplacian sharpening, auto-skipped if
   │                                 it would amplify noise instead of detail
   ▼
Quality Evaluation                →  PSNR, SSIM, MSE, RMSE, Entropy, Edge Preservation Index
   │
   ▼
Enhanced Output Image
```

### Why this order specifically
Denoising happens **before** contrast enhancement on purpose — running CLAHE on a
still-noisy image would stretch the noise right along with real detail. Sharpening
runs **last** and is auto-guarded: if it would amplify noise more than it sharpens
real edges, the pipeline discards it and keeps the pre-sharpen result instead of
silently degrading the image.

---

## Module breakdown

| File | Role |
|---|---|
| `config.py` | Every tunable parameter (diffusion iterations/kappa/lambda, CLAHE clip limit/tile size, sharpening amount) in one place |
| `image_loader.py` | Reads PNG/JPG/JPEG/BMP/TIFF, rescales 16-bit images, converts to grayscale, validates before processing |
| `preprocessing.py` | Normalizes to float [0,1], optionally downsizes large images, estimates noise level |
| `anisotropic_diffusion.py` | **Perona-Malik anisotropic diffusion** — an edge-aware PDE smoothing filter (full math in the docstring) |
| `deep_denoiser.py` | **DnCNN** (17-layer residual-learning CNN, PyTorch) — learned denoising; explains why DnCNN over CBDNet, skips cleanly with no pretrained weights |
| `clahe.py` | **CLAHE** — tile-wise, contrast-limited adaptive histogram equalization |
| `enhancement.py` | Optional sharpening (unsharp mask / Laplacian) with an automatic noise-amplification guard |
| `evaluation.py` | Computes PSNR, SSIM, MSE, RMSE, Shannon entropy, and Edge Preservation Index |
| `pipeline.py` | Orchestrates every stage above and returns all intermediate images + timings + metrics |
| `main.py` | Command-line interface |
| `gui.py` | Tkinter desktop GUI (upload, live parameter sliders, process, save, metrics readout) |
| `train_dncnn.py` | Optional script to train your own `models/dncnn.pth` checkpoint |
| `Hybrid_Xray_Enhancement_Colab.ipynb` | Same pipeline, wrapped for Google Colab (upload/display instead of a desktop window) |

---

## Algorithms at a glance

**Anisotropic Diffusion (Perona-Malik, 1990)** solves `dI/dt = div(c·∇I)`, where the
conductance `c` is high in flat regions (smooths freely) and low near strong edges
(smoothing suppressed). This is what makes it different from a Gaussian blur, which
smooths edges and flat regions equally.

**DnCNN (Zhang et al., 2017)** is a residual-learning CNN: it predicts the *noise*
in the image rather than the clean image directly (`output = input − predicted_noise`),
which converges faster and is a well-documented, widely reproduced architecture. It
was chosen over CBDNet because CBDNet needs a camera-realistic synthetic noise
generator and a second noise-estimation sub-network with no ready pretrained
weights for grayscale medical images.

**CLAHE** equalizes small image tiles independently (rather than the whole image at
once) so both dense bone and faint soft tissue get appropriately enhanced, and clips
each tile's histogram before equalizing to avoid amplifying noise in flat regions.

---

## Quality metrics (example run, synthetic X-ray phantom)

| Metric | Value |
|---|---|
| Detected noise level | high (σ ≈ 27.7) |
| PSNR | 17.42 dB |
| SSIM | 0.4249 |
| MSE / RMSE | 1177.27 / 34.31 |
| Entropy (original → enhanced) | 7.13 → 7.35 bits |
| Edge Preservation Index | 0.4453 |

*(This test used an intentionally heavily-noised synthetic phantom with no
pretrained DnCNN checkpoint loaded, so numbers on real, less-degraded clinical
images — or with a trained DnCNN checkpoint — are expected to be higher.)*

---

## Getting started

```bash
# Desktop
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python gui.py                     # GUI
python main.py sample_images/synthetic_xray_noisy.png   # CLI
```

Or open `Hybrid_Xray_Enhancement_Colab.ipynb` directly in Google Colab — no local
setup required.

Full installation, CLI reference, and user manual: see [`README.md`](README.md).
Theory, math, and full results write-up: see [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md).

---

## Scope note

This is an educational/decision-support image-enhancement tool, **not** a validated
or certified diagnostic device.

