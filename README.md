# Learning Where It Matters: Responsible and Interpretable Text-to-Image Generation with Background Consistency

[![TMLR](https://img.shields.io/badge/TMLR-2026-blue.svg)](https://openreview.net/forum?id=sCOJGbJwAJ)
[![OpenReview](https://img.shields.io/badge/OpenReview-paper-8c1b13.svg)](https://openreview.net/forum?id=sCOJGbJwAJ)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**Authors:**
[Sayedmoslem Shokrolahi](https://openreview.net/profile?id=~Sayedmoslem_Shokrolahi1),
[Jae-Mo Kang](https://openreview.net/profile?id=~Jae-Mo_Kang1),
[Il-Min Kim](https://openreview.net/profile?id=~Il-Min_Kim1)

**Venue:** Transactions on Machine Learning Research (TMLR), May 2026
**Paper:** https://openreview.net/forum?id=sCOJGbJwAJ

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Moslem-Sh21/learning-where-it-matters/blob/main/notebooks/inference_fair_demo.ipynb)

---

## Overview

This repository contains the official implementation of our TMLR 2026 paper. The method learns small concept vectors that are injected into the **mid-block (bottleneck)** of the SDXL UNet — *where it matters* for high-level semantic control — while preserving background structure across the original and concept-modified images that share the same seed.

Each learned concept is **named and interpretable** (e.g. `female`, `male`, `young`, `old`, ...), letting practitioners turn a specific attribute on or off at inference time without retraining the underlying generator.

### Method at a glance

- **Bottleneck Concept Module.** A learnable embedding table of shape `[num_concepts, mid_channels]` (1280 channels for SDXL). At inference, a one-hot concept selector picks a row and adds it as a channel-wise bias to the UNet mid-block activations.
- **Background consistency.** A wavelet-domain operation re-injects low-/high-frequency components from the original generation, so only the targeted attribute changes — the surrounding scene stays close to the original.
- **Same seed → same scene.** Comparing the original and concept-modified outputs at the same seed isolates the effect of the concept vector.

---

## Repository structure

```
learning-where-it-matters/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── checkpoints/
│   ├── concept_female.pt        # pretrained concept module — "female" attribute
│   ├── concept_male.pt          # pretrained concept module — "male"   attribute
│   └── concept_dict.json        # concept name → index mapping (9 concepts)
├── notebooks/
│   └── inference_fair_demo.ipynb   # Colab demo
├── config.py                    # argparse for training / inference
├── train.py                     # trains the BottleneckConceptModule
├── inference.py                 # generates 4 variants per seed
├── data_generation.py           # SDXL data generation for training
└── utils_data.py                # dataset + dataloader utilities
```

---

## Installation

```bash
git clone https://github.com/Moslem-Sh21/learning-where-it-matters.git
cd learning-where-it-matters
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.0+, and a single GPU with at least 24 GB VRAM for training (an A100 / H100 is recommended). Inference runs comfortably on a Colab T4 in fp16.

---

## Quickstart — Colab demo

The fastest way to reproduce a result is the demo notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Moslem-Sh21/learning-where-it-matters/blob/main/notebooks/inference_fair_demo.ipynb)

The notebook clones this repo (so it picks up `checkpoints/` automatically), loads SDXL from Hugging Face, attaches the female/male concept modules, and generates a side-by-side `original | concept` example for two fixed seeds.

---

## Training from scratch

### 1. Generate training images

```bash
python data_generation.py \
    --prompt "a photo of a woman" \
    --output_dir datasets_SDXL_female \
    --num_samples 2000 \
    --seed 42 \
    --steps 50
```

This produces `image_XXXX.jpg`, `labels.json`, `concept_dict.json`, and `metadata.json` in the output directory.

### 2. Train the concept module

```bash
accelerate launch train.py \
    --train_data_dir datasets_SDXL_female \
    --output_dir exps_female_sdxl \
    --train_batch_size 4 \
    --num_train_epochs 20 \
    --learning_rate 1e-2 \
    --seed 42
```

The training script saves `concept_final.pt` (rename to `concept_<attribute>.pt` and drop into `checkpoints/`) along with a loss curve.

---

## Inference

```bash
python inference.py \
    --output_dir checkpoints \
    --prompt "a photo of a doctor" \
    --concept female \
    --fp16
```

This produces four image variants per seed under `images_out/`:

| Variant | Description |
|---|---|
| `original/` | Vanilla SDXL output (no concept, no wavelet) |
| `concept_only/` | Concept vector injected at mid-block |
| `wavelet_only/` | Wavelet background re-injection only |
| `concept_wavelet/` | Both — full method |

---

## Pretrained checkpoints

The `checkpoints/` directory ships with two ready-to-use concept modules:

| File | Concept | Trained on |
|---|---|---|
| `concept_female.pt` | `female` (index 0) | SDXL-generated images of women |
| `concept_male.pt` | `male` (index 1) | SDXL-generated images of men |

`concept_dict.json` lists the full concept vocabulary. Indices not covered by the released checkpoints (`young`, `old`, `white-race`, `black-race`, `anti-sexual`, `smile`, `jump`) can be trained following the recipe above.

---

## Citation

If you use this code or the released checkpoints, please cite:

```bibtex
@article{shokrolahi2026learning,
  title  = {Learning Where It Matters: Responsible and Interpretable Text-to-Image Generation with Background Consistency},
  author = {Shokrolahi, Sayedmoslem and Kang, Jae-Mo and Kim, Il-Min},
  journal= {Transactions on Machine Learning Research},
  year   = {2026},
  month  = {5},
  url    = {https://openreview.net/forum?id=sCOJGbJwAJ}
}
```

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

Built on top of [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and the [Hugging Face `diffusers`](https://github.com/huggingface/diffusers) library.
