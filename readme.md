# Contrast-Phys Heart Rate — rPPG via Contrastive Learning

> Remote Photoplethysmography (rPPG) heart rate estimation using unsupervised/weakly-supervised contrastive learning, with multi-scale temporal training and multi-device camera benchmarking.

Based on [Contrast-Phys](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720488.pdf) (ECCV 2022) and [Contrast-Phys+](https://ieeexplore.ieee.org/document/10440521) (TPAMI 2024).

[Paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136720488.pdf), [Poster](https://github.com/zhaodongsun/contrast-phys/releases/download/aux/0205.pdf), [Video](https://github.com/zhaodongsun/contrast-phys/releases/download/aux/0205.mp4)

![Contrast-Phys Framework](https://github.com/zhaodongsun/contrast-phys/releases/download/aux/all.png)

---

## Overview

This project implements and extends the Contrast-Phys/Contrast-Phys+ framework for camera-based heart rate estimation without requiring ground-truth PPG signals during training. Key contributions beyond the original paper:

1. **Multi-scale temporal training** (`train_multiclips.py`) — gradient accumulation across 60s/30s/20s/10s clips with dynamic stage weighting strategies (equal, curriculum, inv_loss, loss_prop, hybrid)
2. **EfficientPhysNet backbone** — 2D CNN + Temporal Shift Module (TSM), replacing the original PhysNet 3D CNN for mobile-friendly inference at 96×96 input
3. **Multi-level evaluation** (`evaluate_from_test.py`) — PSD-level (training-aligned) + HR-level (clinical MAE/RMSE) + random 10s window evaluation
4. **Multi-camera benchmarking** — comparing rPPG accuracy across Logitech C920, smartphone cameras, and RAW YUV420 recordings

---

## Repository Structure

```
├── readme.md                    # This file
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment
├── preprocess_ubfc.py           # UBFC-rPPG dataset preprocessing
├── test_preprocess.py           # Preprocessing validation
├── evaluate_from_test.py        # Multi-level evaluation (PSD + HR + 10s windows)
├── train_multiclips.py          # Multi-scale temporal training (EfficientPhysNet)
├── prep/                        # Dataset preparation utilities
│
└── contrast-phys+/              # Core model & training code
    ├── train.py                 # Original single-scale training (PhysNet 3D)
    ├── test.py                  # Testing / inference on .h5 dataset
    ├── loss.py                  # Contrastive loss (spatiotemporal)
    ├── IrrelevantPowerRatio.py  # IPR metric (power outside 40-250 BPM)
    ├── PhysNetModel.py          # PhysNet 3D CNN backbone
    ├── live_predict_webcam.py   # Real-time webcam / recorded video inference
    ├── utils_data.py            # Dataset & DataLoader utilities
    ├── utils_sig.py             # Signal processing (bandpass, FFT, HR extraction)
    ├── utils_inference.py       # Inference utilities
    ├── utils_paths.py           # Path management
    ├── Pipeline.txt             # Pipeline documentation
    │
    ├── EfficientPhysNet/        # EfficientPhysNet 2D+TSM model
    │   ├── evaluation/          # Camera benchmark evaluation scripts
    │   ├── train/               # Training configs & pretrained metadata
    │   ├── test/                # Test scripts
    │   ├── live_recorded_infer.py  # Inference on recorded videos
    │   └── run_*.sh             # Shell scripts for training/testing/benchmarking
    │
    ├── PhysNet_2D/              # PhysNet 2D variant (lightweight)
    │   ├── EfficientPhysNet.py  # Model definition
    │   ├── train.py             # Training script
    │   ├── test.py              # Testing script
    │   └── live_predict_webcam_EPN.py  # Live inference
    │
    ├── pretrained/              # Pretrained model weights (22MB)
    │   ├── supervised/          # Fully supervised (PhysNet 3D)
    │   ├── unsupervised/        # Fully unsupervised (PhysNet 3D)
    │   ├── EfficientPhysNet/    # Multi-scale trained variants
    │   │   ├── equal/           # Equal stage weighting
    │   │   ├── curriculum/      # Curriculum weighting
    │   │   ├── hybrid/          # Hybrid weighting
    │   │   ├── inv_loss/        # Inverse-loss weighting
    │   │   └── loss_prop/       # Loss-proportional weighting
    │   └── PhysNet_2D/          # PhysNet 2D pretrained
    │
    ├── results/                 # Benchmark results (tracked selectively)
    │   └── EfficientPhysNet/label_ratio_0/camera_compare/20260424_023317/
    │       ├── camera_session_results.csv
    │       ├── report_figures/  # Visualization charts
    │       ├── gt_proxy/        # Ground truth reference data
    │       ├── common_gt/       # Common GT aligned data
    │       └── per_camera/      # Per-camera inference logs
    │
    └── evaluation/              # Additional evaluation utilities
```

---

## Prerequisites

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.8+
- PyTorch ≥ 1.10 (CUDA recommended)
- OpenFace (for facial landmark extraction)
- scipy, numpy, matplotlib, h5py, sacred

---

## Dataset Preprocessing

### 1. Extract facial landmarks using OpenFace

```bash
# Linux
./FeatureExtraction -f <VideoFileName> -out_dir <dir> -2Dfp

# Windows
python -c "import os; os.system('.\\openface\\FeatureExtraction.exe -f video.avi -out_dir landmarks -2Dfp')"
```

### 2. Crop faces and create .h5 files

```bash
python preprocess_ubfc.py
```

Each .h5 file contains:
```
X.h5
├── imgs   # [N, 128, 128, 3] cropped face frames (or [N, 96, 96, 3] for EfficientPhysNet)
├── bvp    # [N] ground truth PPG signal (optional for unsupervised training)
```

**Important**: For unsupervised training (label_ratio=0), `bvp` is NOT required in training .h5 files. For test sets, `bvp` is always required for evaluation.

---

## Training

### Option A: Original single-scale training (PhysNet 3D)

```bash
cd contrast-phys+
python train.py
```

### Option B: Multi-scale temporal training (EfficientPhysNet) — Recommended

```bash
# Unsupervised, 96×96 input, equal stage weighting
python train_multiclips.py with input_size=96 lr=1e-5 label_ratio=0

# With curriculum weighting strategy
python train_multiclips.py with input_size=96 weight_strategy=curriculum

# Custom stage durations
python train_multiclips.py with stage_secs="[50,30,20,10]" input_size=96
```

**Multi-scale training stages** (gradient accumulation per iteration):

| Stage | Duration | Frames (30fps) | Weight (equal) |
|-------|----------|----------------|----------------|
| 1     | 60s      | 1800           | 0.250          |
| 2     | 30s      | 900            | 0.250          |
| 3     | 20s      | 600            | 0.250          |
| 4     | 10s      | 300            | 0.250          |

**Weight strategies**:
- `equal` — uniform weights across all stages
- `curriculum` — starts long-clip biased, shifts to short-clip focus
- `inv_loss` — stages with smaller loss get more weight
- `loss_prop` — stages with larger loss get more weight
- `hybrid` — blends inv_loss (early epochs) → curriculum (late epochs)

---

## Testing

```bash
cd contrast-phys+

# Test with specific experiment
python test.py with train_exp_num=1

# Test with best model from multi-scale training
python test.py with train_exp_num=<exp_id> e=<best_epoch>
```

Predictions saved as `.npy` files in `./results/<label_ratio>/<exp_id>/<test_id>/`.

---

## Evaluation

```bash
# Full evaluation (PSD + HR + 10s windows)
python evaluate_from_test.py <pred_dir>

# PSD-level only (aligned with training loss)
python evaluate_from_test.py --psd-only <pred_dir>

# HR-level only (clinical metric)
python evaluate_from_test.py --hr-only <pred_dir>

# Save waveform visualizations (pred vs GT)
python evaluate_from_test.py --save-viz <pred_dir>

# Exclude outlier subjects
python evaluate_from_test.py --exclude-subjects=subject4,subject25 <pred_dir>
```

**Evaluation metrics**:

| Level | Metrics | Purpose |
|-------|---------|---------|
| PSD-level | Pearson correlation, MSE on normalized PSD (40-250 BPM) | Aligned with contrastive training loss |
| HR-level | MAE, RMSE, Pearson, P5, P10 | Clinical heart rate accuracy |
| Random 10s | Same as above but on random 10s windows | Short-clip inference capability |

---

## Live / Recorded Video Inference

```bash
cd contrast-phys+

# Webcam real-time inference
python live_predict_webcam.py --train-exp-dir pretrained/unsupervised --source 0 --duration 60

# Recorded video inference
python live_predict_webcam.py --train-exp-dir pretrained/supervised --source video.mp4 --face

# EfficientPhysNet recorded inference
cd EfficientPhysNet
python live_recorded_infer.py --model ../pretrained/EfficientPhysNet/hybrid/best_model.pt --video ../../data/video.avi
```

---

## Camera Benchmark

Multi-camera comparison evaluating rPPG accuracy across different recording devices:

- **Logitech C920** (USB webcam, 30fps, 1080p)
- **Android smartphones** (various models, variable fps)
- **RAW YUV420** recordings (uncompressed reference)

Results: `contrast-phys+/results/EfficientPhysNet/label_ratio_0/camera_compare/20260424_023317/`

Key findings:
- Logitech C920 achieves significantly better MAE than smartphone cameras
- Frame rate stability (≥30fps) is critical for accurate rPPG extraction
- Videos with effective fps < 25 are filtered out as unreliable

---

## Model Architecture Comparison

| Model | Input | Params | Suitable for |
|-------|-------|--------|-------------|
| PhysNet 3D | 128×128, T=300 | ~770K | Server/GPU inference |
| EfficientPhysNet (2D+TSM) | 96×96, T=300 | ~180K | Mobile/edge deployment |
| PhysNet 2D | 96×96, T=300 | ~120K | Ultra-lightweight |

---

## Pretrained Weights

Available in `contrast-phys+/pretrained/`:

| Model | Variant | Training | File |
|-------|---------|----------|------|
| PhysNet 3D | Unsupervised | UBFC, label_ratio=0 | `unsupervised/best_model.pt` |
| PhysNet 3D | Supervised | UBFC, label_ratio=1 | `supervised/best_model.pt` |
| EfficientPhysNet | Equal weighting | UBFC, multi-scale | `EfficientPhysNet/equal/best_model.pt` |
| EfficientPhysNet | Curriculum | UBFC, multi-scale | `EfficientPhysNet/curriculum/best_model.pt` |
| EfficientPhysNet | Hybrid | UBFC, multi-scale | `EfficientPhysNet/hybrid/best_model.pt` |
| EfficientPhysNet | Inv-loss | UBFC, multi-scale | `EfficientPhysNet/inv_loss/best_model.pt` |
| EfficientPhysNet | Loss-prop | UBFC, multi-scale | `EfficientPhysNet/loss_prop/best_model.pt` |
| PhysNet 2D | Supervised | UBFC | `PhysNet_2D/best_model_supervised.pt` |
| PhysNet 2D | Unsupervised | UBFC | `PhysNet_2D/best_model_unsupervised.pt` |

---

## Key Design Decisions

1. **Frame rate**: All models assume 30fps input. Non-30fps videos should be resampled or filtered.
2. **Bandpass**: rPPG signals filtered to 0.6–4.0 Hz (36–240 BPM) for HR extraction.
3. **Contrastive loss**: Operates in frequency domain (40–250 BPM band), encouraging physiological periodicity without explicit GT.
4. **Smart peak selection**: Harmonic-aware FFT peak selection to avoid octave errors.
5. **Target device**: 96×96 input at 30fps, optimized for mobile deployment.

---

## Citation

```bibtex
@article{sun2024,
  title={Contrast-Phys+: Unsupervised and Weakly-supervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast},
  author={Sun, Zhaodong and Li, Xiaobai},
  journal={TPAMI},
  year={2024}
}

@inproceedings{sun2022contrast,
  title={Contrast-Phys: Unsupervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast},
  author={Sun, Zhaodong and Li, Xiaobai},
  booktitle={European Conference on Computer Vision},
  year={2022},
}
```

---

## License

This project builds upon the [original Contrast-Phys repository](https://github.com/zhaodongsun/contrast-phys) by Zhaodong Sun.
