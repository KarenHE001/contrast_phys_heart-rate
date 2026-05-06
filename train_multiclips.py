# -*- coding: utf-8 -*-
"""
train_multiscale.py
───────────────────
Multi-scale temporal training using Option B gradient accumulation.

Four stages per iteration, all derived from ONE 60s batch load:
  Stage 1 — 60s  (T=1800) full clip   weight = 1800/3600 = 0.500
  Stage 2 — 30s  (T= 900) random slice weight =  900/3600 = 0.250
  Stage 3 — 20s  (T= 600) random slice weight =  600/3600 = 0.167
  Stage 4 — 10s  (T= 300) random slice weight =  300/3600 = 0.083

For each iteration:
  opt.zero_grad()
  for stage in [60s, 30s, 20s, 10s]:
      out = model(clip_stage)          # different T each time
      loss = ContrastLoss_stage(out)   # scalar
      (weight * loss).backward()       # ACCUMULATE gradients
  opt.step()                           # ONE coherent update

Shorter clips are sliced from the same 60s tensor already in memory —
no extra data loading. Slicing is seeded per (epoch, iteration) for
exact reproducibility.

Examples:
  # 96x96 model
  python train_multiscale.py with input_size=96 lr=1e-5

  # 64x64 model
  python train_multiscale.py with input_size=64 lr=2e-5

  # Adjust longest stage if your videos are shorter than 60 s:
  python train_multiscale.py with stage_secs="[50,30,20,10]" input_size=96

  # Supervised (10 % labeled data)
  python train_multiscale.py with input_size=96 label_ratio=0.1
"""

import cv2                          # noqa: F401 (kept for dataset compat)
import matplotlib.pyplot as plt     # noqa: F401
import numpy as np
import os
import sys
import h5py                         # noqa: F401 (used by H5Dataset)
import torch
from torch.cuda.amp import autocast, GradScaler

from EfficientPhysNet import EfficientPhysNet
from loss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio

from utils_data import *            # H5Dataset, UBFC_LU_split, etc.
from utils_sig import *
from utils_paths import format_label_ratio, get_exp_root
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('multiscale_train', save_git_info=False)

# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


# ─────────────────────────────────────────────────────────────
# Sacred config
# ─────────────────────────────────────────────────────────────
@ex.config
def my_config():
    # ── Model ─────────────────────────────────────────────────
    input_size   = 96             # spatial H=W: 96 or 64 (must be divisible by 16)
    S            = 2              # ST-rPPG spatial grid (2×2)
    in_ch        = 3              # input channels: 3=RGB, 1=NIR

    # ── Optimiser ─────────────────────────────────────────────
    total_epoch  = 30
    lr           = 1e-5           # 1e-5 for 96×96; try 2e-5 for 64×64

    # ── Dataset ───────────────────────────────────────────────
    fs           = 30             # TODO: set to your camera frame rate
    label_ratio  = 0              # 0 = fully unsupervised

    # ── Multi-scale stages ────────────────────────────────────
    # stage_secs[0] is the LONGEST stage — the DataLoader loads this many
    # seconds per clip.  MUST be <= min video length in your dataset.
    # Your dataset videos are ~51–67 s → 50 s is the safe maximum.
    # If all videos are >=60 s (verified), change stage_secs[0] to 60.
    stage_secs   = [60, 30, 20, 10]   # seconds per stage (longest first)

    # K = number of rPPG samples drawn per spatial position inside ContrastLoss.
    # Longer clips can support more samples; recommended: 4 for ≤10s, scale up.
    stage_K      = [8,  6,  5,  4]    # K per stage (index matches stage_secs)

    # ── DataLoader ─────────────────────────────────────────────
    batch_size   = 2              # reduce to 1 if CUDA OOM with long clips

    # ── Mixed precision ───────────────────────────────────────
    use_amp      = True           # fp16 activations: ~50% less GPU memory

    # ── Stage weight strategy ────────────────────────────────
    # 'equal'      : fixed [1/N, ...] all epochs  (default / baseline)
    # 'curriculum' : starts long-clip biased, shifts to short-clip by final epoch
    # 'inv_loss'   : stages with SMALLER loss get MORE weight (boost weak stages)
    # 'loss_prop'  : stages with LARGER  loss get MORE weight (reinforce strong stages)
    # 'hybrid'     : blends inv_loss (early) → curriculum (late)
    weight_strategy = 'equal'

    # ── Experiment bookkeeping ────────────────────────────────
    test_mode    = False
    result_dir   = get_exp_root(label_ratio)
    os.makedirs(result_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(result_dir))


# ─────────────────────────────────────────────────────────────
# Weight strategies
# ─────────────────────────────────────────────────────────────
def compute_weights(strategy, e, total_epoch, n_stages, stage_avg_prev):
    """Return normalized stage weights for epoch e.

    Args:
        strategy       : 'equal' | 'curriculum' | 'inv_loss' | 'loss_prop' | 'hybrid'
        e              : current epoch index (0-based)
        total_epoch    : total number of epochs
        n_stages       : number of stages
        stage_avg_prev : list[float] of previous epoch's mean ContrastLoss per stage
                         (None for epoch 0)
    Returns:
        list[float] of length n_stages, summing to 1.0
    """
    def _norm(v):
        s = sum(v) or 1e-9
        return [x / s for x in v]

    if strategy == 'equal':
        return [1.0 / n_stages] * n_stages

    elif strategy == 'curriculum':
        # Epoch 0: weight concentrated on long clips  [0.50, 0.33, 0.17, 0.00]
        # Midpoint: equal                             [0.25, 0.25, 0.25, 0.25]
        # Last epoch: weight concentrated on short   [0.00, 0.17, 0.33, 0.50]
        progress = e / max(total_epoch - 1, 1)   # 0.0 → 1.0
        raw = []
        for i in range(n_stages):
            sp = i / max(n_stages - 1, 1)        # 0=longest, 1=shortest
            raw.append((1 - progress) * (1 - sp) + progress * sp)
        return _norm(raw)

    elif strategy == 'inv_loss':
        # Stages with SMALLER |loss| (harder) get MORE weight
        if stage_avg_prev is None:
            return [1.0 / n_stages] * n_stages   # equal on epoch 0
        inv = [1.0 / (abs(l) + 1e-8) for l in stage_avg_prev]
        return _norm(inv)

    elif strategy == 'loss_prop':
        # Stages with LARGER |loss| (easier/stronger) get MORE weight
        if stage_avg_prev is None:
            return [1.0 / n_stages] * n_stages   # equal on epoch 0
        mag = [abs(l) for l in stage_avg_prev]
        return _norm(mag)

    elif strategy == 'hybrid':
        # Early epochs: inv_loss dominates (boost struggling stages)
        # Late  epochs: curriculum dominates (shift to short clips)
        progress = e / max(total_epoch - 1, 1)
        # curriculum component
        raw_curr = []
        for i in range(n_stages):
            sp = i / max(n_stages - 1, 1)
            raw_curr.append((1 - progress) * (1 - sp) + progress * sp)
        w_curr = _norm(raw_curr)
        # inverse-loss component
        if stage_avg_prev is None:
            w_inv = [1.0 / n_stages] * n_stages
        else:
            inv = [1.0 / (abs(l) + 1e-8) for l in stage_avg_prev]
            w_inv = _norm(inv)
        # blend: alpha=0 → all inv_loss (epoch 0),  alpha=1 → all curriculum (last epoch)
        alpha = progress
        blended = [alpha * wc + (1 - alpha) * wi for wc, wi in zip(w_curr, w_inv)]
        return _norm(blended)

    else:
        raise ValueError(f"Unknown weight_strategy: '{strategy}'. "
                         "Choose: equal | curriculum | inv_loss | loss_prop | hybrid")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _scan_min_frames(file_list):
    """Read ALL .h5 files and return the minimum frame count.
    Used to auto-cap T_max so H5Dataset never gets a clip longer than the
    shortest video in the dataset.
    """
    import h5py as _h5
    min_f = float('inf')
    for path in file_list:
        try:
            with _h5.File(path, 'r') as f:
                n = min(f['imgs'].shape[0], f['bvp'].shape[0])
                min_f = min(min_f, n)
        except Exception:
            pass
    return int(min_f) if min_f != float('inf') else 1800


def random_slice(imgs, GT_sig, T_target, rng):
    """Return a (T_target,) slice from a (2, 3, T_src, H, W) / (2, T_src) pair.

    Args:
        imgs    : (2, C, T_src, H, W) float32 tensor on device
        GT_sig  : (2, T_src) float32 tensor on device
        T_target: desired clip length in frames
        rng     : np.random.Generator (seeded)
    """
    T_src = imgs.shape[2]
    if T_src <= T_target:
        return imgs, GT_sig
    start = int(rng.integers(0, T_src - T_target))
    return (
        imgs[:, :, start:start + T_target, :, :].contiguous(),
        GT_sig[:, start:start + T_target].contiguous(),
    )


def train_step(model, opt, scaler, imgs, GT_sig, label_flag,
               T_frames, stage_losses, weights, rng, use_amp):
    """One gradient-accumulation step over all stages for a single batch.

    Returns:
        batch_weighted_loss : float — sum of weighted stage losses
        stage_loss_vals     : list[float] — raw ContrastLoss per stage
        stage_mse_vals      : list[float] — MSE(rppg, GT_sig) per stage
        rppg                : Tensor (2, T_last) — last-stage rPPG (for IPR)
    """
    opt.zero_grad()
    batch_weighted_loss = 0.0
    stage_loss_vals = []
    stage_mse_vals  = []
    rppg = None

    for stage_idx, (T_s, loss_fn, w) in enumerate(
            zip(T_frames, stage_losses, weights)):

        imgs_s, GT_s = (imgs, GT_sig) if stage_idx == 0 \
            else random_slice(imgs, GT_sig, T_s, rng)

        with autocast(enabled=use_amp):
            model_output = model(imgs_s)
            rppg = model_output[:, -1].float()  # (2, T_s), fp32 for loss
            loss, p_loss, n_loss, *_ = loss_fn(
                model_output.float(), GT_s, label_flag)

        scaler.scale(w * loss).backward()

        # MSE between predicted rPPG and GT signal (diagnostic; 0 when unsupervised)
        with torch.no_grad():
            mse = torch.mean((rppg - GT_s) ** 2).item()

        batch_weighted_loss += w * loss.item()
        stage_loss_vals.append(loss.item())
        stage_mse_vals.append(mse)

    scaler.step(opt)
    scaler.update()
    return batch_weighted_loss, stage_loss_vals, stage_mse_vals, rppg


def run_epoch(model, opt, scaler, dataloader, T_frames, stage_losses, weights,
              IPR, ex, stage_secs, epoch_idx, num_iterations, use_amp):
    """Run one full training epoch.

    Returns:
        avg_weighted_loss : float
        stage_avgs        : list[float] — mean ContrastLoss per stage
        stage_mse_avgs    : list[float] — mean MSE per stage
    """
    model.train()
    epoch_train_losses = []
    epoch_stage_losses = [[] for _ in stage_secs]
    epoch_stage_mses   = [[] for _ in stage_secs]

    for it in range(num_iterations):
        for imgs, GT_sig, label_flag in dataloader:
            imgs       = imgs.to(device)
            GT_sig     = GT_sig.to(device)
            label_flag = label_flag.to(device)

            seed = epoch_idx * 10_000 + it
            rng  = np.random.default_rng(seed=seed)
            print(f"  [iter seed={seed}] ", end='', flush=True)

            batch_loss, stage_vals, stage_mses, rppg = train_step(
                model, opt, scaler, imgs, GT_sig, label_flag,
                T_frames, stage_losses, weights, rng, use_amp)

            ipr = torch.mean(IPR(rppg.clone().detach()))
            epoch_train_losses.append(batch_loss)
            for i, (v, m) in enumerate(zip(stage_vals, stage_mses)):
                epoch_stage_losses[i].append(v)
                epoch_stage_mses[i].append(m)

            # Sacred logging
            ex.log_scalar("train_weighted_loss", batch_loss)
            ex.log_scalar("train_ipr",           ipr.item())
            for i, (v, m, s) in enumerate(zip(stage_vals, stage_mses, stage_secs)):
                ex.log_scalar(f"train_s{s}s_loss", v)
                ex.log_scalar(f"train_s{s}s_mse",  m)

    avg  = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
    avgs = [np.mean(sl) if sl else 0.0 for sl in epoch_stage_losses]
    mses = [np.mean(ml) if ml else 0.0 for ml in epoch_stage_mses]
    return avg, avgs, mses


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
@ex.automain
def my_main(_run,
            total_epoch, lr, in_ch, fs, S, input_size,
            stage_secs, stage_K, batch_size, use_amp,
            weight_strategy,
            label_ratio, test_mode, result_dir):

    exp_dir = os.path.join(result_dir, str(int(_run._id)))
    os.makedirs(exp_dir, exist_ok=True)

    # ── Sanity checks ─────────────────────────────────────────
    assert input_size % 16 == 0, f"input_size={input_size} must be divisible by 16"
    assert (input_size // 16) // S >= 1, f"input_size={input_size} too small for S={S}"
    assert len(stage_secs) == len(stage_K), "stage_secs and stage_K must have same length"
    assert stage_secs == sorted(stage_secs, reverse=True), "stage_secs must be in descending order"

    # ── Header ───────────────────────────────────────────────
    T_frames = [int(fs * s) for s in stage_secs]
    total_T  = sum(T_frames)
    n_stages = len(stage_secs)
    weights  = compute_weights(weight_strategy, 0, total_epoch, n_stages, None)

    print("\n" + "=" * 60)
    print("Multi-Scale Temporal Training (Option B)")
    print("=" * 60)
    print(f"  Experiment ID   : {int(_run._id)}")
    print(f"  Result dir      : {exp_dir}")
    print(f"  input_size      : {input_size}×{input_size}  S={S}  in_ch={in_ch}")
    print(f"  batch_size      : {batch_size}")
    print(f"  use_amp         : {use_amp}  (fp16 activations)")
    print(f"  weight_strategy : {weight_strategy}")
    print(f"  device          : {device}")
    if test_mode:
        print("  ⚠️  TEST MODE — using minimal data")
    print()
    print(f"  {'Stage':<8} {'secs':>6} {'T':>6} {'delta_t':>8} {'K':>4}  {'weight':>8}")
    for i, (s, T_s, K_s, w) in enumerate(zip(stage_secs, T_frames, stage_K, weights)):
        dt = T_s // 2
        print(f"  Stage {i+1:<2}  {s:>6}s  {T_s:>5}  {dt:>8}  {K_s:>4}  {w:>8.4f}")
    print(f"  {'total':>8}        {total_T:>5}                  {sum(weights):>8.4f}  (equal weights: 1/{n_stages})")
    print("=" * 60 + "\n")
    sys.stdout.flush()

    # ── Build per-stage ContrastLoss instances ────────────────
    # Each stage needs its own loss with matching delta_t and K.
    stage_losses = []
    for T_s, K_s in zip(T_frames, stage_K):
        delta_t_s = T_s // 2
        stage_losses.append(
            ContrastLoss(delta_t_s, K_s, fs, high_pass=40, low_pass=250)
        )

    T_max = T_frames[0]  # longest stage — DataLoader clips

    # ── Dataset split ─────────────────────────────────────────
    print("=== Initialising dataset ===")
    # TODO: replace UBFC_LU_split with your own split function
    train_list, val_list, test_list = UBFC_LU_split(test_mode=test_mode)
    print(f"  Train: {len(train_list)}  Val: {len(val_list)}  Test: {len(test_list)}")
    np.save(exp_dir + '/train_list.npy', train_list)
    np.save(exp_dir + '/val_list.npy',   val_list)
    np.save(exp_dir + '/test_list.npy',  test_list)

    # ── Auto-cap T_max to actual video length ─────────────────
    # Scan a few files so T_max never exceeds the shortest video.
    min_frames = _scan_min_frames(train_list)
    if T_max >= min_frames:
        # Subtract 2 so np.random.choice(img_length - T) always has range >= 1
        T_max_capped = max(min_frames - 2, T_frames[-1])  # at least shortest stage
        print(f"  ⚠️  T_max={T_max} >= min_video={min_frames} frames → capping to {T_max_capped}")
        T_max = T_max_capped
        # Rebuild stage frames; weights will be recomputed at epoch start
        T_frames = [min(T_s, T_max) for T_s in T_frames]
        total_T  = sum(T_frames)
        print(f"  Adjusted stages: {list(zip([str(s)+'s' for s in stage_secs], T_frames))}")
    sys.stdout.flush()

    # ── DataLoader (loads T_max frames per sample) ────────────
    # H5Dataset randomly samples a T_max-length window each call.
    # Your videos must be at least T_max+1 frames long.
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory  = torch.cuda.is_available()
    drop_last   = len(train_list) >= 2

    train_dataset    = H5Dataset(train_list, T_max, label_ratio)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
    )
    val_dataloader = None
    if len(val_list) >= 2:
        val_dataset    = H5Dataset(val_list, T_max, label_ratio)
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        )
    print(f"  DataLoader clips: T_max={T_max} frames ({stage_secs[0]} s @ {fs} fps)")
    sys.stdout.flush()

    # ── Model & optimiser ────────────────────────────────────
    print(f"\nBuilding EfficientPhysNet(S={S}, in_ch={in_ch}, input_size={input_size})...")
    model = EfficientPhysNet(S, in_ch=in_ch, input_size=input_size).to(device).train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  ({n_params*4/1e6:.2f} MB float32)")

    opt    = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)
    IPR    = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # ── num_iterations: how many times we pull a batch per epoch ──
    # With T_max=1800 (60 s) and video_length=60 s → 1 iteration.
    # With T_max=1500 (50 s) and video_length=60 s → 1 iteration.
    video_length_secs = stage_secs[0]  # assume dataset videos ≈ longest stage
    num_iterations    = max(1, round(video_length_secs / (T_max / fs)))

    # ── Training loop ────────────────────────────────────────
    best_val_loss  = float('inf')
    best_epoch     = -1
    stage_avg_prev = None   # previous epoch's stage losses (for dynamic strategies)

    for e in range(total_epoch):
        # Recompute weights at start of every epoch (epoch 0 uses None prev)
        weights = compute_weights(weight_strategy, e, total_epoch, n_stages, stage_avg_prev)
        print(f"\n=== Epoch {e+1}/{total_epoch}  [{weight_strategy}]  "
              f"weights=[{', '.join(f'{w:.3f}' for w in weights)}] ===")
        sys.stdout.flush()

        avg_train, stage_avg, stage_mse = run_epoch(
            model, opt, scaler, train_dataloader,
            T_frames, stage_losses, weights,
            IPR, ex, stage_secs, epoch_idx=e,
            num_iterations=num_iterations, use_amp=use_amp)

        ex.log_scalar("epoch_train_loss", avg_train, step=e + 1)
        stage_avg_prev = stage_avg   # save for next epoch's weight update

        # ── Per-stage breakdown ───────────────────────────────
        total_weighted = sum(w * l for w, l in zip(weights, stage_avg)) or 1e-9
        print(f"  Weighted loss: {avg_train:.6f}")
        print(f"  {'Stage':<10} {'secs':>5} {'loss':>12} {'MSE':>12} {'contrib%':>10}")
        print(f"  {'-'*52}")
        for i, (s, T_s, lv, mv, w) in enumerate(
                zip(stage_secs, T_frames, stage_avg, stage_mse, weights)):
            contrib = (w * lv / total_weighted * 100) if total_weighted != 0 else 0.0
            print(f"  Stage {i+1:<4}  {s:>4}s  {lv:>12.6f}  {mv:>12.6f}  {contrib:>9.2f}%")
        print(f"  {'-'*52}")
        sys.stdout.flush()

        # ── Validation ───────────────────────────────────────
        if val_dataloader is not None:
            model.eval()
            val_losses = []
            print("  Validating...")
            with torch.no_grad():
                val_loss_fn = stage_losses[0]
                for imgs, GT_sig, label_flag in val_dataloader:
                    imgs       = imgs.to(device)
                    GT_sig     = GT_sig.to(device)
                    label_flag = label_flag.to(device)
                    with autocast(enabled=use_amp):
                        out      = model(imgs)
                        loss, *_ = val_loss_fn(out.float(), GT_sig, label_flag)
                    val_losses.append(loss.item())

            avg_val = np.mean(val_losses) if val_losses else 0.0
            ex.log_scalar("val_loss", avg_val, step=e + 1)
            print(f"  Val loss (stage1 {stage_secs[0]}s): {avg_val:.6f}")
            sys.stdout.flush()

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch    = e
                best_path     = os.path.join(exp_dir, 'best_model.pt')
                torch.save(model.state_dict(), best_path)
                print(f"  ✅ New best model saved (val={best_val_loss:.6f})")
                sys.stdout.flush()
        else:
            print("  Skipping validation (no val set)")
            sys.stdout.flush()

        # ── Save epoch checkpoint ─────────────────────────────
        ckpt = os.path.join(exp_dir, f'epoch{e}.pt')
        torch.save(model.state_dict(), ckpt)
        print(f"  Saved: {ckpt}")
        sys.stdout.flush()

    # ── Training summary ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Experiment ID  : {int(_run._id)}")
    print(f"  Result dir     : {exp_dir}")
    print(f"  input_size     : {input_size}×{input_size}")
    stage_str = " + ".join(f"{s}s" for s in stage_secs)
    print(f"  Stages trained : {stage_str}")
    print(f"  Saved checkpoints: epoch0.pt ~ epoch{total_epoch-1}.pt")
    if val_dataloader is not None and best_epoch >= 0:
        print(f"  Best model     : best_model.pt  (Epoch {best_epoch+1}, val={best_val_loss:.6f})")
        print(f"\nNext step:")
        print(f"  python test.py with train_exp_num={int(_run._id)} e={best_epoch}")
    else:
        print(f"\nNext step:")
        print(f"  python test.py with train_exp_num={int(_run._id)} e={total_epoch-1}")
    print("=" * 60 + "\n")
