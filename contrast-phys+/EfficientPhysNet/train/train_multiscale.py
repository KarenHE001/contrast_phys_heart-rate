# -*- coding: utf-8 -*-
"""
EfficientPhysNet 多尺度训练 — 从 contrast-phys+ 目录运行

运行: cd contrast-phys+ && python EfficientPhysNet/train/train_multiscale.py with input_size=96
"""
import os
import sys

_EPN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CP = os.path.dirname(_EPN)  # contrast-phys+
_EPN2D = os.path.join(_CP, "PhysNet 2D")
sys.path.insert(0, _CP)
sys.path.insert(0, _EPN2D)
os.chdir(_CP)

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler

from EfficientPhysNet import EfficientPhysNet
from loss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio
from utils_data import H5Dataset, UBFC_LU_split
from utils_paths import format_label_ratio, get_exp_root
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('epn_multiscale_train', save_git_info=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


@ex.config
def my_config():
    input_size   = 96
    S            = 2
    in_ch        = 3
    total_epoch  = 30
    lr           = 1e-5
    fs           = 30
    label_ratio  = 0
    stage_secs   = [60, 30, 20, 10]
    stage_K      = [8, 6, 5, 4]
    batch_size   = 2
    use_amp      = True
    weight_strategy = 'equal'
    test_mode    = False
    result_dir   = os.path.join(_CP, "results", "EfficientPhysNet", format_label_ratio(label_ratio))
    os.makedirs(result_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(result_dir))


def compute_weights(strategy, e, total_epoch, n_stages, stage_avg_prev):
    def _norm(v):
        s = sum(v) or 1e-9
        return [x / s for x in v]
    if strategy == 'equal':
        return [1.0 / n_stages] * n_stages
    elif strategy == 'curriculum':
        progress = e / max(total_epoch - 1, 1)
        raw = [(1 - progress) * (1 - i / max(n_stages - 1, 1)) + progress * (i / max(n_stages - 1, 1)) for i in range(n_stages)]
        return _norm(raw)
    elif strategy == 'inv_loss':
        if stage_avg_prev is None:
            return [1.0 / n_stages] * n_stages
        return _norm([1.0 / (abs(l) + 1e-8) for l in stage_avg_prev])
    elif strategy == 'loss_prop':
        if stage_avg_prev is None:
            return [1.0 / n_stages] * n_stages
        return _norm([abs(l) for l in stage_avg_prev])
    elif strategy == 'hybrid':
        progress = e / max(total_epoch - 1, 1)
        raw_curr = [(1 - progress) * (1 - i / max(n_stages - 1, 1)) + progress * (i / max(n_stages - 1, 1)) for i in range(n_stages)]
        w_curr = _norm(raw_curr)
        w_inv = [1.0 / n_stages] * n_stages if stage_avg_prev is None else _norm([1.0 / (abs(l) + 1e-8) for l in stage_avg_prev])
        return _norm([progress * wc + (1 - progress) * wi for wc, wi in zip(w_curr, w_inv)])
    else:
        raise ValueError(f"Unknown weight_strategy: '{strategy}'")


def _scan_min_frames(file_list):
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
    T_src = imgs.shape[2]
    if T_src <= T_target:
        return imgs, GT_sig
    start = int(rng.integers(0, T_src - T_target))
    return (
        imgs[:, :, start:start + T_target, :, :].contiguous(),
        GT_sig[:, start:start + T_target].contiguous(),
    )


def train_step(model, opt, scaler, imgs, GT_sig, label_flag, T_frames, stage_losses, weights, rng, use_amp):
    opt.zero_grad()
    batch_weighted_loss = 0.0
    stage_loss_vals, stage_mse_vals = [], []
    rppg = None
    for stage_idx, (T_s, loss_fn, w) in enumerate(zip(T_frames, stage_losses, weights)):
        imgs_s, GT_s = (imgs, GT_sig) if stage_idx == 0 else random_slice(imgs, GT_sig, T_s, rng)
        with autocast(enabled=use_amp):
            model_output = model(imgs_s)
            rppg = model_output[:, -1].float()
            loss, p_loss, n_loss, *_ = loss_fn(model_output.float(), GT_s, label_flag)
        scaler.scale(w * loss).backward()
        with torch.no_grad():
            mse = torch.mean((rppg - GT_s) ** 2).item()
        batch_weighted_loss += w * loss.item()
        stage_loss_vals.append(loss.item())
        stage_mse_vals.append(mse)
    scaler.step(opt)
    scaler.update()
    return batch_weighted_loss, stage_loss_vals, stage_mse_vals, rppg


def run_epoch(model, opt, scaler, dataloader, T_frames, stage_losses, weights, IPR, ex, stage_secs, epoch_idx, num_iterations, use_amp):
    model.train()
    epoch_train_losses, epoch_stage_losses = [], [[] for _ in stage_secs]
    epoch_stage_mses = [[] for _ in stage_secs]
    for it in range(num_iterations):
        for imgs, GT_sig, label_flag in dataloader:
            imgs = imgs.to(device)
            GT_sig = GT_sig.to(device)
            label_flag = label_flag.to(device)
            rng = np.random.default_rng(seed=epoch_idx * 10_000 + it)
            batch_loss, stage_vals, stage_mses, rppg = train_step(
                model, opt, scaler, imgs, GT_sig, label_flag, T_frames, stage_losses, weights, rng, use_amp)
            ipr = torch.mean(IPR(rppg.clone().detach()))
            epoch_train_losses.append(batch_loss)
            for i, (v, m) in enumerate(zip(stage_vals, stage_mses)):
                epoch_stage_losses[i].append(v)
                epoch_stage_mses[i].append(m)
            ex.log_scalar("train_weighted_loss", batch_loss)
            ex.log_scalar("train_ipr", ipr.item())
            for i, (v, m, s) in enumerate(zip(stage_vals, stage_mses, stage_secs)):
                ex.log_scalar(f"train_s{s}s_loss", v)
                ex.log_scalar(f"train_s{s}s_mse", m)
    avg = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
    avgs = [np.mean(sl) if sl else 0.0 for sl in epoch_stage_losses]
    mses = [np.mean(ml) if ml else 0.0 for ml in epoch_stage_mses]
    return avg, avgs, mses


@ex.automain
def my_main(_run, total_epoch, lr, in_ch, fs, S, input_size, stage_secs, stage_K, batch_size, use_amp,
            weight_strategy, label_ratio, test_mode, result_dir):
    exp_dir = os.path.join(result_dir, str(int(_run._id)))
    os.makedirs(exp_dir, exist_ok=True)
    assert input_size % 16 == 0
    assert len(stage_secs) == len(stage_K)
    assert stage_secs == sorted(stage_secs, reverse=True)

    T_frames = [int(fs * s) for s in stage_secs]
    n_stages = len(stage_secs)
    stage_losses = [ContrastLoss(T_s // 2, K_s, fs, high_pass=40, low_pass=250) for T_s, K_s in zip(T_frames, stage_K)]
    T_max = T_frames[0]

    train_list, val_list, test_list = UBFC_LU_split(test_mode=test_mode)
    np.save(exp_dir + '/train_list.npy', train_list)
    np.save(exp_dir + '/val_list.npy', val_list)
    np.save(exp_dir + '/test_list.npy', test_list)

    min_frames = _scan_min_frames(train_list)
    if T_max >= min_frames:
        T_max = max(min_frames - 2, T_frames[-1])
        T_frames = [min(T_s, T_max) for T_s in T_frames]

    train_dataset = H5Dataset(train_list, T_max, label_ratio)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4 if torch.cuda.is_available() else 0,
                                  pin_memory=torch.cuda.is_available(), drop_last=len(train_list) >= 2)
    val_dataloader = None
    if len(val_list) >= 2:
        val_dataset = H5Dataset(val_list, T_max, label_ratio)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=4 if torch.cuda.is_available() else 0,
                                    pin_memory=torch.cuda.is_available(), drop_last=False)

    model = EfficientPhysNet(S, in_ch=in_ch, input_size=input_size).to(device).train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)
    num_iterations = max(1, round(stage_secs[0] / (T_max / fs)))

    best_val_loss, best_epoch, stage_avg_prev = float('inf'), -1, None
    for e in range(total_epoch):
        weights = compute_weights(weight_strategy, e, total_epoch, n_stages, stage_avg_prev)
        avg_train, stage_avg, stage_mse = run_epoch(model, opt, scaler, train_dataloader,
            T_frames, stage_losses, weights, IPR, ex, stage_secs, e, num_iterations, use_amp)
        ex.log_scalar("epoch_train_loss", avg_train, step=e + 1)
        stage_avg_prev = stage_avg

        if val_dataloader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for imgs, GT_sig, label_flag in val_dataloader:
                    imgs = imgs.to(device)
                    GT_sig = GT_sig.to(device)
                    label_flag = label_flag.to(device)
                    with autocast(enabled=use_amp):
                        out = model(imgs)
                        loss, *_ = stage_losses[0](out.float(), GT_sig, label_flag)
                    val_losses.append(loss.item())
            avg_val = np.mean(val_losses) if val_losses else 0.0
            ex.log_scalar("val_loss", avg_val, step=e + 1)
            if avg_val < best_val_loss:
                best_val_loss, best_epoch = avg_val, e
                torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pt'))

        torch.save(model.state_dict(), os.path.join(exp_dir, f'epoch{e}.pt'))

    print(f"Training complete. Best: epoch{best_epoch}.pt (val={best_val_loss:.6f})")
