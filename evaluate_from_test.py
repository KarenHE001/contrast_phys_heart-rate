# -*- coding: utf-8 -*-
"""
评估测试结果 - PSD 级 + HR 级

【方式1】PSD 级 - 与 loss.py CalculateNormPSD 一致，与训练目标对齐:
  - 频段: 40-250 BPM，归一化 PSD 的 Pearson、MSE

【方式2】HR 级 (README 流程) - 主指标:
  - rppg/bvp -> butter_bandpass -> hr_fft -> MAE, RMSE, P5, P10

【可视化】--save-viz 保存 rppg (predicted) vs bvp (ground truth) 波形图，无指标计算。
"""
import numpy as np
import os
import sys

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scipy.stats import pearsonr
from scipy.fft import fft
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils_paths import get_exp_root
from utils_sig import (
    hr_fft, butter_bandpass,
    hr_fft_zp, hr_fft_parabolic,
    compute_fft_peaks, select_hr_from_peaks
)

import glob

# 解析命令行参数
use_psd_eval = True        # PSD 级（与训练目标对齐）
use_hr_eval = True         # HR 级（README 流程，主指标）
use_harmonics_removal = True
verbose = False
hr_method = 'parabolic'
smart_peak = True
save_viz = False           # 是否保存 waveform 可视化（pred vs GT，无指标）
exclude_subjects = []      # --exclude-subjects subject4,subject25 排除后算 MAE

# 解析 flags
for a in sys.argv[1:]:
    if a == '--psd-only':
        use_psd_eval, use_hr_eval = True, False
    elif a == '--hr-only':
        use_psd_eval, use_hr_eval = False, True
    elif a == '--save-viz':
        save_viz = True
    elif a in ('--no-harmonics', '-n'):
        use_harmonics_removal = False
    elif a in ('--verbose', '-v'):
        verbose = True
    elif a == '--no-smart-peak':
        smart_peak = False
    elif a == '--parabolic':
        hr_method = 'parabolic'
    elif a == '--zp':
        hr_method = 'zp'
    elif a == '--original':
        hr_method = 'original'  # README 明确提到的 hr_fft
    elif a.startswith('--exclude-subjects='):
        exclude_subjects = [s.strip() for s in a.split('=', 1)[1].split(',') if s.strip()]

# 路径：第一个非 flag 参数，否则自动查找 label_ratio_0 下最新 pred
path_candidates = [a for a in sys.argv[1:] if not a.startswith('-')]
if path_candidates:
    pred_dir = path_candidates[0]
else:
    result_dirs = glob.glob(os.path.join(get_exp_root(0), '*', '*'))
    result_dirs = [d for d in result_dirs if os.path.isdir(d)]
    if result_dirs:
        def _sort_key(p):
            parts = p.replace('\\', '/').split('/')
            try:
                return (int(parts[-2]), int(parts[-1]))
            except (ValueError, IndexError):
                return (0, 0)
        result_dirs_sorted = sorted(result_dirs, key=_sort_key)
        pred_dir = result_dirs_sorted[-1]
        print("⚠️  未指定路径，自动使用: {}".format(pred_dir))
    else:
        pred_dir = os.path.join(get_exp_root(0), '1', '1')
        print("⚠️  未指定路径，使用默认: {}".format(pred_dir))

# 评估结果统一保存到 pred_dir/eval/
eval_out_dir = os.path.join(pred_dir, "eval")
fs = 30

print("="*60)
print("评估: Model Output (rPPG) vs Ground Truth (BVP)")
print("="*60)
print("预测结果目录: {}".format(pred_dir))
print("【方式1】PSD 级 (与 loss 对齐): {}".format("启用" if use_psd_eval else "跳过"))
print("【方式2】HR 级 (README 流程): {}".format("启用" if use_hr_eval else "跳过"))
print("波形可视化 (pred vs GT): {}".format("是" if save_viz else "否"))
print("="*60)

all_psd_corr = []
all_psd_mse = []
all_hr_pred = []
all_hr_gt = []
all_rppg_filtered = []
all_bvp_filtered = []
per_subject_data = {}
all_clip_details = []
# 【方式3】Random 10s clip accumulators
all_10s_psd_corr = []
all_10s_psd_mse = []
all_10s_hr_pred = []
all_10s_hr_gt = []


def estimate_hr_smart(sig_filtered, fs, method, use_harmonics, enable_smart):
    """HR 估计（辅评估用）"""
    if method == 'parabolic':
        hr_raw, _, _ = hr_fft_parabolic(sig_filtered, fs=fs, harmonics_removal=use_harmonics)
    elif method == 'zp':
        hr_raw, _, _ = hr_fft_zp(sig_filtered, fs=fs, harmonics_removal=use_harmonics)
    else:
        hr_raw, _, _ = hr_fft(sig_filtered, fs=fs, harmonics_removal=use_harmonics)

    if not enable_smart:
        return hr_raw

    peaks = compute_fft_peaks(sig_filtered, fs)
    hr_selected, _ = select_hr_from_peaks(peaks, fs, len(sig_filtered), use_harmonics_removal=use_harmonics)
    return hr_selected if hr_selected is not None else hr_raw


def compute_norm_psd(sig, fs, high_pass_bpm=40, low_pass_bpm=250):
    """
    与 loss.py CalculateNormPSD 一致的归一化 PSD 计算（numpy 版本）
    - 去均值 -> FFT -> |fft|^2 -> 保留 40-250 BPM 频段 -> 归一化使 sum=1
    """
    sig = np.reshape(sig, -1).astype(np.float64)
    sig = sig - np.mean(sig)
    n = len(sig)
    if n < 10:
        return None
    x = np.fft.rfft(sig, norm='forward')
    psd = (x.real ** 2 + x.imag ** 2)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    low_hz, high_hz = high_pass_bpm / 60.0, low_pass_bpm / 60.0
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    psd_masked = psd[mask]
    s = np.sum(psd_masked)
    if s < 1e-12:
        return None
    return psd_masked / s


def psd_metrics(rppg_f, bvp_f, fs):
    """
    PSD 级指标（与训练 loss 空间一致）
    - Pearson: 归一化 PSD 向量的相关系数
    - MSE: 归一化 PSD 的均方误差
    """
    pr = compute_norm_psd(rppg_f, fs)
    pb = compute_norm_psd(bvp_f, fs)
    if pr is None or pb is None:
        return np.nan, np.nan
    n = min(len(pr), len(pb))
    pr, pb = pr[:n], pb[:n]
    if n < 5:
        return np.nan, np.nan
    try:
        r, _ = pearsonr(pr, pb)
    except:
        r = np.nan
    mse = np.mean((pr - pb) ** 2)
    return r, mse


# 加载
npy_files = [f for f in sorted(os.listdir(pred_dir)) if f.endswith('.npy')]
if not npy_files:
    print("错误: 在 {} 中没有找到 .npy 文件".format(pred_dir))
    sys.exit(1)

T10 = int(10 * fs)   # 10-second clip length in frames

for f in npy_files:
    data = np.load(os.path.join(pred_dir, f), allow_pickle=True).item()
    subject_name = f.replace('.npy', '')

    print("\n{}:".format(subject_name))
    print("-" * 50)

    subject_psd_corr = []
    subject_psd_mse = []
    subject_hr_pred = []
    subject_hr_gt = []
    subject_errors = []
    sub_clips_buf = []   # buffer for 【方式3】sub-clips, printed after full clips

    for i in range(len(data['rppg_list'])):
        rppg = data['rppg_list'][i]   # Model output
        bvp_gt = data['bvp_list'][i]  # Ground truth (BVP waveform)
        clip_dur_s = len(rppg) / fs   # actual clip duration in seconds

        # 滤波（与 README 一致）
        rppg_filtered = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fs)
        bvp_filtered = butter_bandpass(bvp_gt, lowcut=0.6, highcut=4, fs=fs)

        # 【方式1】PSD 级指标（与 loss.py 对齐）
        psd_corr, psd_mse = psd_metrics(rppg_filtered, bvp_filtered, fs) if use_psd_eval else (np.nan, np.nan)
        if use_psd_eval:
            subject_psd_corr.append(psd_corr)
            subject_psd_mse.append(psd_mse)
            all_psd_corr.append(psd_corr)
            all_psd_mse.append(psd_mse)

        # 【方式2】HR 级（README: butter_bandpass + hr_fft）
        hr_pred = hr_gt = np.nan
        error_hr = np.nan
        if use_hr_eval:
            hr_pred = estimate_hr_smart(rppg_filtered, fs, hr_method, use_harmonics_removal, smart_peak)
            hr_gt = estimate_hr_smart(bvp_filtered, fs, hr_method, use_harmonics_removal, smart_peak)
            error_hr = abs(hr_pred - hr_gt)
            subject_hr_pred.append(hr_pred)
            subject_hr_gt.append(hr_gt)
            subject_errors.append(error_hr)
            all_hr_pred.append(hr_pred)
            all_hr_gt.append(hr_gt)

        # 打印
        parts = []
        if use_psd_eval:
            pc_str = f"{psd_corr:.4f}" if not np.isnan(psd_corr) else "N/A"
            pm_str = f"{psd_mse:.4f}" if not np.isnan(psd_mse) else "N/A"
            parts.append("PSD_corr={}, PSD_MSE={}".format(pc_str, pm_str))
        if use_hr_eval:
            parts.append("HR_pred={:.1f}, HR_gt={:.1f}, err={:.1f} BPM".format(hr_pred, hr_gt, error_hr))
        print("  Clip {:<3}[{:3.0f}s]: {}".format(str(i+1), clip_dur_s, " | ".join(parts)))

        # 可视化：rppg vs bvp 波形（保存到 pred_dir/eval/viz_waveform/）
        if save_viz:
            viz_dir = os.path.join(eval_out_dir, "viz_waveform")
            os.makedirs(viz_dir, exist_ok=True)
            fig, ax = plt.subplots(2, 1, figsize=(12, 6))
            t = np.arange(len(rppg_filtered)) / fs
            ax[0].plot(t, rppg_filtered, 'b-', label='rppg (pred)', linewidth=0.8)
            ax[0].plot(t, bvp_filtered, 'r-', alpha=0.7, label='bvp (GT)', linewidth=0.8)
            ax[0].set_title("{} Clip {} - Filtered Waveform".format(subject_name, i+1))
            ax[0].set_xlabel("Time (s)")
            ax[0].legend()
            ax[0].grid(True, alpha=0.3)
            ax[1].plot(t, rppg_filtered - np.mean(rppg_filtered), 'b-', label='rppg (demean)', linewidth=0.8)
            ax[1].plot(t, bvp_filtered - np.mean(bvp_filtered), 'r-', alpha=0.7, label='bvp (demean)', linewidth=0.8)
            ax[1].set_xlabel("Time (s)")
            ax[1].legend()
            ax[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, "{}_{}.png".format(subject_name, i+1)), dpi=120)
            plt.close()

        all_clip_details.append({
            'subject': subject_name,
            'clip_idx': i,
            'psd_corr': psd_corr if use_psd_eval else None,
            'psd_mse': psd_mse if use_psd_eval else None,
            'hr_pred': hr_pred if use_hr_eval else None,
            'hr_gt': hr_gt if use_hr_eval else None,
            'error': error_hr if use_hr_eval else None,
        })

        # 【方式3】Compute random 10s windows now, print later (after all full clips)
        n_rand = max(1, int(clip_dur_s / 10))
        clip_frames = len(rppg)
        if clip_frames >= T10:
            for j in range(n_rand):
                letter  = chr(ord('a') + j)
                seed_j  = abs(hash("{}_{}_{}".format(subject_name, i, letter))) % (2 ** 31)
                rng_j   = np.random.default_rng(seed_j)
                start_j = int(rng_j.integers(0, clip_frames - T10 + 1))
                rppg_j  = rppg[start_j : start_j + T10]
                bvp_j   = bvp_gt[start_j : start_j + T10]
                rppg_jf = butter_bandpass(rppg_j, lowcut=0.6, highcut=4, fs=fs)
                bvp_jf  = butter_bandpass(bvp_j,  lowcut=0.6, highcut=4, fs=fs)
                psd_cj, psd_mj = psd_metrics(rppg_jf, bvp_jf, fs) if use_psd_eval else (np.nan, np.nan)
                hr_pj = hr_gj = np.nan
                if use_hr_eval:
                    hr_pj = estimate_hr_smart(rppg_jf, fs, hr_method, use_harmonics_removal, smart_peak)
                    hr_gj = estimate_hr_smart(bvp_jf,  fs, hr_method, use_harmonics_removal, smart_peak)
                    all_10s_hr_pred.append(hr_pj)
                    all_10s_hr_gt.append(hr_gj)
                if use_psd_eval and not np.isnan(psd_cj):
                    all_10s_psd_corr.append(psd_cj)
                    all_10s_psd_mse.append(psd_mj)
                sub_clips_buf.append((i, letter, seed_j, start_j, psd_cj, psd_mj, hr_pj, hr_gj))

    # Print all 【方式3】sub-clips after the full-clip list
    for (ci, letter, seed_j, start_j, psd_cj, psd_mj, hr_pj, hr_gj) in sub_clips_buf:
        partsj = []
        if use_psd_eval:
            pc_str = "{:.4f}".format(psd_cj) if not np.isnan(psd_cj) else "N/A"
            pm_str = "{:.4f}".format(psd_mj) if not np.isnan(psd_mj) else "N/A"
            partsj.append("PSD_corr={}, PSD_MSE={}".format(pc_str, pm_str))
        if use_hr_eval:
            partsj.append("HR_pred={:.1f}, HR_gt={:.1f}, err={:.1f} BPM".format(
                hr_pj, hr_gj, abs(hr_pj - hr_gj)))
        label_j  = "{}{}".format(ci + 1, letter)
        suffix_j = "  ── seed={}  start={:.1f}s".format(seed_j, start_j / fs)
        print("  Clip {:<3}[ 10s]: {}{}".format(label_j, " | ".join(partsj), suffix_j))

    if subject_psd_corr or subject_hr_pred:
        d = per_subject_data.setdefault(subject_name, {})  # keep any 10s data already stored
        if use_psd_eval and subject_psd_corr:
            valid_pc = [c for c in subject_psd_corr if not np.isnan(c)]
            valid_pm = [m for m in subject_psd_mse if not np.isnan(m)]
            d['psd_corr_mean'] = np.mean(valid_pc) if valid_pc else np.nan
            d['psd_mse_mean'] = np.mean(valid_pm) if valid_pm else np.nan
        if use_hr_eval and subject_hr_pred:
            d['mae'] = np.mean(subject_errors)
            d['hr_pred'] = subject_hr_pred
            d['hr_gt'] = subject_hr_gt

# ========================================================================
# Exclude subjects filter applied globally
# ========================================================================
if exclude_subjects:
    all_psd_corr  = [v for v, d in zip(all_psd_corr,  all_clip_details) if d['subject'] not in exclude_subjects]
    all_psd_mse   = [v for v, d in zip(all_psd_mse,   all_clip_details) if d['subject'] not in exclude_subjects]
    all_hr_pred   = [v for v, d in zip(all_hr_pred,   all_clip_details) if d['subject'] not in exclude_subjects and d['hr_pred'] is not None]
    all_hr_gt     = [v for v, d in zip(all_hr_gt,     all_clip_details) if d['subject'] not in exclude_subjects and d['hr_gt']   is not None]

# ========================================================================
# 【方式1】PSD 级汇总（与 loss.py 对齐）
# ========================================================================
if use_psd_eval and all_psd_corr:
    valid_pc = [c for c in all_psd_corr if not np.isnan(c)]
    valid_pm = [m for m in all_psd_mse if not np.isnan(m)]
    print("\n" + "="*60)
    print("【方式1】PSD 级 - 归一化 PSD 相似度（与训练目标对齐）")
    print("="*60)
    print("  样本数: {}".format(len(all_psd_corr)))
    if valid_pc:
        print("  PSD Pearson 相关系数: {:.4f} ± {:.4f}".format(np.mean(valid_pc), np.std(valid_pc)))
    if valid_pm:
        print("  PSD MSE: {:.6f} ± {:.6f}".format(np.mean(valid_pm), np.std(valid_pm)))
    print("="*60)

# ========================================================================
# 【方式2】HR 级汇总（README: butter_bandpass + hr_fft）
# ========================================================================
if use_hr_eval and all_hr_pred:
    errors = np.array([abs(p - g) for p, g in zip(all_hr_pred, all_hr_gt)])
    mae = np.mean(errors)
    rmse_hr = np.sqrt(np.mean(errors**2))
    try:
        pearson_hr, _ = pearsonr(all_hr_pred, all_hr_gt)
    except:
        pearson_hr = np.nan
    p5 = np.mean(errors <= 5) * 100
    p10 = np.mean(errors <= 10) * 100

    print("\n【方式2】HR 级 - README 指示的 butter_bandpass + hr_fft 流程")
    print("-"*60)
    print("  MAE: {:.2f} BPM".format(mae))
    print("  RMSE: {:.2f} BPM".format(rmse_hr))
    if not np.isnan(pearson_hr):
        print("  Pearson (HR): {:.4f}".format(pearson_hr))
    print("  P5: {:.1f}%, P10: {:.1f}%".format(p5, p10))
    if exclude_subjects and all_clip_details:
        kept = [c for c in all_clip_details if c['subject'] not in exclude_subjects and c['error'] is not None]
        if kept:
            err_ex = np.array([c['error'] for c in kept])
            print("  [排除 {} 后] MAE: {:.2f} BPM, RMSE: {:.2f} BPM, 样本数: {}".format(
                ','.join(exclude_subjects), np.mean(err_ex), np.sqrt(np.mean(err_ex**2)), len(kept)))
    print("="*60)

# ========================================================================
# 【方式3】Random 10s clip 汇总
# ========================================================================
if all_10s_hr_pred or all_10s_psd_corr:
    print("\n" + "="*60)
    print("【方式3】Random 10s clip - 每段 clip 随机抽取, 30s clip → 3 窗口 (1a/1b/1c)")
    print("="*60)
    if use_psd_eval and all_10s_psd_corr:
        vpc10 = [c for c in all_10s_psd_corr if not np.isnan(c)]
        vpm10 = [m for m in all_10s_psd_mse  if not np.isnan(m)]
        print("  样本数 (10s windows): {}".format(len(vpc10)))
        if vpc10:
            print("  PSD Pearson 相关系数: {:.4f} ± {:.4f}".format(np.mean(vpc10), np.std(vpc10)))
        if vpm10:
            print("  PSD MSE: {:.6f} ± {:.6f}".format(np.mean(vpm10), np.std(vpm10)))
    if use_hr_eval and all_10s_hr_pred:
        err10 = np.array([abs(p - g) for p, g in zip(all_10s_hr_pred, all_10s_hr_gt)])
        mae10  = np.mean(err10)
        rmse10 = np.sqrt(np.mean(err10**2))
        try:
            pc10, _ = pearsonr(all_10s_hr_pred, all_10s_hr_gt)
        except:
            pc10 = np.nan
        p5_10  = np.mean(err10 <= 5)  * 100
        p10_10 = np.mean(err10 <= 10) * 100
        print("  MAE:  {:.2f} BPM".format(mae10))
        print("  RMSE: {:.2f} BPM".format(rmse10))
        if not np.isnan(pc10):
            print("  Pearson (HR): {:.4f}".format(pc10))
        print("  P5: {:.1f}%, P10: {:.1f}%".format(p5_10, p10_10))
    print("="*60)

# 每个 subject 汇总
if len(per_subject_data) <= 15:
    print("\n每个 Subject 统计:")
    print("-" * 50)
    for name, d in sorted(per_subject_data.items()):
        parts = []
        if use_psd_eval and 'psd_corr_mean' in d:
            parts.append("PSD_corr={:.4f}, PSD_MSE={:.6f}".format(d['psd_corr_mean'], d['psd_mse_mean']))
        if use_hr_eval and 'mae' in d:
            parts.append("MAE={:.2f} BPM".format(d['mae']))
        print("{}: {}".format(name, ", ".join(parts)))

# 保存评估摘要和可视化到 pred_dir/eval/
os.makedirs(eval_out_dir, exist_ok=True)
n_samples = len(all_psd_corr) or len(all_hr_pred) or 0
summary_lines = ["Evaluation Summary", "=" * 50, "Pred dir: {}".format(pred_dir), "Sample count: {}".format(n_samples)]
if use_psd_eval and all_psd_corr:
    vpc, vpm = [c for c in all_psd_corr if not np.isnan(c)], [m for m in all_psd_mse if not np.isnan(m)]
    summary_lines.append("")
    summary_lines.append("【方式1】PSD级: PSD_corr {:.4f}±{:.4f}, PSD_MSE {:.6f}±{:.6f}".format(
        np.mean(vpc), np.std(vpc), np.mean(vpm), np.std(vpm)) if vpc and vpm else "【方式1】PSD级: N/A")
if use_hr_eval and all_hr_pred:
    errs = np.array([abs(p - g) for p, g in zip(all_hr_pred, all_hr_gt)])
    summary_lines.append("")
    summary_lines.append("【方式2】HR级: MAE {:.2f} BPM, RMSE {:.2f} BPM, P5 {:.1f}%, P10 {:.1f}%".format(
        np.mean(errs), np.sqrt(np.mean(errs**2)), np.mean(errs <= 5) * 100, np.mean(errs <= 10) * 100))
if all_10s_hr_pred:
    err10s = np.array([abs(p - g) for p, g in zip(all_10s_hr_pred, all_10s_hr_gt)])
    summary_lines.append("")
    summary_lines.append("【方式3】Random 10s: MAE {:.2f} BPM, RMSE {:.2f} BPM, P5 {:.1f}%, P10 {:.1f}%, n={}".format(
        np.mean(err10s), np.sqrt(np.mean(err10s**2)),
        np.mean(err10s <= 5) * 100, np.mean(err10s <= 10) * 100, len(err10s)))
if all_10s_psd_corr:
    vpc10 = [c for c in all_10s_psd_corr if not np.isnan(c)]
    vpm10 = [m for m in all_10s_psd_mse  if not np.isnan(m)]
    if vpc10:
        summary_lines.append("【方式3】Random 10s PSD: corr {:.4f}±{:.4f}, MSE {:.6f}±{:.6f}".format(
            np.mean(vpc10), np.std(vpc10), np.mean(vpm10), np.std(vpm10)))
with open(os.path.join(eval_out_dir, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))
print("\n评估结果已保存到: {}".format(eval_out_dir))
print("  - summary.txt")
if save_viz:
    print("  - viz_waveform/*.png")
print("\n✓ 评估完成")
