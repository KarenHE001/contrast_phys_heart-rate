# -*- coding: utf-8 -*-
"""
EfficientPhysNet 评估 — 两层结构

Layer 1 波形质量（Signal-level）: 模型输出 rPPG 是否含生理节律、与 GT 频谱/形状一致性
  - PSD Pearson / PSD MSE（频谱形状，与训练 ContrastLoss 一致）
  - Capture-SNR(dB): GT HR 及 2×HR 频点附近能量 vs 其余（utils_sig.SNR_get）
  - IPR: 40–250 BPM 外能量占比，与训练 IrrelevantPowerRatio 一致（用原始 rppg）
  - WaveCorr@0 / WaveCorr@lag / ACF Corr
  - HR_sub_std: 同 clip 内两段子窗 HR 标准差（稳定性，仅当帧数足够）

Layer 2 心率结果（Device-level）: MAE / RMSE / P5 / P10 / Pearson(HR)

诊断: Random 10s 子窗口

运行:
  cd contrast-phys+
  python EfficientPhysNet/evaluation/evaluate.py <pred_dir> [--save-viz]
"""
import json
import os
import re
import sys

import numpy as np
from scipy.stats import pearsonr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_EPN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CP = os.path.dirname(_EPN)
if _CP not in sys.path:
    sys.path.insert(0, _CP)

from utils_sig import (
    butter_bandpass,
    compute_fft_peaks,
    hr_fft,
    hr_fft_parabolic,
    hr_fft_zp,
    select_hr_from_peaks,
    SNR_get,
)


use_psd_eval = True
use_hr_eval = True
use_harmonics_removal = True
hr_method = "parabolic"
smart_peak = True
save_viz = False
exclude_subjects = []
fs = 30
max_lag_sec = 2.0


def _parse_args(argv):
    global use_psd_eval, use_hr_eval, use_harmonics_removal
    global hr_method, smart_peak, save_viz, exclude_subjects

    for a in argv:
        if a == "--psd-only":
            use_psd_eval, use_hr_eval = True, False
        elif a == "--hr-only":
            use_psd_eval, use_hr_eval = False, True
        elif a == "--save-viz":
            save_viz = True
        elif a in ("--no-harmonics", "-n"):
            use_harmonics_removal = False
        elif a == "--no-smart-peak":
            smart_peak = False
        elif a == "--parabolic":
            hr_method = "parabolic"
        elif a == "--zp":
            hr_method = "zp"
        elif a == "--original":
            hr_method = "original"
        elif a.startswith("--exclude-subjects="):
            exclude_subjects = [
                s.strip() for s in a.split("=", 1)[1].split(",") if s.strip()
            ]


def _infer_pred_dir(argv):
    path_candidates = [a for a in argv if not a.startswith("-")]
    if path_candidates:
        return path_candidates[0]

    default_pred = os.path.join(
        _CP,
        "results",
        "EfficientPhysNet",
        "label_ratio_0",
        "inference",
        "curriculum",
        "t10",
        "1",
    )
    if os.path.isdir(default_pred):
        print(f"⚠️  未指定路径，使用: {default_pred}")
        return default_pred

    print(
        "用法: python EfficientPhysNet/evaluation/evaluate.py "
        "<pred_dir> [--save-viz] [--hr-only] [--psd-only]"
    )
    print("  例: results/EfficientPhysNet/label_ratio_0/inference/curriculum/t10/1")
    sys.exit(1)


def _safe_pearson(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(x), len(y))
    if n < 5:
        return np.nan
    x = x[:n]
    y = y[:n]
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    try:
        r, _ = pearsonr(x, y)
        return r
    except Exception:
        return np.nan


def _compute_norm_psd(sig, fs_hz, high_pass_bpm=40, low_pass_bpm=250):
    sig = np.asarray(sig).reshape(-1).astype(np.float64)
    sig = sig - np.mean(sig)
    n = len(sig)
    if n < 10:
        return None, None
    x = np.fft.rfft(sig, norm="forward")
    psd = x.real**2 + x.imag**2
    freqs_hz = np.fft.rfftfreq(n, 1.0 / fs_hz)
    low_hz, high_hz = high_pass_bpm / 60.0, low_pass_bpm / 60.0
    mask = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)
    psd_band = psd[mask]
    freqs_bpm = freqs_hz[mask] * 60.0
    s = np.sum(psd_band)
    if s < 1e-12:
        return None, None
    return freqs_bpm, psd_band / s


def _psd_metrics(rppg_f, bvp_f, fs_hz):
    _, pr = _compute_norm_psd(rppg_f, fs_hz)
    _, pb = _compute_norm_psd(bvp_f, fs_hz)
    if pr is None or pb is None:
        return np.nan, np.nan
    n = min(len(pr), len(pb))
    pr = pr[:n]
    pb = pb[:n]
    if n < 5:
        return np.nan, np.nan
    return _safe_pearson(pr, pb), np.mean((pr - pb) ** 2)


def _max_lag_corr(x, y, fs_hz, max_lag_s=2.0):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = min(len(x), len(y))
    if n < 5:
        return np.nan
    x = x[:n]
    y = y[:n]
    max_lag = int(max_lag_s * fs_hz)
    best = np.nan
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs = x[lag:]
            ys = y[: n - lag]
        else:
            xs = x[: n + lag]
            ys = y[-lag:]
        r = _safe_pearson(xs, ys)
        if np.isnan(r):
            continue
        if np.isnan(best) or r > best:
            best = r
    return best


def _acf(sig, max_lag):
    x = np.asarray(sig).reshape(-1).astype(np.float64)
    x = x - np.mean(x)
    if np.std(x) < 1e-12:
        return None
    corr = np.correlate(x, x, mode="full")
    mid = len(corr) // 2
    acf = corr[mid : mid + max_lag + 1]
    if acf[0] == 0:
        return None
    return acf / acf[0]


def _acf_corr(x, y, fs_hz, acf_lag_s=4.0):
    max_lag = int(acf_lag_s * fs_hz)
    ax = _acf(x, max_lag)
    ay = _acf(y, max_lag)
    if ax is None or ay is None:
        return np.nan
    return _safe_pearson(ax, ay)


def _ipr_numpy(sig, fs_hz, high_pass_bpm=40, low_pass_bpm=250):
    """与 IrrelevantPowerRatio 一致: 带外能量 / (带内+带外)，输入为时域波形。"""
    sig = np.asarray(sig).reshape(-1).astype(np.float64)
    n = len(sig)
    if n < 8:
        return np.nan
    x = np.fft.rfft(sig, norm="forward")
    mag = np.abs(x)
    freqs_hz = np.fft.rfftfreq(n, 1.0 / fs_hz)
    low_hz, high_hz = high_pass_bpm / 60.0, low_pass_bpm / 60.0
    mask = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)
    use_e = float(np.sum(mag[mask]))
    zero_e = float(np.sum(mag[~mask]))
    denom = use_e + zero_e
    if denom < 1e-20:
        return np.nan
    return zero_e / denom


def _capture_snr_db(rppg_f, hr_gt_bpm, fs_hz):
    """GT HR 及谐波附近能量 vs 总能量，dB。hr_gt 无效时返回 nan。"""
    if hr_gt_bpm is None or np.isnan(hr_gt_bpm) or hr_gt_bpm < 30 or hr_gt_bpm > 220:
        return np.nan
    try:
        snr = SNR_get(np.asarray(rppg_f).reshape(-1), float(hr_gt_bpm), fs_hz, filtered=False)
        if np.isnan(snr) or np.isinf(snr):
            return np.nan
        return float(snr)
    except Exception:
        return np.nan


def _estimate_hr(sig_filtered, fs_hz, method, use_harmonics, enable_smart):
    if method == "parabolic":
        hr_raw, _, _ = hr_fft_parabolic(
            sig_filtered, fs=fs_hz, harmonics_removal=use_harmonics
        )
    elif method == "zp":
        hr_raw, _, _ = hr_fft_zp(
            sig_filtered, fs=fs_hz, harmonics_removal=use_harmonics
        )
    else:
        hr_raw, _, _ = hr_fft(sig_filtered, fs=fs_hz, harmonics_removal=use_harmonics)

    if not enable_smart:
        return hr_raw

    peaks = compute_fft_peaks(sig_filtered, fs_hz)
    hr_sel, _ = select_hr_from_peaks(
        peaks, fs_hz, len(sig_filtered), use_harmonics_removal=use_harmonics
    )
    return hr_sel if hr_sel is not None else hr_raw


def _hr_subwindow_stability(rppg, fs_hz, method, use_harmonics, enable_smart):
    """同 clip 两半段各估 HR，返回标准差（稳定性）；帧不足则 nan。"""
    rppg = np.asarray(rppg).reshape(-1)
    n = len(rppg)
    half = n // 2
    if half < int(2.5 * fs_hz):
        return np.nan
    a = butter_bandpass(rppg[:half], 0.6, 4, fs=fs_hz)
    b = butter_bandpass(rppg[half:], 0.6, 4, fs=fs_hz)
    h1 = _estimate_hr(a, fs_hz, method, use_harmonics, enable_smart)
    h2 = _estimate_hr(b, fs_hz, method, use_harmonics, enable_smart)
    if np.isnan(h1) or np.isnan(h2):
        return np.nan
    return float(np.std([h1, h2]))


def _extract_path_meta(pred_dir):
    path = pred_dir.replace("\\", "/")
    m = re.search(
        r"results/EfficientPhysNet/(label_ratio_[^/]+)/inference/([^/]+)/t(\d+)/(\d+)$",
        path,
    )
    if not m:
        return {}
    return {
        "label_ratio_tag": m.group(1),
        "strategy": m.group(2),
        "time_interval_sec": int(m.group(3)),
        "run_id": int(m.group(4)),
    }


def _load_meta(pred_dir):
    meta_path = os.path.join(pred_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _fmt(v, nd=4):
    if v is None or np.isnan(v):
        return "N/A"
    return f"{v:.{nd}f}"


def _safe_nanmean(arr):
    """np.nanmean 在全部为 nan 时会触发 RuntimeWarning，先检查"""
    if not arr:
        return np.nan
    a = np.asarray(arr, dtype=float)
    if np.all(np.isnan(a)):
        return np.nan
    return float(np.nanmean(a))


def main():
    _parse_args(sys.argv[1:])
    pred_dir = _infer_pred_dir(sys.argv[1:])
    eval_out_dir = os.path.join(pred_dir, "eval")
    os.makedirs(eval_out_dir, exist_ok=True)

    meta_path = _extract_path_meta(pred_dir)
    meta_json = _load_meta(pred_dir)
    global fs
    if "fs" in meta_json:
        fs = int(meta_json["fs"])

    print("=" * 68)
    print("评估: Layer 1 波形质量 + Layer 2 心率结果")
    print("=" * 68)
    print(f"pred_dir: {pred_dir}")
    if meta_path:
        print(
            f"scale=t{meta_path['time_interval_sec']}s | "
            f"strategy={meta_path['strategy']} | run={meta_path['run_id']} | "
            f"{meta_path['label_ratio_tag']}"
        )
    if meta_json:
        print(
            "meta: input_size={} | inference_time={:.2f}s | clips={}".format(
                meta_json.get("input_size", "N/A"),
                float(meta_json.get("inference_seconds", np.nan)),
                meta_json.get("clips_total", "N/A"),
            )
        )
    print(f"Layer 1 PSD: {'启用' if use_psd_eval else '跳过'} | HR 用于 Capture-SNR: {'是' if use_hr_eval else '否'}")
    print(f"Layer 2 HR 指标: {'启用' if use_hr_eval else '跳过'}")
    print("诊断: Random 10s 子窗 | WaveCorr / ACF")
    print("=" * 68)

    npy_files = [f for f in sorted(os.listdir(pred_dir)) if f.endswith(".npy")]
    if not npy_files:
        print(f"错误: {pred_dir} 中无 .npy 文件")
        sys.exit(1)

    all_psd_corr, all_psd_mse = [], []
    all_hr_pred, all_hr_gt = [], []
    all_wave_corr0, all_wave_corr_lag, all_acf_corr = [], [], []
    all_capture_snr, all_ipr, all_hr_sub_std = [], [], []
    all_10s_psd_corr, all_10s_psd_mse = [], []
    all_10s_hr_pred, all_10s_hr_gt = [], []
    all_clip_details = []
    per_subject_data = {}
    T10 = int(10 * fs)

    for f in npy_files:
        data = np.load(os.path.join(pred_dir, f), allow_pickle=True).item()
        subject = f.replace(".npy", "")
        print(f"\n{subject}:")
        print("-" * 56)

        s_psd_corr, s_psd_mse = [], []
        s_hr_pred, s_hr_gt, s_hr_err = [], [], []
        s_wlag, s_acf = [], []
        s_capture_snr, s_ipr, s_hr_sub_std = [], [], []
        sub_windows_buf = []

        for i in range(len(data["rppg_list"])):
            rppg = np.asarray(data["rppg_list"][i], dtype=np.float64)
            bvp = np.asarray(data["bvp_list"][i], dtype=np.float64)
            clip_dur_s = len(rppg) / fs

            rppg_f = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fs)
            bvp_f = butter_bandpass(bvp, lowcut=0.6, highcut=4, fs=fs)

            psd_corr, psd_mse = (
                _psd_metrics(rppg_f, bvp_f, fs) if use_psd_eval else (np.nan, np.nan)
            )
            if use_psd_eval:
                s_psd_corr.append(psd_corr)
                s_psd_mse.append(psd_mse)
                all_psd_corr.append(psd_corr)
                all_psd_mse.append(psd_mse)

            hr_pred = hr_gt = np.nan
            hr_err = np.nan
            if use_hr_eval:
                hr_pred = _estimate_hr(
                    rppg_f, fs, hr_method, use_harmonics_removal, smart_peak
                )
                hr_gt = _estimate_hr(
                    bvp_f, fs, hr_method, use_harmonics_removal, smart_peak
                )
                hr_err = abs(hr_pred - hr_gt)
                s_hr_pred.append(hr_pred)
                s_hr_gt.append(hr_gt)
                s_hr_err.append(hr_err)
                all_hr_pred.append(hr_pred)
                all_hr_gt.append(hr_gt)

            ipr_val = _ipr_numpy(rppg, fs)
            cap_snr = (
                _capture_snr_db(rppg_f, hr_gt, fs)
                if use_hr_eval and not np.isnan(hr_gt)
                else np.nan
            )
            hr_std_sw = (
                _hr_subwindow_stability(
                    rppg, fs, hr_method, use_harmonics_removal, smart_peak
                )
                if use_hr_eval
                else np.nan
            )
            s_ipr.append(ipr_val)
            s_capture_snr.append(cap_snr)
            s_hr_sub_std.append(hr_std_sw)
            all_ipr.append(ipr_val)
            all_capture_snr.append(cap_snr)
            all_hr_sub_std.append(hr_std_sw)

            w_corr0 = _safe_pearson(rppg_f, bvp_f)
            w_corr_lag = _max_lag_corr(rppg_f, bvp_f, fs_hz=fs, max_lag_s=max_lag_sec)
            acf_corr = _acf_corr(rppg_f, bvp_f, fs_hz=fs, acf_lag_s=4.0)
            s_wlag.append(w_corr_lag)
            s_acf.append(acf_corr)
            all_wave_corr0.append(w_corr0)
            all_wave_corr_lag.append(w_corr_lag)
            all_acf_corr.append(acf_corr)

            parts = []
            if use_psd_eval:
                parts.append(
                    f"PSD_corr={_fmt(psd_corr, 4)}, PSD_MSE={_fmt(psd_mse, 4)}"
                )
            if use_hr_eval:
                parts.append(
                    f"HR_pred={_fmt(hr_pred, 1)}, HR_gt={_fmt(hr_gt, 1)}, err={_fmt(hr_err, 1)} BPM"
                )
            parts.append(f"WaveLagCorr={_fmt(w_corr_lag, 3)}")
            parts.append(f"CapSNR={_fmt(cap_snr, 2)}dB")
            parts.append(f"IPR={_fmt(ipr_val, 4)}")
            parts.append(f"HR_half_std={_fmt(hr_std_sw, 2)}")
            print(f"  Clip {i + 1:<3}[{clip_dur_s:3.0f}s]: {' | '.join(parts)}")

            if save_viz:
                viz_dir = os.path.join(eval_out_dir, "viz_waveform")
                os.makedirs(viz_dir, exist_ok=True)
                t = np.arange(len(rppg_f)) / fs
                fbpm_p, psd_p = _compute_norm_psd(rppg_f, fs)
                fbpm_g, psd_g = _compute_norm_psd(bvp_f, fs)
                acf_lag = int(4.0 * fs)
                acf_p = _acf(rppg_f, acf_lag)
                acf_g = _acf(bvp_f, acf_lag)
                lag_sec = np.arange(acf_lag + 1) / fs

                rppg_plot = (rppg_f - np.mean(rppg_f)) / (np.std(rppg_f) + 1e-12)
                bvp_plot = (bvp_f - np.mean(bvp_f)) / (np.std(bvp_f) + 1e-12)

                fig, ax = plt.subplots(3, 1, figsize=(12, 9))
                ax[0].plot(t, rppg_plot, "b-", label="rppg (pred)", linewidth=0.8)
                ax[0].plot(t, bvp_plot, "r-", alpha=0.7, label="bvp (GT)", linewidth=0.8)
                ax[0].set_ylabel("Amplitude (z-score)")
                ax[0].set_title(f"{subject} Clip {i + 1} - Waveform")
                ax[0].legend()
                ax[0].grid(True, alpha=0.3)

                if fbpm_p is not None and fbpm_g is not None:
                    ax[1].plot(fbpm_p, psd_p, "b-", linewidth=1.0, label="pred PSD")
                    ax[1].plot(fbpm_g, psd_g, "r-", linewidth=1.0, alpha=0.8, label="GT PSD")
                    ax[1].set_xlim(40, 250)
                    ax[1].set_title("Normalized PSD (40-250 BPM)")
                    ax[1].legend()
                ax[1].grid(True, alpha=0.3)

                if acf_p is not None and acf_g is not None:
                    ax[2].plot(lag_sec, acf_p, "b-", linewidth=1.0, label="pred ACF")
                    ax[2].plot(lag_sec, acf_g, "r-", linewidth=1.0, alpha=0.8, label="GT ACF")
                    ax[2].set_title("Autocorrelation (0-4s)")
                    ax[2].legend()
                ax[2].grid(True, alpha=0.3)
                ax[2].set_xlabel("Time / BPM axis")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(viz_dir, f"{subject}_{i + 1}.png"),
                    dpi=120,
                )
                plt.close()

            all_clip_details.append(
                {
                    "subject": subject,
                    "clip_idx": i,
                    "psd_corr": psd_corr if use_psd_eval else None,
                    "psd_mse": psd_mse if use_psd_eval else None,
                    "hr_pred": hr_pred if use_hr_eval else None,
                    "hr_gt": hr_gt if use_hr_eval else None,
                    "error": hr_err if use_hr_eval else None,
                    "capture_snr_db": cap_snr,
                    "ipr": ipr_val,
                    "hr_half_std": hr_std_sw,
                }
            )

            n_rand = max(1, int(clip_dur_s / 10))
            clip_frames = len(rppg)
            if clip_frames >= T10:
                for j in range(n_rand):
                    letter = chr(ord("a") + j)
                    seed_j = abs(hash(f"{subject}_{i}_{letter}")) % (2**31)
                    rng_j = np.random.default_rng(seed_j)
                    start_j = int(rng_j.integers(0, clip_frames - T10 + 1))
                    rppg_j = rppg[start_j : start_j + T10]
                    bvp_j = bvp[start_j : start_j + T10]
                    rppg_jf = butter_bandpass(rppg_j, lowcut=0.6, highcut=4, fs=fs)
                    bvp_jf = butter_bandpass(bvp_j, lowcut=0.6, highcut=4, fs=fs)
                    psd_cj, psd_mj = (
                        _psd_metrics(rppg_jf, bvp_jf, fs)
                        if use_psd_eval
                        else (np.nan, np.nan)
                    )
                    hr_pj = hr_gj = np.nan
                    if use_hr_eval:
                        hr_pj = _estimate_hr(
                            rppg_jf,
                            fs,
                            hr_method,
                            use_harmonics_removal,
                            smart_peak,
                        )
                        hr_gj = _estimate_hr(
                            bvp_jf,
                            fs,
                            hr_method,
                            use_harmonics_removal,
                            smart_peak,
                        )
                        all_10s_hr_pred.append(hr_pj)
                        all_10s_hr_gt.append(hr_gj)
                    if use_psd_eval and not np.isnan(psd_cj):
                        all_10s_psd_corr.append(psd_cj)
                        all_10s_psd_mse.append(psd_mj)
                    sub_windows_buf.append(
                        (i, letter, seed_j, start_j, psd_cj, psd_mj, hr_pj, hr_gj)
                    )

        for (ci, letter, seed_j, start_j, psd_cj, psd_mj, hr_pj, hr_gj) in sub_windows_buf:
            partsj = []
            if use_psd_eval:
                partsj.append(
                    f"PSD_corr={_fmt(psd_cj, 4)}, PSD_MSE={_fmt(psd_mj, 4)}"
                )
            if use_hr_eval:
                partsj.append(
                    f"HR_pred={_fmt(hr_pj, 1)}, HR_gt={_fmt(hr_gj, 1)}, "
                    f"err={_fmt(abs(hr_pj - hr_gj), 1)} BPM"
                )
            print(
                f"  Clip {ci + 1}{letter:<2}[ 10s]: {' | '.join(partsj)}"
                f"  ── seed={seed_j} start={start_j / fs:.1f}s"
            )

        if s_psd_corr or s_hr_pred:
            d = per_subject_data.setdefault(subject, {})
            if use_psd_eval:
                d["psd_corr_mean"] = np.nanmean(s_psd_corr) if s_psd_corr else np.nan
                d["psd_mse_mean"] = np.nanmean(s_psd_mse) if s_psd_mse else np.nan
            if use_hr_eval:
                d["mae"] = np.nanmean(s_hr_err) if s_hr_err else np.nan
            d["wave_lag_mean"] = _safe_nanmean(s_wlag)
            d["acf_corr_mean"] = np.nanmean(s_acf) if s_acf else np.nan
            d["capture_snr_mean"] = _safe_nanmean(s_capture_snr)
            d["ipr_mean"] = _safe_nanmean(s_ipr)
            d["hr_half_std_mean"] = _safe_nanmean(s_hr_sub_std)

    if exclude_subjects:
        all_psd_corr = [
            v
            for v, d in zip(all_psd_corr, all_clip_details)
            if d["subject"] not in exclude_subjects
        ]
        all_psd_mse = [
            v
            for v, d in zip(all_psd_mse, all_clip_details)
            if d["subject"] not in exclude_subjects
        ]
        all_hr_pred = [
            v
            for v, d in zip(all_hr_pred, all_clip_details)
            if d["subject"] not in exclude_subjects and d["hr_pred"] is not None
        ]
        all_hr_gt = [
            v
            for v, d in zip(all_hr_gt, all_clip_details)
            if d["subject"] not in exclude_subjects and d["hr_gt"] is not None
        ]
        all_capture_snr = [
            v
            for v, d in zip(all_capture_snr, all_clip_details)
            if d["subject"] not in exclude_subjects
        ]
        all_ipr = [
            v for v, d in zip(all_ipr, all_clip_details) if d["subject"] not in exclude_subjects
        ]
        all_hr_sub_std = [
            v
            for v, d in zip(all_hr_sub_std, all_clip_details)
            if d["subject"] not in exclude_subjects
        ]
        all_wave_corr0 = [
            v
            for v, d in zip(all_wave_corr0, all_clip_details)
            if d["subject"] not in exclude_subjects
        ]
        all_wave_corr_lag = [
            v
            for v, d in zip(all_wave_corr_lag, all_clip_details)
            if d["subject"] not in exclude_subjects
        ]
        all_acf_corr = [
            v for v, d in zip(all_acf_corr, all_clip_details) if d["subject"] not in exclude_subjects
        ]

    print("\n" + "=" * 68)
    print("Layer 1 波形质量（rPPG vs GT 规律 / 频谱 / 稳定性）")
    print("=" * 68)
    if use_psd_eval and all_psd_corr:
        print(f"  PSD Pearson: {_fmt(np.nanmean(all_psd_corr), 4)} ± {_fmt(np.nanstd(all_psd_corr), 4)}  (n={len(all_psd_corr)})")
        print(f"  PSD MSE:     {_fmt(np.nanmean(all_psd_mse), 6)} ± {_fmt(np.nanstd(all_psd_mse), 6)}")
    if use_hr_eval and all_capture_snr:
        print(
            "  Capture-SNR: {:.2f} ± {:.2f} dB  (GT HR 频点及谐波附近能量)".format(
                _safe_nanmean(all_capture_snr),
                float(np.nanstd(np.asarray(all_capture_snr, dtype=float))),
            )
        )
    if all_ipr:
        print(
            "  IPR:         {:.4f} ± {:.4f}  (40–250 BPM 外能量占比，越低越好)".format(
                _safe_nanmean(all_ipr),
                float(np.nanstd(np.asarray(all_ipr, dtype=float))),
            )
        )
    print(
        "  WaveCorr@0={:.4f} | WaveCorr@lag={:.4f} | ACF={:.4f}".format(
            _safe_nanmean(all_wave_corr0),
            _safe_nanmean(all_wave_corr_lag),
            _safe_nanmean(all_acf_corr),
        )
    )
    if all_hr_sub_std:
        print(
            "  HR_half_std: {:.2f} ± {:.2f} BPM  (同 clip 两半段 HR 离散度)".format(
                _safe_nanmean(all_hr_sub_std),
                float(np.nanstd(np.asarray(all_hr_sub_std, dtype=float))),
            )
        )

    print("\n" + "=" * 68)
    print("Layer 2 心率结果（设备侧）")
    print("=" * 68)
    if use_hr_eval and all_hr_pred:
        errors = np.array([abs(p - g) for p, g in zip(all_hr_pred, all_hr_gt)])
        mae = np.mean(errors)
        rmse_hr = np.sqrt(np.mean(errors**2))
        pearson_hr = _safe_pearson(all_hr_pred, all_hr_gt)
        p5 = np.mean(errors <= 5) * 100
        p10 = np.mean(errors <= 10) * 100
        print(f"  n={len(errors)}")
        print(f"  MAE={mae:.2f} BPM, RMSE={rmse_hr:.2f} BPM, P5={p5:.1f}%, P10={p10:.1f}%")
        print(f"  Pearson(HR)={_fmt(pearson_hr, 4)}")
    if all_10s_hr_pred or all_10s_psd_corr:
        print("\n" + "=" * 68)
        print("诊断：Random 10s 子窗口")
        print("=" * 68)
        if use_psd_eval and all_10s_psd_corr:
            print(
                "  PSD Pearson={:.4f} ± {:.4f}, PSD MSE={:.6f} ± {:.6f}, n={}".format(
                    float(np.nanmean(all_10s_psd_corr)),
                    float(np.nanstd(all_10s_psd_corr)),
                    float(np.nanmean(all_10s_psd_mse)),
                    float(np.nanstd(all_10s_psd_mse)),
                    len(all_10s_psd_corr),
                )
            )
        if use_hr_eval and all_10s_hr_pred:
            err10 = np.array([abs(p - g) for p, g in zip(all_10s_hr_pred, all_10s_hr_gt)])
            print(
                "  MAE={:.2f} BPM, RMSE={:.2f} BPM, P5={:.1f}%, P10={:.1f}%, n={}".format(
                    float(np.mean(err10)),
                    float(np.sqrt(np.mean(err10**2))),
                    float(np.mean(err10 <= 5) * 100),
                    float(np.mean(err10 <= 10) * 100),
                    len(err10),
                )
            )

    if len(per_subject_data) <= 15:
        print("\n每个 Subject (Layer 1 + 2):")
        for name, d in sorted(per_subject_data.items()):
            parts = []
            if use_psd_eval and "psd_corr_mean" in d:
                parts.append(
                    f"PSD={_fmt(d['psd_corr_mean'], 4)} MSE={_fmt(d['psd_mse_mean'], 6)}"
                )
            parts.append(f"CapSNR={_fmt(d.get('capture_snr_mean', np.nan), 2)}dB")
            parts.append(f"IPR={_fmt(d.get('ipr_mean', np.nan), 4)}")
            parts.append(f"WLag={_fmt(d['wave_lag_mean'], 3)} ACF={_fmt(d['acf_corr_mean'], 3)}")
            parts.append(f"HR_½std={_fmt(d.get('hr_half_std_mean', np.nan), 2)}")
            if use_hr_eval and "mae" in d:
                parts.append(f"MAE={_fmt(d['mae'], 2)}BPM")
            print(f"  {name}: " + " | ".join(parts))

    summary_lines = []
    summary_lines.append("Evaluation Summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"pred_dir: {pred_dir}")
    if meta_path:
        summary_lines.append(
            "path_meta: strategy={strategy}, scale=t{time_interval_sec}, run={run_id}, {label_ratio_tag}".format(
                **meta_path
            )
        )
    if meta_json:
        summary_lines.append(
            "meta: input_size={input_size}, inference_time={inference_seconds:.2f}s, "
            "clips_total={clips_total}, device={device}".format(
                input_size=meta_json.get("input_size", "N/A"),
                inference_seconds=float(meta_json.get("inference_seconds", np.nan)),
                clips_total=meta_json.get("clips_total", "N/A"),
                device=meta_json.get("device", "N/A"),
            )
        )
    summary_lines.append("")
    summary_lines.append("Layer 1 waveform quality (signal-level)")
    if use_psd_eval and all_psd_corr:
        summary_lines.append(
            "  PSD: corr {:.4f}±{:.4f}, mse {:.6f}±{:.6f}, n={}".format(
                float(np.nanmean(all_psd_corr)),
                float(np.nanstd(all_psd_corr)),
                float(np.nanmean(all_psd_mse)),
                float(np.nanstd(all_psd_mse)),
                len(all_psd_corr),
            )
        )
    if use_hr_eval and all_capture_snr:
        summary_lines.append(
            "  Capture-SNR: {:.2f}±{:.2f} dB".format(
                _safe_nanmean(all_capture_snr),
                float(np.nanstd(np.asarray(all_capture_snr, dtype=float))),
            )
        )
    if all_ipr:
        summary_lines.append(
            "  IPR: {:.4f}±{:.4f}".format(
                _safe_nanmean(all_ipr),
                float(np.nanstd(np.asarray(all_ipr, dtype=float))),
            )
        )
    summary_lines.append(
        "  Wave: corr0 {:.4f}, corr_lag {:.4f}, acf {:.4f}".format(
            _safe_nanmean(all_wave_corr0),
            _safe_nanmean(all_wave_corr_lag),
            _safe_nanmean(all_acf_corr),
        )
    )
    if all_hr_sub_std:
        summary_lines.append(
            "  HR_half_std: {:.2f}±{:.2f} BPM".format(
                _safe_nanmean(all_hr_sub_std),
                float(np.nanstd(np.asarray(all_hr_sub_std, dtype=float))),
            )
        )
    summary_lines.append("")
    summary_lines.append("Layer 2 heart rate (device-level)")
    if use_hr_eval and all_hr_pred:
        errs = np.array([abs(p - g) for p, g in zip(all_hr_pred, all_hr_gt)])
        summary_lines.append(
            "  MAE {:.2f} BPM, RMSE {:.2f} BPM, P5 {:.1f}%, P10 {:.1f}%, "
            "Pearson {:.4f}, n={}".format(
                float(np.mean(errs)),
                float(np.sqrt(np.mean(errs**2))),
                float(np.mean(errs <= 5) * 100),
                float(np.mean(errs <= 10) * 100),
                float(_safe_pearson(all_hr_pred, all_hr_gt)),
                len(errs),
            )
        )
    if all_10s_hr_pred or all_10s_psd_corr:
        summary_lines.append("Diagnostics: Random10s windows — see console")

    # Per-subject summary
    summary_lines.append("")
    summary_lines.append("-" * 60)
    summary_lines.append("Per-subject:")
    for name, d in sorted(per_subject_data.items()):
        parts = []
        if use_psd_eval and "psd_corr_mean" in d:
            parts.append("PSD_corr={:.4f} PSD_MSE={:.6f}".format(
                d["psd_corr_mean"], d["psd_mse_mean"]))
        parts.append("CapSNR={:.2f}dB IPR={:.4f}".format(
            d.get("capture_snr_mean", np.nan), d.get("ipr_mean", np.nan)))
        parts.append("WaveLag={:.3f} ACF={:.3f} HR_½std={:.2f}".format(
            d.get("wave_lag_mean", np.nan),
            d.get("acf_corr_mean", np.nan),
            d.get("hr_half_std_mean", np.nan),
        ))
        if use_hr_eval and "mae" in d:
            parts.append("MAE={:.2f}BPM".format(d["mae"]))
        summary_lines.append("  {}: {}".format(name, " | ".join(parts)))

    # Per-clip HR: subject | clip | HR_pred | HR_gt | err
    details_filtered = [c for c in all_clip_details if c["subject"] not in exclude_subjects]
    if use_hr_eval and details_filtered:
        summary_lines.append("")
        summary_lines.append("-" * 60)
        summary_lines.append("Per-clip HR (subject | clip | HR_pred | HR_gt | err BPM):")
        for c in details_filtered:
            hp = c.get("hr_pred")
            hg = c.get("hr_gt")
            er = c.get("error")
            if hp is not None and hg is not None:
                summary_lines.append("  {} clip{:2d}: pred={:.1f} gt={:.1f} err={:.1f}".format(
                    c["subject"], c["clip_idx"] + 1, hp, hg, er if er is not None else abs(hp - hg)))

    with open(os.path.join(eval_out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n评估结果已保存到: {eval_out_dir}")
    print("  - summary.txt")
    if save_viz:
        print("  - viz_waveform/*.png")
    print("✓ 评估完成")


if __name__ == "__main__":
    main()
