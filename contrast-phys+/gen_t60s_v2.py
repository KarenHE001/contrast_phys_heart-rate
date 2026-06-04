#!/usr/bin/env python3
"""Generate t60s waveform + PSD plots for ALL sessions.
Top subplot: concatenated ~60s normalized rPPG waveforms (per camera) + GT BVP
Bottom subplot: PSD (frequency domain) with HR peak markers
"""
import csv, sys, os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

_COLOR_MAP = {
    "gt": "#111111",
    "video_RAW_YUV420": "#d1495b",
    "android_311YJP3P3080D200020": "#00798c",
    "android_RFCN3050F7T": "#edae49",
}
_SHORT = {
    "video_RAW_YUV420": "C920",
    "android_311YJP3P3080D200020": "Android 311",
    "android_RFCN3050F7T": "Android RFCN",
}
CAMERA_ORDER = ["video_RAW_YUV420", "android_311YJP3P3080D200020", "android_RFCN3050F7T"]
FS = 30.0  # model output sample rate


def normalize_01(x):
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return x - mn
    return (x - mn) / (mx - mn)


def compute_psd(signal, fs):
    """Return (freqs_bpm, normalized_psd, peak_bpm) using Welch's method."""
    # Use Welch with ~10s segments (nperseg=300 at 30fps) to avoid clip boundary artifacts
    nperseg = min(len(signal), int(fs * 10))
    freqs, pxx = welch(signal - np.mean(signal), fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2, window='hann')
    # Restrict to physiological range 0.5-4 Hz (30-240 BPM)
    mask = (freqs >= 0.5) & (freqs <= 4.0)
    freqs_m = freqs[mask]
    pxx_m = pxx[mask]
    freqs_bpm = freqs_m * 60.0
    # Normalize to [0,1]
    pxx_norm = normalize_01(pxx_m) if pxx_m.max() > 0 else pxx_m
    peak_idx = np.argmax(pxx_norm)
    peak_bpm = freqs_bpm[peak_idx]
    return freqs_bpm, pxx_norm, peak_bpm


def load_session_waveform(session_npy_path):
    """Load session.npy and concatenate clips into a single waveform.
    Each clip is individually zero-meaned to avoid boundary discontinuities."""
    d = np.load(session_npy_path, allow_pickle=True).item()
    rppg = np.array(d['rppg_list'], dtype=float)  # (n_clips, 300)
    bvp = np.array(d['bvp_list'], dtype=float)    # (n_clips, 300)
    # Zero-mean each clip independently to reduce boundary jumps
    rppg_clips = [rppg[i] - np.mean(rppg[i]) for i in range(rppg.shape[0])]
    bvp_clips = [bvp[i] - np.mean(bvp[i]) for i in range(bvp.shape[0])]
    rppg_cat = np.concatenate(rppg_clips)
    bvp_cat = np.concatenate(bvp_clips)
    return rppg_cat, bvp_cat


def plot_t60s(session_data, subj, sess, out_path):
    """
    session_data: dict of cam_key -> (rppg_waveform, bvp_waveform)
    """
    fig, (ax_wave, ax_psd) = plt.subplots(2, 1, figsize=(16, 8),
                                           gridspec_kw={'height_ratios': [1, 1]})

    # Use GT from the camera with most clips (usually C920)
    gt_src = None
    gt_len = 0
    for cam in CAMERA_ORDER:
        if cam in session_data:
            _, bvp = session_data[cam]
            if len(bvp) > gt_len:
                gt_len = len(bvp)
                gt_src = bvp

    if gt_src is None:
        plt.close(fig)
        return False

    # â”€â”€ Top: Waveform â”€â”€
    gt_norm = normalize_01(gt_src)
    t_gt = np.arange(len(gt_src)) / FS
    ax_wave.plot(t_gt, gt_norm, color=_COLOR_MAP['gt'], alpha=0.5,
                 linewidth=1.0, label='GT (BVP)')

    for cam in CAMERA_ORDER:
        if cam not in session_data:
            continue
        rppg, _ = session_data[cam]
        rppg_norm = normalize_01(rppg)
        t_cam = np.arange(len(rppg)) / FS
        ax_wave.plot(t_cam, rppg_norm, color=_COLOR_MAP[cam], alpha=0.7,
                     linewidth=0.8, label=_SHORT[cam])

    ax_wave.set_xlabel('Time (s)')
    ax_wave.set_ylabel('Normalized Amplitude')
    ax_wave.set_title(f'{subj} / {sess} - Full Session Waveform (~60s)')
    ax_wave.legend(loc='upper right', fontsize=8)
    ax_wave.set_xlim(0, t_gt[-1])

    # â”€â”€ Bottom: PSD â”€â”€
    f_gt, psd_gt, bpm_gt = compute_psd(gt_src, FS)
    ax_psd.plot(f_gt, psd_gt, color=_COLOR_MAP['gt'], alpha=0.5,
                linewidth=1.0, label=f'GT ({bpm_gt:.0f} BPM)')
    ax_psd.axvline(bpm_gt, color=_COLOR_MAP['gt'], linestyle=':', alpha=0.4, linewidth=0.8)

    for cam in CAMERA_ORDER:
        if cam not in session_data:
            continue
        rppg, _ = session_data[cam]
        f_cam, psd_cam, bpm = compute_psd(rppg, FS)
        ax_psd.plot(f_cam, psd_cam, color=_COLOR_MAP[cam], alpha=0.7,
                    linewidth=0.8, label=f'{_SHORT[cam]} ({bpm:.0f} BPM)')
        ax_psd.axvline(bpm, color=_COLOR_MAP[cam], linestyle=':', alpha=0.4, linewidth=0.8)

    ax_psd.set_xlabel('Heart Rate (BPM)')
    ax_psd.set_ylabel('Normalized PSD')
    ax_psd.set_title('Power Spectral Density')
    ax_psd.set_xlim(30, 240)
    ax_psd.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    run_root = Path(sys.argv[1])
    viz_dir = run_root / "common_gt" / "viz_t60s"
    viz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_root / "camera_session_results.csv"
    rows = list(csv.DictReader(open(csv_path)))

    # Group by (subject, session)
    grouped = defaultdict(dict)
    for r in rows:
        if r.get("status") == "ok" and r.get("eval_bundle_dir"):
            grouped[(r["subject"], r["session"])][r["camera_key"]] = r

    generated = 0
    skipped = 0
    for (subj, sess), by_cam in sorted(grouped.items()):
        session_data = {}
        for cam in CAMERA_ORDER:
            if cam not in by_cam:
                continue
            bundle_dir = by_cam[cam]["eval_bundle_dir"]
            npy_path = os.path.join(bundle_dir, "session.npy")
            if not os.path.isfile(npy_path):
                print(f"  [SKIP] {subj}/{sess}/{cam}: session.npy not found")
                continue
            try:
                rppg, bvp = load_session_waveform(npy_path)
                session_data[cam] = (rppg, bvp)
            except Exception as e:
                print(f"  [ERR] {subj}/{sess}/{cam}: {e}")

        if not session_data:
            skipped += 1
            continue

        out_name = f"{subj}__{sess}__t60_waveform_psd.png"
        out_path = viz_dir / out_name
        ok = plot_t60s(session_data, subj, sess, out_path)
        if ok:
            generated += 1
            print(f"  [OK] {out_path.name}")
        else:
            skipped += 1

    print(f"\nDone: {generated} generated, {skipped} skipped")


if __name__ == "__main__":
    main()
