#!/usr/bin/env python3
"""
gen_all_viz.py  — Generate t10s (waveform+PSD) and t60s (HR trend+error bar) plots
for ALL sessions from existing inference results. No re-inference needed.

Usage:
    python gen_all_viz.py <run_root>

Output:
    <run_root>/viz/t10s/{Subject}__{Session}__clip_{N:03d}.png
    <run_root>/viz/t60s/{Subject}__{Session}__t60_summary.png
"""
import json, csv, sys, os, glob
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from scipy import signal as sp_signal
from scipy.signal import butter, filtfilt

# ── Config ──────────────────────────────────────────────────────────────────
CAMERA_ORDER = ["video_RAW_YUV420", "android_311YJP3P3080D200020", "android_RFCN3050F7T"]
SHORT = {"video_RAW_YUV420": "C920", "android_311YJP3P3080D200020": "311", "android_RFCN3050F7T": "RFCN"}
COLORS = {"gt": "#111111", "video_RAW_YUV420": "#d1495b",
          "android_311YJP3P3080D200020": "#00798c", "android_RFCN3050F7T": "#edae49"}
FS = 30
DPI = 150

# ── Signal helpers ───────────────────────────────────────────────────────────
def bandpass(sig, fs=30, lo=0.6, hi=4.0):
    sig = np.asarray(sig, dtype=np.float32).ravel()
    if len(sig) < 16: return sig
    b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)

def normalize(sig):
    sig = np.asarray(sig, dtype=np.float32).ravel()
    sig = sig - sig.mean()
    s = sig.std()
    return sig / s if s > 1e-8 else sig

def compute_psd_bpm(sig, fs=30):
    sig = np.asarray(sig, dtype=np.float64).ravel()
    N = len(sig)
    win = sig * sp_signal.windows.hann(N)
    mag = np.abs(fft(win))[:N // 2]
    freq_bpm = np.arange(N // 2) / N * fs * 60
    return freq_bpm, mag

def norm_mag(mag):
    mx = mag.max()
    return mag / mx if mx > 1e-9 else mag

# ── Data loading ─────────────────────────────────────────────────────────────
def find_live_run(cam_dir):
    """Return the most recent live_run directory."""
    runs = sorted(glob.glob(str(Path(cam_dir) / "live_runs" / "*")))
    return runs[-1] if runs else None

def load_camera_clips(live_run_dir, scale="t10s"):
    """Load clip metadata + rppg + bvp from a live_run directory."""
    live_run_dir = Path(live_run_dir)
    info = json.loads((live_run_dir / "inference_by_scale.json").read_text())
    clips_meta = info.get(scale, {}).get("clips", [])

    npy_path = live_run_dir / f"eval_bundle_{scale}" / "session.npy"
    bundle = np.load(str(npy_path), allow_pickle=True)
    if bundle.shape == (): bundle = bundle.item()
    rppg_list = list(bundle.get("rppg_list", []))
    bvp_list  = list(bundle.get("bvp_list", []))

    clips = []
    for i, meta in enumerate(clips_meta):
        if i >= len(rppg_list): break
        rppg = np.asarray(rppg_list[i], dtype=np.float32).ravel()
        gt   = np.asarray(bvp_list[i],  dtype=np.float32).ravel() if i < len(bvp_list) else None
        clips.append({
            "idx": int(meta.get("clip_idx", i + 1)),
            "t_start": float(meta.get("t_start_s", 0)),
            "t_end":   float(meta.get("t_end_s", 10)),
            "time_start_wall": float(meta.get("time_start_wall", 0)),
            "time_end_wall":   float(meta.get("time_end_wall", 0)),
            "hr_pred": float(meta.get("hr_pred_bpm", 0)),
            "hr_gt":   float(meta.get("hr_gt_bpm", 0)) if meta.get("hr_gt_bpm") is not None else None,
            "hr_err":  float(meta.get("hr_err_bpm", 0)) if meta.get("hr_err_bpm") is not None else None,
            "rppg": rppg,
            "gt_bvp": gt,
        })
    return clips

# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_t10s(out_path, clip_idx, subject, session, scale_sec, per_cam_clip):
    """
    per_cam_clip: dict cam -> clip dict (containing rppg, gt_bvp, hr_pred, hr_gt, hr_err)
    """
    fig, (ax_wave, ax_psd) = plt.subplots(2, 1, figsize=(12, 7.5), constrained_layout=True)

    # ── Waveform ──
    # plot GT from first available camera
    gt_plotted = False
    for cam in CAMERA_ORDER:
        if cam not in per_cam_clip: continue
        clip = per_cam_clip[cam]
        if clip["gt_bvp"] is not None and not gt_plotted:
            gt_sig = normalize(bandpass(clip["gt_bvp"]))
            t = np.linspace(0, scale_sec, len(gt_sig))
            hr_gt = clip["hr_gt"]
            label = f"GT ({hr_gt:.1f} BPM)" if hr_gt is not None else "GT"
            ax_wave.plot(t, gt_sig, color=COLORS["gt"], linewidth=2.2, label=label, zorder=10)
            gt_plotted = True
        break

    for cam in CAMERA_ORDER:
        if cam not in per_cam_clip: continue
        clip = per_cam_clip[cam]
        rppg_f = bandpass(clip["rppg"])
        sig = normalize(rppg_f)
        t = np.linspace(0, scale_sec, len(sig))
        short = SHORT.get(cam, cam)
        hr_err = clip["hr_err"]
        label = f"{short} ({clip['hr_pred']:.1f} BPM, err {hr_err:.1f})" if hr_err is not None else f"{short} ({clip['hr_pred']:.1f} BPM)"
        ax_wave.plot(t, sig, color=COLORS[cam], linewidth=1.5, alpha=0.9, label=label)

    ax_wave.set_title(f"{subject}/{session}  |  clip {clip_idx:02d}  |  {scale_sec}s window")
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Normalized amplitude")
    ax_wave.set_xlim(0, scale_sec)
    ax_wave.grid(True, alpha=0.25)
    ax_wave.legend(loc="upper right", fontsize=8)

    # ── Normalized PSD ──
    bpm_min, bpm_max = 36, 240
    # GT PSD
    gt_plotted_psd = False
    for cam in CAMERA_ORDER:
        if cam not in per_cam_clip: continue
        clip = per_cam_clip[cam]
        if clip["gt_bvp"] is not None and not gt_plotted_psd:
            gt_f = bandpass(clip["gt_bvp"])
            freq, mag = compute_psd_bpm(gt_f, FS)
            mask = (freq >= bpm_min) & (freq <= bpm_max)
            mag_n = norm_mag(mag[mask])
            ax_psd.plot(freq[mask], mag_n, color=COLORS["gt"], linewidth=2.0,
                        label=f"GT ({clip['hr_gt']:.1f} BPM)" if clip["hr_gt"] else "GT")
            # GT peak line
            peak_bpm = freq[mask][np.argmax(mag_n)]
            ax_psd.axvline(peak_bpm, color=COLORS["gt"], linestyle="--", alpha=0.35, linewidth=1)
            gt_plotted_psd = True
        break

    for cam in CAMERA_ORDER:
        if cam not in per_cam_clip: continue
        clip = per_cam_clip[cam]
        rppg_f = bandpass(clip["rppg"])
        actual_fs = FS  # model always outputs at 30fps
        freq, mag = compute_psd_bpm(rppg_f, actual_fs)
        mask = (freq >= bpm_min) & (freq <= bpm_max)
        mag_n = norm_mag(mag[mask])
        color = COLORS[cam]
        short = SHORT.get(cam, cam)
        ax_psd.plot(freq[mask], mag_n, color=color, linewidth=1.4, alpha=0.85,
                    label=f"{short} ({clip['hr_pred']:.1f} BPM)")
        # mark selected peak with triangle
        closest = np.argmin(np.abs(freq[mask] - clip["hr_pred"]))
        ax_psd.plot(freq[mask][closest], mag_n[closest], marker="v",
                    markersize=9, color=color, zorder=5)

    ax_psd.set_title("Frequency domain (normalized PSD)  —  ▼ = predicted HR")
    ax_psd.set_xlabel("Heart rate (BPM)")
    ax_psd.set_ylabel("Normalized magnitude")
    ax_psd.set_xlim(bpm_min, bpm_max)
    ax_psd.set_ylim(-0.05, 1.15)
    ax_psd.grid(True, alpha=0.25)
    ax_psd.legend(loc="upper right", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_t60s(out_path, subject, session, all_cam_clips):
    """
    all_cam_clips: dict cam -> list of clip dicts
    """
    scale_sec = 10
    fig, (ax_hr, ax_err) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

    cam_list = [c for c in CAMERA_ORDER if c in all_cam_clips and all_cam_clips[c]]
    max_clips = max(len(all_cam_clips[c]) for c in cam_list)

    # ── Top: HR trend ──
    ref_cam = cam_list[0]
    ref_clips = all_cam_clips[ref_cam]
    gt_x = np.array([(i + 0.5) * scale_sec for i in range(len(ref_clips))])
    gt_y = [c["hr_gt"] for c in ref_clips if c["hr_gt"] is not None]
    if gt_y:
        ax_hr.plot(gt_x[:len(gt_y)], gt_y, color=COLORS["gt"], marker="o",
                   linewidth=2.2, label="GT", zorder=10)

    for cam in cam_list:
        clips = all_cam_clips[cam]
        x = np.array([(i + 0.5) * scale_sec for i in range(len(clips))])
        y = [c["hr_pred"] for c in clips]
        mae = np.mean([c["hr_err"] for c in clips if c["hr_err"] is not None])
        short = SHORT.get(cam, cam)
        ax_hr.plot(x, y, color=COLORS[cam], marker="s", linewidth=1.6, alpha=0.85,
                   label=f"{short} (MAE={mae:.1f})")

    total_dur = max_clips * scale_sec
    ax_hr.set_title(f"{subject}/{session}  |  {max_clips} clips  |  ~{total_dur}s")
    ax_hr.set_xlabel("Time (s)")
    ax_hr.set_ylabel("HR (BPM)")
    ax_hr.set_xlim(0, total_dur)
    ax_hr.grid(True, alpha=0.25)
    ax_hr.legend(loc="best", fontsize=8)

    # ── Bottom: per-clip error bars ──
    n_cams = len(cam_list)
    bar_w = 0.7 / n_cams
    for ci, cam in enumerate(cam_list):
        clips = all_cam_clips[cam]
        n = len(clips)
        x = np.arange(1, n + 1) + (ci - n_cams / 2 + 0.5) * bar_w
        errs = [c["hr_err"] if c["hr_err"] is not None else 0 for c in clips]
        short = SHORT.get(cam, cam)
        ax_err.bar(x, errs, width=bar_w, color=COLORS[cam], alpha=0.8, label=short)

    ax_err.set_title("Per-clip absolute error  |pred - GT|")
    ax_err.set_xlabel("Clip index")
    ax_err.set_ylabel("|pred − GT| (BPM)")
    ax_err.set_xticks(np.arange(1, max_clips + 1))
    axes_xticklabels = [f"{i}\n({i*scale_sec-scale_sec+1}-{i*scale_sec}s)" for i in range(1, max_clips + 1)]
    ax_err.set_xticklabels(axes_xticklabels, fontsize=7)
    ax_err.grid(True, axis="y", alpha=0.25)
    ax_err.legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    run_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "results/EfficientPhysNet/label_ratio_0/camera_compare/20260415_095210"
    )
    per_cam_root = run_root / "per_camera"
    viz_t10 = run_root / "viz" / "t10s"
    viz_t60 = run_root / "viz" / "t60s"
    viz_t10.mkdir(parents=True, exist_ok=True)
    viz_t60.mkdir(parents=True, exist_ok=True)

    # Discover all sessions
    sessions = []
    for subj_dir in sorted(per_cam_root.iterdir()):
        if not subj_dir.is_dir(): continue
        for sess_dir in sorted(subj_dir.iterdir()):
            if not sess_dir.is_dir(): continue
            sessions.append((subj_dir.name, sess_dir.name, sess_dir))

    t10_total = t60_total = 0

    for subject, session, sess_dir in sessions:
        # Load all cameras
        all_cam_clips = {}
        for cam in CAMERA_ORDER:
            cam_dir = sess_dir / cam
            if not cam_dir.exists(): continue
            live_run = find_live_run(cam_dir)
            if not live_run: continue
            try:
                clips = load_camera_clips(live_run)
                if clips:
                    all_cam_clips[cam] = clips
            except Exception as e:
                print(f"  WARN {subject}/{session}/{cam}: {e}")

        if not all_cam_clips:
            print(f"  SKIP {subject}/{session}: no data")
            continue

        # ── t10s: plot per-clip aligned groups ──
        # Use C920 as reference if available, else first camera
        ref_cam = next((c for c in CAMERA_ORDER if c in all_cam_clips), None)
        ref_clips = all_cam_clips[ref_cam]
        n_ref = len(ref_clips)

        for i, ref_clip in enumerate(ref_clips):
            clip_idx = ref_clip["idx"]
            ref_start = ref_clip["time_start_wall"]
            ref_end   = ref_clip["time_end_wall"]
            ref_dur   = ref_end - ref_start

            per_cam_clip = {}
            # ref camera
            per_cam_clip[ref_cam] = ref_clip

            # match other cameras by time overlap
            for cam in CAMERA_ORDER:
                if cam == ref_cam or cam not in all_cam_clips: continue
                best = None
                best_ov = 0
                for cc in all_cam_clips[cam]:
                    ov = max(0, min(ref_end, cc["time_end_wall"]) - max(ref_start, cc["time_start_wall"]))
                    if ov > best_ov:
                        best_ov = ov
                        best = cc
                if best is not None and best_ov > ref_dur * 0.4:
                    per_cam_clip[cam] = best

            out_path = viz_t10 / f"{subject}__{session}__clip_{clip_idx:03d}.png"
            try:
                plot_t10s(out_path, clip_idx, subject, session, ref_dur, per_cam_clip)
                t10_total += 1
            except Exception as e:
                print(f"  WARN t10s {subject}/{session} clip {clip_idx}: {e}")

        # ── t60s ──
        out_path = viz_t60 / f"{subject}__{session}__t60_summary.png"
        try:
            plot_t60s(out_path, subject, session, all_cam_clips)
            t60_total += 1
        except Exception as e:
            print(f"  WARN t60s {subject}/{session}: {e}")

        print(f"  {subject}/{session}: {len(all_cam_clips)} cams, {n_ref} clips")

    print(f"\nDone. t10s: {t10_total} plots, t60s: {t60_total} plots")
    print(f"Output: {run_root / 'viz'}")

if __name__ == "__main__":
    main()
