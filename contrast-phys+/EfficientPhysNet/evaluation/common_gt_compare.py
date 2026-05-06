#!/usr/bin/env python3
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy import signal as sp_signal

CP = Path(__file__).resolve().parents[2]
if str(CP) not in sys.path:
    sys.path.insert(0, str(CP))

from utils_sig import butter_bandpass, compute_fft_peaks, hr_fft_parabolic, select_hr_from_peaks

_USE_HARMONICS = True
_COLOR_MAP = {
    "gt": "#111111",
    "video_RAW_YUV420": "#d1495b",
    "android_311YJP3P3080D200020": "#00798c",
    "android_RFCN3050F7T": "#edae49",
}


def mean_or_none(values):
    cleaned = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        cleaned.append(value)
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def rmse_or_none(values):
    cleaned = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isnan(value):
            continue
        cleaned.append(value)
    if not cleaned:
        return None
    arr = np.asarray(cleaned, dtype=np.float64)
    return float(np.sqrt(np.mean(arr ** 2)))


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def load_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_object_npy(path):
    obj = np.load(str(path), allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        obj = obj.item()
    return obj


def load_eval_bundle(bundle_dir):
    bundle_path = Path(bundle_dir) / "session.npy"
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    obj = load_object_npy(bundle_path)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected bundle type: {type(obj)}")
    return obj


def load_clip_payload(live_run_dir, scale_sec, subject, session, camera_key):
    live_run_dir = Path(live_run_dir)
    info = load_json(live_run_dir / "inference_by_scale.json")
    meta = load_json(live_run_dir / "meta.json")
    key = f"t{scale_sec}s"
    clips_meta = info.get(key, {}).get("clips", [])
    bundle = load_eval_bundle(live_run_dir / f"eval_bundle_{key}")
    rppg_list = list(bundle.get("rppg_list", []))

    clips = []
    for index, clip_meta in enumerate(clips_meta):
        if index >= len(rppg_list):
            break
        start_wall = clip_meta.get("time_start_wall")
        end_wall = clip_meta.get("time_end_wall")
        if start_wall is None or end_wall is None:
            continue
        rppg = np.asarray(rppg_list[index], dtype=np.float32).reshape(-1)
        if rppg.size < 8:
            continue
        clips.append(
            {
                "subject": subject,
                "session": session,
                "camera_key": camera_key,
                "clip_idx": int(clip_meta.get("clip_idx", index + 1)),
                "time_start_wall": float(start_wall),
                "time_end_wall": float(end_wall),
                "duration_sec": float(end_wall) - float(start_wall),
                "hr_pred_native_bpm": clip_meta.get("hr_pred_bpm"),
                "rppg": rppg,
            }
        )

    return {
        "clips": clips,
        "fs": int(meta.get("fs", 30)),
        "input_size": int(meta.get("input_size", 96)),
    }


def load_session_gt_pack(session_dir):
    session_dir = Path(session_dir)
    bvp_path = session_dir / "BVP.csv"
    if not bvp_path.is_file():
        raise FileNotFoundError(bvp_path)

    bvp_df = pd.read_csv(bvp_path)
    t_bvp = bvp_df.iloc[:, 0].values.astype(np.float64)
    bvp_raw = bvp_df.iloc[:, 1].values.astype(np.float64)
    interp_bvp = interp1d(
        t_bvp,
        bvp_raw,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return {
        "interp_bvp": interp_bvp,
        "time_min": float(np.min(t_bvp)),
        "time_max": float(np.max(t_bvp)),
    }


def _estimate_hr(sig_filtered, fs_hz):
    hr_raw, _, _ = hr_fft_parabolic(
        sig_filtered, fs=fs_hz, harmonics_removal=_USE_HARMONICS
    )
    peaks = compute_fft_peaks(sig_filtered, fs_hz)
    hr_sel, _ = select_hr_from_peaks(
        peaks, fs_hz, len(sig_filtered), use_harmonics_removal=_USE_HARMONICS
    )
    return float(hr_sel if hr_sel is not None else hr_raw)


def bandpass_signal(sig, fs_hz):
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if sig.size < 8:
        return None
    try:
        return butter_bandpass(sig, lowcut=0.6, highcut=4.0, fs=fs_hz)
    except Exception:
        return None


def normalize_signal(sig):
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if sig.size == 0:
        return sig
    sig = sig - np.mean(sig)
    std = float(np.std(sig))
    if std < 1e-8:
        return sig
    return sig / std


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def choose_reference_camera(clips_by_camera, camera_order):
    return min(camera_order, key=lambda camera: len(clips_by_camera[camera]))


def build_relative_timeline(payload_by_camera, camera_order):
    relative = {}
    first_starts = []
    for camera in camera_order:
        clips = payload_by_camera[camera]["clips"]
        if not clips:
            relative[camera] = []
            continue
        offset = clips[0]["time_start_wall"]
        first_starts.append(offset)
        rel_clips = []
        for clip in clips:
            rel_clip = dict(clip)
            rel_clip["match_start_sec"] = float(clip["time_start_wall"] - offset)
            rel_clip["match_end_sec"] = float(clip["time_end_wall"] - offset)
            rel_clips.append(rel_clip)
        relative[camera] = rel_clips
    gt_anchor_wall = float(np.median(first_starts)) if first_starts else 0.0
    return relative, gt_anchor_wall


def match_common_clip_groups(clips_by_camera, camera_order, scale_sec):
    min_overlap_sec = max(scale_sec * 0.8, scale_sec - 2.0)
    reference_camera = choose_reference_camera(clips_by_camera, camera_order)
    used = {camera: set() for camera in camera_order if camera != reference_camera}
    groups = []

    for ref_clip in sorted(clips_by_camera[reference_camera], key=lambda item: item["match_start_sec"]):
        matched = {reference_camera: ref_clip}
        starts = [ref_clip["match_start_sec"]]
        ends = [ref_clip["match_end_sec"]]
        ok = True
        for camera in camera_order:
            if camera == reference_camera:
                continue
            best_clip = None
            best_overlap = 0.0
            for candidate in clips_by_camera[camera]:
                if candidate["clip_idx"] in used[camera]:
                    continue
                cur_overlap = overlap_seconds(
                    ref_clip["match_start_sec"],
                    ref_clip["match_end_sec"],
                    candidate["match_start_sec"],
                    candidate["match_end_sec"],
                )
                if cur_overlap > best_overlap:
                    best_clip = candidate
                    best_overlap = cur_overlap
            if best_clip is None or best_overlap < min_overlap_sec:
                ok = False
                break
            matched[camera] = best_clip
            starts.append(best_clip["match_start_sec"])
            ends.append(best_clip["match_end_sec"])
        if not ok:
            continue

        shared_start = max(starts)
        shared_end = min(ends)
        shared_duration = shared_end - shared_start
        if shared_duration < min_overlap_sec:
            continue

        for camera in camera_order:
            if camera == reference_camera:
                continue
            used[camera].add(matched[camera]["clip_idx"])

        groups.append(
            {
                "shared_start_sec": float(shared_start),
                "shared_end_sec": float(shared_end),
                "shared_duration_sec": float(shared_duration),
                "clips": matched,
            }
        )
    return groups


def resample_signal_to_window(sig, src_start, src_end, dst_times):
    sig = np.asarray(sig, dtype=np.float32).reshape(-1)
    if sig.size < 8 or src_end <= src_start:
        return None
    src_times = np.linspace(src_start, src_end, num=sig.size, endpoint=False, dtype=np.float64)
    if dst_times[0] < src_times[0] - 1e-6 or dst_times[-1] > src_times[-1] + 1e-3:
        return None
    return np.interp(dst_times, src_times, sig).astype(np.float32)


def evaluate_common_group(subject, session, group_idx, group, gt_pack, gt_anchor_wall, camera_order, fs_hz):
    shared_start_sec = group["shared_start_sec"]
    shared_end_sec = group["shared_end_sec"]
    duration = group["shared_duration_sec"]
    n_samples = max(int(round(duration * fs_hz)), 64)
    dst_times_rel = np.linspace(shared_start_sec, shared_end_sec, num=n_samples, endpoint=False, dtype=np.float64)
    dst_times_abs = gt_anchor_wall + dst_times_rel

    gt_wave = gt_pack["interp_bvp"](dst_times_abs).astype(np.float32)
    gt_filtered = bandpass_signal(gt_wave, fs_hz)
    if gt_filtered is None:
        return None, None
    hr_gt = _estimate_hr(gt_filtered, fs_hz)

    row = {
        "subject": subject,
        "session": session,
        "common_clip_idx": group_idx,
        "window_start_rel_s": float(shared_start_sec),
        "window_end_rel_s": float(shared_end_sec),
        "gt_window_start_wall": float(dst_times_abs[0]),
        "gt_window_end_wall": float(dst_times_abs[-1]) if len(dst_times_abs) else float(gt_anchor_wall),
        "window_duration_sec": float(duration),
        "gt_hr_bpm": float(hr_gt),
    }
    payload = {
        "subject": subject,
        "session": session,
        "common_clip_idx": group_idx,
        "window_duration_sec": float(duration),
        "time_sec": dst_times_rel - shared_start_sec,
        "signals": {"gt": gt_filtered},
        "fs": fs_hz,
        "gt_hr_bpm": float(hr_gt),
    }

    err_candidates = []
    for camera in camera_order:
        clip = group["clips"][camera]
        pred_wave = resample_signal_to_window(
            clip["rppg"],
            clip["match_start_sec"],
            clip["match_end_sec"],
            dst_times_rel,
        )
        row[f"{camera}_clip_idx"] = clip["clip_idx"]
        if pred_wave is None:
            row[f"{camera}_pred_hr_bpm"] = None
            row[f"{camera}_err_bpm"] = None
            continue

        pred_filtered = bandpass_signal(pred_wave, fs_hz)
        if pred_filtered is None:
            row[f"{camera}_pred_hr_bpm"] = None
            row[f"{camera}_err_bpm"] = None
            continue

        hr_pred = _estimate_hr(pred_filtered, fs_hz)
        err = abs(hr_pred - hr_gt)
        row[f"{camera}_pred_hr_bpm"] = float(hr_pred)
        row[f"{camera}_err_bpm"] = float(err)
        payload["signals"][camera] = pred_filtered
        err_candidates.append((float(err), camera))

    row["best_camera_by_err"] = min(err_candidates)[1] if err_candidates else ""
    return row, payload


def _compute_psd(sig, fs):
    """Compute one-sided PSD (magnitude) and frequency axis in BPM."""
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    N = len(sig)
    windowed = sig * sp_signal.windows.hann(N)
    mag = np.abs(fft(windowed))[:N // 2]
    freq_hz = np.arange(N // 2) / N * fs
    freq_bpm = freq_hz * 60.0
    return freq_bpm, mag


def save_clip_waveform_plot(plot_path, payload, row, camera_order):
    fs = payload.get("fs", 30)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    ax_wave, ax_psd = axes

    time_sec = payload["time_sec"]

    # ── Top: waveform ──
    gt_sig = normalize_signal(payload["signals"]["gt"])
    ax_wave.plot(time_sec, gt_sig, label=f"GT ({row['gt_hr_bpm']:.1f} BPM)", color=_COLOR_MAP["gt"], linewidth=2.2)

    for camera in camera_order:
        if camera not in payload["signals"]:
            continue
        sig = normalize_signal(payload["signals"][camera])
        hr_val = row.get(f"{camera}_pred_hr_bpm")
        err_val = row.get(f"{camera}_err_bpm")
        label = camera
        if hr_val is not None and err_val is not None:
            label = f"{camera} ({hr_val:.1f} BPM, err {err_val:.1f})"
        ax_wave.plot(time_sec, sig, label=label, linewidth=1.6, alpha=0.95, color=_COLOR_MAP.get(camera))

    ax_wave.set_title(
        f"{row['subject']}/{row['session']} | aligned clip {row['common_clip_idx']:02d} | shared window {row['window_duration_sec']:.2f}s"
    )
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Normalized amplitude")
    ax_wave.grid(True, alpha=0.25)
    ax_wave.legend(loc="upper right", fontsize=8)

    # ── Bottom: frequency domain with peak picking (each signal normalized to [0,1]) ──
    bpm_mask = None

    def _norm_mag(mag):
        mx = np.max(mag)
        return mag / mx if mx > 1e-9 else mag

    # GT PSD
    gt_bpm, gt_mag = _compute_psd(payload["signals"]["gt"], fs)
    bpm_mask = (gt_bpm >= 36) & (gt_bpm <= 240)
    gt_mag_n = _norm_mag(gt_mag[bpm_mask])
    ax_psd.plot(gt_bpm[bpm_mask], gt_mag_n, label=f"GT ({row['gt_hr_bpm']:.1f} BPM)",
                color=_COLOR_MAP["gt"], linewidth=2.0)
    # Mark GT peak
    gt_peak_idx = np.argmax(gt_mag_n)
    ax_psd.axvline(x=gt_bpm[bpm_mask][gt_peak_idx], color=_COLOR_MAP["gt"], linestyle="--", alpha=0.4, linewidth=1.0)

    for camera in camera_order:
        if camera not in payload["signals"]:
            continue
        cam_bpm, cam_mag = _compute_psd(payload["signals"][camera], fs)
        cam_mag_n = _norm_mag(cam_mag[bpm_mask])
        hr_val = row.get(f"{camera}_pred_hr_bpm")
        err_val = row.get(f"{camera}_err_bpm")
        label = camera
        if hr_val is not None and err_val is not None:
            label = f"{camera} ({hr_val:.1f} BPM)"
        color = _COLOR_MAP.get(camera)
        ax_psd.plot(cam_bpm[bpm_mask], cam_mag_n, label=label,
                    linewidth=1.4, alpha=0.85, color=color)
        # Mark selected peak with triangle
        if hr_val is not None:
            closest = np.argmin(np.abs(cam_bpm[bpm_mask] - hr_val))
            ax_psd.plot(cam_bpm[bpm_mask][closest], cam_mag_n[closest],
                        marker="v", markersize=9, color=color, zorder=5)

    ax_psd.set_title("Frequency domain (normalized PSD) — ▼ = selected HR")
    ax_psd.set_xlabel("Heart rate (BPM)")
    ax_psd.set_ylabel("Normalized magnitude")
    ax_psd.set_xlim(36, 240)
    ax_psd.set_ylim(-0.05, 1.15)
    ax_psd.grid(True, alpha=0.25)
    ax_psd.legend(loc="upper right", fontsize=8)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def save_t60_summary_plot(plot_path, session_rows, session_payloads, camera_order, scale_sec):
    """Generate session summary: HR per clip (top) + concatenated waveform (bottom).
    Uses ALL available clips regardless of total duration."""
    if not session_rows or not session_payloads:
        return
    rows = session_rows  # use ALL clips
    payloads = session_payloads

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), constrained_layout=True)

    # ── Top: HR per clip with time on x-axis ──
    # Use clip midpoint time as x position
    clip_mids = []
    for row in rows:
        t_start = row.get("window_start_rel_s", 0)
        t_end = row.get("window_end_rel_s", t_start + scale_sec)
        clip_mids.append((t_start + t_end) / 2.0)
    x = np.array(clip_mids)
    total_dur = max(row.get("window_end_rel_s", scale_sec) for row in rows)

    axes[0].plot(x, [row["gt_hr_bpm"] for row in rows], marker="o", linewidth=2.2,
                 color=_COLOR_MAP["gt"], label="GT")
    present_cameras = set()
    for camera in camera_order:
        y = [row.get(f"{camera}_pred_hr_bpm") for row in rows]
        if any(v is not None for v in y):
            present_cameras.add(camera)
            axes[0].plot(x, y, marker="o", linewidth=1.6, label=camera,
                         color=_COLOR_MAP.get(camera))
    axes[0].set_title(
        f"{rows[0]['subject']}/{rows[0]['session']} | {len(rows)} clips | {total_dur:.0f}s"
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("HR (BPM)")
    axes[0].set_xlim(0, total_dur)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    # ── Bottom: concatenated waveform ──
    concat_times = []
    concat_signals = {"gt": []}
    for camera in camera_order:
        concat_signals[camera] = []
    offset = 0.0
    for payload in payloads:
        time_axis = payload["time_sec"]
        if len(time_axis) == 0:
            continue
        dt = float(time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 1.0 / float(payload["fs"])
        local_time = offset + np.arange(len(time_axis), dtype=np.float64) * dt
        concat_times.append(local_time)
        concat_signals["gt"].append(normalize_signal(payload["signals"]["gt"]))
        for camera in camera_order:
            if camera in payload["signals"]:
                concat_signals[camera].append(normalize_signal(payload["signals"][camera]))
        offset = float(local_time[-1] + dt)

    if concat_times:
        full_time = np.concatenate(concat_times)
        axes[1].plot(full_time, np.concatenate(concat_signals["gt"]), linewidth=2.0,
                     color=_COLOR_MAP["gt"], label="GT")
        for camera in camera_order:
            if concat_signals[camera]:
                axes[1].plot(full_time, np.concatenate(concat_signals[camera]),
                             linewidth=1.2, alpha=0.9, label=camera,
                             color=_COLOR_MAP.get(camera))
        actual_time = float(full_time[-1])
        axes[1].set_title(f"Concatenated waveform ({actual_time:.1f}s)")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_xlim(0, actual_time)
        axes[1].set_ylabel("Normalized amplitude")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper right", fontsize=8)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def build_common_gt_outputs(detail_rows, camera_order, run_root, scale_sec):
    run_root = Path(run_root)
    common_root = run_root / "common_gt"
    per_session_root = common_root / "per_session"
    viz_t10_root = common_root / "viz_t10s"
    viz_t60_root = common_root / "viz_t60s"
    common_root.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(dict)
    for row in detail_rows:
        if row.get("status") != "ok":
            continue
        grouped[(row["subject"], row["session"])][row["camera_key"]] = row

    aligned_clip_rows = []
    session_summary_rows = []
    issues = []
    clip_win_counts = {camera: 0 for camera in camera_order}
    session_win_counts = {camera: 0 for camera in camera_order}

    for (subject, session), by_camera in sorted(grouped.items()):
        if any(camera not in by_camera for camera in camera_order):
            issues.append(f"skip {subject}/{session}: incomplete camera set")
            continue

        try:
            session_dir = Path(by_camera[camera_order[0]]["bvp_path"]).parent
            gt_pack = load_session_gt_pack(session_dir)
            payload_by_camera = {}
            fs_values = set()
            for camera in camera_order:
                payload = load_clip_payload(
                    by_camera[camera]["live_run_dir"],
                    scale_sec,
                    subject,
                    session,
                    camera,
                )
                payload_by_camera[camera] = payload
                fs_values.add(payload["fs"])
            if len(fs_values) != 1:
                issues.append(f"skip {subject}/{session}: inconsistent fs {sorted(fs_values)}")
                continue
            if any(not payload_by_camera[camera]["clips"] for camera in camera_order):
                issues.append(f"skip {subject}/{session}: missing timing-aware clips")
                continue

            fs_hz = fs_values.pop()
            relative_clips, gt_anchor_wall = build_relative_timeline(payload_by_camera, camera_order)
            groups = match_common_clip_groups(relative_clips, camera_order, scale_sec)
            if not groups:
                issues.append(f"skip {subject}/{session}: no common overlapping clips")
                continue

            session_rows = []
            session_payloads = []
            for group_idx, group in enumerate(groups, start=1):
                row, payload = evaluate_common_group(
                    subject,
                    session,
                    group_idx,
                    group,
                    gt_pack,
                    gt_anchor_wall,
                    camera_order,
                    fs_hz,
                )
                if row is None or payload is None:
                    continue
                aligned_clip_rows.append(row)
                session_rows.append(row)
                session_payloads.append(payload)
                if row.get("best_camera_by_err"):
                    clip_win_counts[row["best_camera_by_err"]] += 1
                save_clip_waveform_plot(
                    viz_t10_root / f"{subject}__{session}__clip_{group_idx:03d}.png",
                    payload,
                    row,
                    camera_order,
                )

            if not session_rows:
                issues.append(f"skip {subject}/{session}: common clips could not be evaluated")
                continue

            session_rows = sorted(session_rows, key=lambda item: item["common_clip_idx"])
            session_payloads = sorted(session_payloads, key=lambda item: item["common_clip_idx"])
            session_summary = {
                "subject": subject,
                "session": session,
                "n_common_clips": len(session_rows),
            }
            mae_candidates = []
            for camera in camera_order:
                errs = [row.get(f"{camera}_err_bpm") for row in session_rows]
                mean_mae = mean_or_none(errs)
                rmse = rmse_or_none(errs)
                session_summary[f"{camera}_mean_mae_bpm"] = mean_mae
                session_summary[f"{camera}_rmse_bpm"] = rmse
                if mean_mae is not None:
                    mae_candidates.append((mean_mae, camera))
            session_summary["best_camera_by_mae"] = min(mae_candidates)[1] if mae_candidates else ""
            if session_summary["best_camera_by_mae"]:
                session_win_counts[session_summary["best_camera_by_mae"]] += 1
            session_summary_rows.append(session_summary)

            session_dir_out = per_session_root / subject / session
            session_dir_out.mkdir(parents=True, exist_ok=True)
            clip_fields = [
                "subject",
                "session",
                "common_clip_idx",
                "window_start_rel_s",
                "window_end_rel_s",
                "gt_window_start_wall",
                "gt_window_end_wall",
                "window_duration_sec",
                "gt_hr_bpm",
            ]
            for camera in camera_order:
                clip_fields.extend(
                    [
                        f"{camera}_clip_idx",
                        f"{camera}_pred_hr_bpm",
                        f"{camera}_err_bpm",
                    ]
                )
            clip_fields.append("best_camera_by_err")
            write_csv(session_rows, session_dir_out / "aligned_clip_comparison.csv", clip_fields)
            save_t60_summary_plot(
                viz_t60_root / f"{subject}__{session}__t60_summary.png",
                session_rows,
                session_payloads,
                camera_order,
                scale_sec,
            )
        except Exception as exc:
            issues.append(f"skip {subject}/{session}: {exc}")

    aligned_clip_fields = [
        "subject",
        "session",
        "common_clip_idx",
        "window_start_rel_s",
        "window_end_rel_s",
        "gt_window_start_wall",
        "gt_window_end_wall",
        "window_duration_sec",
        "gt_hr_bpm",
    ]
    for camera in camera_order:
        aligned_clip_fields.extend(
            [
                f"{camera}_clip_idx",
                f"{camera}_pred_hr_bpm",
                f"{camera}_err_bpm",
            ]
        )
    aligned_clip_fields.append("best_camera_by_err")
    write_csv(aligned_clip_rows, common_root / "aligned_clip_comparison.csv", aligned_clip_fields)

    session_fields = ["subject", "session", "n_common_clips"]
    for camera in camera_order:
        session_fields.extend([f"{camera}_mean_mae_bpm", f"{camera}_rmse_bpm"])
    session_fields.append("best_camera_by_mae")
    write_csv(session_summary_rows, common_root / "aligned_session_summary.csv", session_fields)

    camera_summary_rows = []
    for camera in camera_order:
        camera_rows = [row for row in aligned_clip_rows if row.get(f"{camera}_err_bpm") is not None]
        err_values = [row.get(f"{camera}_err_bpm") for row in camera_rows]
        camera_summary_rows.append(
            {
                "camera_key": camera,
                "n_common_clips": len(camera_rows),
                "n_sessions": sum(1 for row in session_summary_rows if row.get(f"{camera}_mean_mae_bpm") is not None),
                "mean_common_mae_bpm": mean_or_none(err_values),
                "rmse_common_bpm": rmse_or_none(err_values),
                "clip_wins": clip_win_counts.get(camera, 0),
                "session_wins": session_win_counts.get(camera, 0),
            }
        )
    write_csv(
        camera_summary_rows,
        common_root / "aligned_camera_summary.csv",
        [
            "camera_key",
            "n_common_clips",
            "n_sessions",
            "mean_common_mae_bpm",
            "rmse_common_bpm",
            "clip_wins",
            "session_wins",
        ],
    )

    summary_lines = []
    summary_lines.append("Common-GT aligned comparison")
    summary_lines.append("=" * 60)
    summary_lines.append("alignment: per-camera relative start -> shared GT anchor")
    summary_lines.append(f"run_root: {run_root}")
    summary_lines.append(f"aligned clips: {len(aligned_clip_rows)}")
    summary_lines.append(f"aligned sessions: {len(session_summary_rows)}")
    summary_lines.append(f"issues: {len(issues)}")
    summary_lines.append("")
    for item in camera_summary_rows:
        summary_lines.append(
            "  {camera}: clips={clips}, sessions={sessions}, MAE={mae}, RMSE={rmse}, clip_wins={clip_wins}, session_wins={session_wins}".format(
                camera=item["camera_key"],
                clips=item["n_common_clips"],
                sessions=item["n_sessions"],
                mae="N/A" if item["mean_common_mae_bpm"] is None else f"{item['mean_common_mae_bpm']:.2f}",
                rmse="N/A" if item["rmse_common_bpm"] is None else f"{item['rmse_common_bpm']:.2f}",
                clip_wins=item["clip_wins"],
                session_wins=item["session_wins"],
            )
        )
    if issues:
        summary_lines.append("")
        summary_lines.append("Issues")
        for issue in issues:
            summary_lines.append(f"  {issue}")
    (common_root / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # ── Generate t60s for sessions that were skipped (partial camera sets) ──
    generated_sessions = {(r["subject"], r["session"]) for r in session_summary_rows}
    for (subject, session), by_camera in sorted(grouped.items()):
        if (subject, session) in generated_sessions:
            continue  # already has a t60s plot
        # Try to build per-camera-only summary from whatever cameras are available
        try:
            available_cameras = [c for c in camera_order if c in by_camera]
            if not available_cameras:
                continue
            session_dir = Path(by_camera[available_cameras[0]]["bvp_path"]).parent
            gt_pack = load_session_gt_pack(session_dir)
            first_camera = available_cameras[0]
            payload_first = load_clip_payload(
                by_camera[first_camera]["live_run_dir"],
                scale_sec, subject, session, first_camera,
            )
            fs_hz = payload_first["fs"]
            if not payload_first["clips"]:
                continue

            # Build payloads for each available camera independently
            payload_by_camera = {}
            for camera in available_cameras:
                payload_by_camera[camera] = load_clip_payload(
                    by_camera[camera]["live_run_dir"],
                    scale_sec, subject, session, camera,
                )

            # Use first camera's clips as reference timeline
            ref_clips = payload_by_camera[first_camera]["clips"]
            partial_rows = []
            partial_payloads = []
            for clip_idx, ref_clip in enumerate(ref_clips, start=1):
                ref_start_rel = ref_clip["time_start_wall"] - ref_clips[0]["time_start_wall"]
                ref_end_rel = ref_clip["time_end_wall"] - ref_clips[0]["time_start_wall"]
                duration = ref_end_rel - ref_start_rel
                n_samples = max(int(round(duration * fs_hz)), 64)
                dst_times_rel = np.linspace(ref_start_rel, ref_end_rel, num=n_samples, endpoint=False, dtype=np.float64)
                dst_times_abs = ref_clips[0]["time_start_wall"] + dst_times_rel

                gt_wave = gt_pack["interp_bvp"](dst_times_abs).astype(np.float32)
                gt_filtered = bandpass_signal(gt_wave, fs_hz)
                if gt_filtered is None:
                    continue
                hr_gt = _estimate_hr(gt_filtered, fs_hz)

                row = {
                    "subject": subject,
                    "session": session,
                    "common_clip_idx": clip_idx,
                    "window_start_rel_s": float(ref_start_rel),
                    "window_end_rel_s": float(ref_end_rel),
                    "window_duration_sec": float(duration),
                    "gt_hr_bpm": float(hr_gt),
                }
                payload = {
                    "subject": subject,
                    "session": session,
                    "common_clip_idx": clip_idx,
                    "window_duration_sec": float(duration),
                    "time_sec": dst_times_rel - ref_start_rel,
                    "signals": {"gt": gt_filtered},
                    "fs": fs_hz,
                    "gt_hr_bpm": float(hr_gt),
                }

                for camera in available_cameras:
                    cam_clips = payload_by_camera[camera]["clips"]
                    # Find best overlapping clip
                    best_clip = None
                    for cc in cam_clips:
                        cc_start_rel = cc["time_start_wall"] - ref_clips[0]["time_start_wall"]
                        cc_end_rel = cc["time_end_wall"] - ref_clips[0]["time_start_wall"]
                        ov = max(0.0, min(ref_end_rel, cc_end_rel) - max(ref_start_rel, cc_start_rel))
                        if ov > duration * 0.5:
                            best_clip = cc
                            break
                    if best_clip is not None:
                        pred_wave = resample_signal_to_window(
                            best_clip["rppg"],
                            best_clip["time_start_wall"] - ref_clips[0]["time_start_wall"],
                            best_clip["time_end_wall"] - ref_clips[0]["time_start_wall"],
                            dst_times_rel,
                        )
                        if pred_wave is not None:
                            pred_filtered = bandpass_signal(pred_wave, fs_hz)
                            if pred_filtered is not None:
                                hr_pred = _estimate_hr(pred_filtered, fs_hz)
                                row[f"{camera}_pred_hr_bpm"] = float(hr_pred)
                                row[f"{camera}_err_bpm"] = abs(float(hr_pred) - float(hr_gt))
                                payload["signals"][camera] = pred_filtered

                partial_rows.append(row)
                partial_payloads.append(payload)

            if partial_rows:
                save_t60_summary_plot(
                    viz_t60_root / f"{subject}__{session}__t60_summary.png",
                    partial_rows,
                    partial_payloads,
                    available_cameras,
                    scale_sec,
                )
        except Exception as exc:
            issues.append(f"t60s partial {subject}/{session}: {exc}")

    return {
        "clip_rows": aligned_clip_rows,
        "session_summary_rows": session_summary_rows,
        "camera_summary_rows": camera_summary_rows,
        "issues": issues,
        "common_root": str(common_root),
    }
