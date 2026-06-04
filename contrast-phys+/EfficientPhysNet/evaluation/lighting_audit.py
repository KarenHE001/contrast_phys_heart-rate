#!/usr/bin/env python3
"""
lighting_audit.py – Quantify lighting conditions, flicker, row-banding,
and frame-timing stability for self-recorded rPPG videos.

Outputs:
  lighting_summary.csv            per-video aggregate metrics
  series/*_lighting_series.csv    per-frame time series
  plots/*_lighting.png            brightness time-domain plot
  plots/*_lighting_fft.png        brightness FFT (power spectral density)
  plots/*_frame_timing.png        frame interval + aligned brightness plot

Optionally generates lighting-perturbed video variants for sensitivity testing:
  --write-variants --variant exposure=0.7 --variant gamma=1.35 --variant flicker=0.06,10
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as sig

# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------


def _roi_center(frame: np.ndarray) -> np.ndarray:
    """Central 1/4 area of the frame."""
    h, w = frame.shape[:2]
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    return frame[y1:y2, x1:x2]


def _roi_full(frame: np.ndarray) -> np.ndarray:
    return frame


def _load_openface_landmarks(csv_path: str) -> Optional[np.ndarray]:
    """Load OpenFace landmark CSV, return Nx68x2 array or None."""
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    # Column naming conventions: ' x_0'...' x_67' or 'x_0'...'x_67'
    xp = "x_" if "x_0" in df.columns else " x_"
    yp = "y_" if "y_0" in df.columns else " y_"
    sc = "success" if "success" in df.columns else " success"
    n = len(df)
    lms = np.zeros((n, 68, 2), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        if not df[sc].iloc[i]:
            continue
        valid[i] = True
        for j in range(68):
            lms[i, j, 0] = df[xp + str(j)].iloc[i]
            lms[i, j, 1] = df[yp + str(j)].iloc[i]
    return lms if valid.any() else None


def _roi_openface(frame: np.ndarray, landmarks_68: np.ndarray) -> np.ndarray:
    """Crop face bounding box from 68 landmarks (x, y)."""
    h, w = frame.shape[:2]
    xs = landmarks_68[:, 0]
    ys = landmarks_68[:, 1]
    x1 = max(0, int(xs.min()) - 10)
    y1 = max(0, int(ys.min()) - 10)
    x2 = min(w, int(xs.max()) + 10)
    y2 = min(h, int(ys.max()) + 10)
    if x2 <= x1 or y2 <= y1:
        return _roi_center(frame)
    return frame[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Timestamp loading
# ---------------------------------------------------------------------------


def _load_timestamps(ts_path: str) -> Optional[np.ndarray]:
    """Load frame timestamps (seconds) from CSV (column index 1)."""
    ts_list: List[float] = []
    try:
        with open(ts_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        ts_list.append(float(row[1]))
                    except ValueError:
                        continue
    except Exception:
        return None
    if len(ts_list) < 10:
        return None
    return np.array(ts_list)


def _find_timestamp_csv(
    video_path: str, timestamps_dir: Optional[str]
) -> Optional[str]:
    """Try to find a matching timestamp CSV for the video."""
    if timestamps_dir is None:
        return None
    stem = Path(video_path).stem
    parent_name = Path(video_path).parent.name
    # Try direct name match
    for candidate in [
        os.path.join(timestamps_dir, stem + "_frames_timestamp.csv"),
        os.path.join(timestamps_dir, stem + ".csv"),
        os.path.join(timestamps_dir, parent_name, stem + "_frames_timestamp.csv"),
        os.path.join(timestamps_dir, "frames_timestamp.csv"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    # Search in timestamps_dir for any csv containing the video stem
    for f in Path(timestamps_dir).rglob("*frames_timestamp.csv"):
        if stem in str(f):
            return str(f)
    return None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_video(
    video_path: str,
    roi_mode: str = "center",
    landmarks_path: Optional[str] = None,
    timestamps_path: Optional[str] = None,
    expected_fps: float = 30.0,
) -> dict:
    """Analyze a single video and return metrics dict + time series arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or expected_fps

    # Load landmarks if openface
    lms = None
    if roi_mode == "openface" and landmarks_path:
        lms = _load_openface_landmarks(landmarks_path)

    # Load timestamps
    timestamps = None
    if timestamps_path:
        timestamps = _load_timestamps(timestamps_path)

    # --- Read frames and compute per-frame luma ---
    luma_series: List[float] = []
    row_banding_series: List[float] = []
    dark_pct_series: List[float] = []
    bright_pct_series: List[float] = []

    last_lm = None
    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Select ROI
        if roi_mode == "openface" and lms is not None and idx < len(lms):
            if lms[idx].sum() > 0:
                last_lm = lms[idx]
            if last_lm is not None:
                roi = _roi_openface(frame, last_lm)
            else:
                roi = _roi_center(frame)
        elif roi_mode == "center":
            roi = _roi_center(frame)
        else:
            roi = _roi_full(frame)

        # Convert to grayscale for luma
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float64)

        luma_series.append(gray.mean())
        dark_pct_series.append(100.0 * np.mean(gray < 30))
        bright_pct_series.append(100.0 * np.mean(gray > 225))

        # Row banding: std of row means (horizontal banding indicator)
        row_means = gray.mean(axis=1)
        row_banding_series.append(row_means.std())

    cap.release()

    luma = np.array(luma_series)
    row_band = np.array(row_banding_series)
    n = len(luma)
    if n < 30:
        raise ValueError(f"Too few frames ({n}) in {video_path}")

    # --- Compute aggregate metrics ---
    mean_luma = float(luma.mean())
    std_luma = float(luma.std())
    luma_cv_pct = 100.0 * std_luma / mean_luma if mean_luma > 0 else 0.0
    p05 = float(np.percentile(luma, 5))
    p95 = float(np.percentile(luma, 95))
    luma_drift_pct = 100.0 * (p95 - p05) / mean_luma if mean_luma > 0 else 0.0

    # Frame-to-frame delta
    delta_luma = np.abs(np.diff(luma))
    frame_delta_luma_pct = (
        100.0 * delta_luma.mean() / mean_luma if mean_luma > 0 else 0.0
    )

    dark_pixel_pct = float(np.mean(dark_pct_series))
    bright_pixel_pct = float(np.mean(bright_pct_series))

    # Row banding
    row_banding_mean = float(row_band.mean())
    row_banding_p95 = float(np.percentile(row_band, 95))

    # --- Frequency analysis of brightness ---
    # Use timestamps-derived fs if available, else metadata fps
    if timestamps is not None and len(timestamps) >= n:
        ts = timestamps[:n]
        dt_arr = np.diff(ts)
        fs_actual = 1.0 / np.median(dt_arr) if np.median(dt_arr) > 0 else expected_fps
    else:
        ts = None
        dt_arr = None
        fs_actual = fps_meta if fps_meta > 0 else expected_fps

    # Detrend luma for FFT
    luma_detrended = luma - np.convolve(
        luma, np.ones(int(fs_actual * 2)) / int(fs_actual * 2), mode="same"
    )

    # Welch PSD
    nperseg = min(512, n // 2)
    if nperseg < 32:
        nperseg = 32
    freqs, psd = sig.welch(luma_detrended, fs=fs_actual, nperseg=nperseg)

    # Flicker detection: peak in 1-15 Hz range (above Nyquist may alias)
    nyquist = fs_actual / 2.0
    flicker_mask = (freqs >= 1.0) & (freqs <= min(nyquist - 0.5, 15.0))
    if flicker_mask.any():
        peak_idx = np.argmax(psd[flicker_mask])
        flicker_peak_hz = float(freqs[flicker_mask][peak_idx])
        flicker_peak_power = float(psd[flicker_mask][peak_idx])
        median_power = float(np.median(psd[flicker_mask]))
        flicker_peak_ratio = (
            flicker_peak_power / median_power if median_power > 0 else 0.0
        )
    else:
        flicker_peak_hz = 0.0
        flicker_peak_ratio = 0.0

    # Power in physiological band (0.6-4 Hz = 36-240 BPM)
    phys_mask = (freqs >= 0.6) & (freqs <= 4.0)
    total_mask = freqs > 0.1
    phys_power = np.trapz(psd[phys_mask], freqs[phys_mask]) if phys_mask.any() else 0.0
    total_power = (
        np.trapz(psd[total_mask], freqs[total_mask]) if total_mask.any() else 1e-10
    )
    brightness_phys_band_ratio = float(phys_power / total_power)

    # 50 Hz / 100 Hz alias detection
    # At 30 fps, 50 Hz aliases to |50 - 30| = 20 Hz → still above Nyquist for 30fps
    # But 50 Hz - 30 = 20, 50 - 60 = -10 → alias at 10 Hz? With 30fps Nyquist=15Hz
    # 100 Hz → 100 mod 30 = 10 Hz alias at 10 Hz
    # 120 Hz → 120 mod 30 = 0 Hz (DC) or 120 - 4*30 = 0
    # Actually: f_alias = |f_source - N*fs| where N chosen so alias is in [0, fs/2]
    # 50 Hz @ 30 fps: 50 - 1*30 = 20 > 15, 50 - 2*30 = -10 → |10| < 15 → alias at 10 Hz
    # 100 Hz @ 30 fps: 100 - 3*30 = 10 < 15 → alias at 10 Hz
    # 60 Hz @ 30 fps: 60 - 2*30 = 0 → DC
    # 120 Hz @ 30 fps: 120 - 4*30 = 0 → DC
    alias_50_100_freq = abs(50 - round(50 / fs_actual) * fs_actual)
    alias_60_120_freq = abs(60 - round(60 / fs_actual) * fs_actual)

    def _power_at_freq(f_target: float, bw: float = 0.5) -> float:
        mask = (freqs >= f_target - bw) & (freqs <= f_target + bw)
        if not mask.any():
            return 0.0
        return float(np.max(psd[mask]))

    median_psd = float(np.median(psd[freqs > 0.5])) if (freqs > 0.5).any() else 1e-10
    alias_50_100_ratio = _power_at_freq(alias_50_100_freq) / median_psd
    alias_60_120_ratio = _power_at_freq(alias_60_120_freq) / median_psd

    # --- Frame timing metrics ---
    interval_mean_ms = 0.0
    interval_std_ms = 0.0
    interval_spike_pct = 0.0
    interval_max_ms = 0.0
    abs_delta_luma_interval_corr = 0.0

    if dt_arr is not None and len(dt_arr) > 10:
        dt_ms = dt_arr * 1000.0
        # Filter out extreme outliers (> 1 second)
        valid_dt = dt_ms[dt_ms < 1000.0]
        if len(valid_dt) > 10:
            interval_mean_ms = float(np.mean(valid_dt))
            interval_std_ms = float(np.std(valid_dt))
            expected_dt = 1000.0 / expected_fps
            interval_spike_pct = float(
                100.0 * np.mean(valid_dt > 1.5 * expected_dt)
            )
            interval_max_ms = float(np.max(valid_dt))

            # Correlation between brightness jumps and interval spikes
            min_len = min(len(delta_luma), len(dt_ms))
            if min_len > 10:
                dl = delta_luma[:min_len]
                di = dt_ms[:min_len]
                if np.std(dl) > 1e-10 and np.std(di) > 1e-10:
                    abs_delta_luma_interval_corr = float(
                        np.corrcoef(dl, di)[0, 1]
                    )

    metrics = {
        "video": os.path.basename(video_path),
        "n_frames": n,
        "fps_metadata": round(fps_meta, 2),
        "fps_actual": round(fs_actual, 2),
        "mean_luma": round(mean_luma, 2),
        "std_luma_over_time": round(std_luma, 4),
        "luma_cv_pct": round(luma_cv_pct, 3),
        "luma_drift_pct_p95_p05": round(luma_drift_pct, 3),
        "frame_delta_luma_pct": round(frame_delta_luma_pct, 4),
        "dark_pixel_pct": round(dark_pixel_pct, 2),
        "bright_pixel_pct": round(bright_pixel_pct, 2),
        "row_banding_mean": round(row_banding_mean, 3),
        "row_banding_p95": round(row_banding_p95, 3),
        "flicker_peak_hz": round(flicker_peak_hz, 3),
        "flicker_peak_ratio": round(flicker_peak_ratio, 2),
        "brightness_phys_band_ratio": round(brightness_phys_band_ratio, 5),
        "alias_50_100_ratio": round(alias_50_100_ratio, 2),
        "alias_60_120_ratio": round(alias_60_120_ratio, 2),
        "interval_mean_ms": round(interval_mean_ms, 2),
        "interval_std_ms": round(interval_std_ms, 2),
        "interval_spike_pct_1p5x": round(interval_spike_pct, 2),
        "interval_max_ms": round(interval_max_ms, 2),
        "abs_delta_luma_interval_corr": round(abs_delta_luma_interval_corr, 4),
    }

    series = {
        "luma": luma,
        "luma_detrended": luma_detrended,
        "row_banding": row_band,
        "dark_pct": np.array(dark_pct_series),
        "bright_pct": np.array(bright_pct_series),
        "freqs": freqs,
        "psd": psd,
        "timestamps": ts,
        "dt_ms": dt_arr * 1000.0 if dt_arr is not None else None,
        "delta_luma": delta_luma,
        "fs_actual": fs_actual,
    }

    return metrics, series


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_lighting(
    metrics: dict, series: dict, output_path: str, expected_fps: float
) -> None:
    """Generate lighting time-domain plot."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    n = len(series["luma"])
    fs = series["fs_actual"]
    t = np.arange(n) / fs

    # Raw luma
    ax = axes[0]
    ax.plot(t, series["luma"], linewidth=0.5, color="goldenrod")
    ax.set_ylabel("Mean Luma (0–255)")
    ax.set_title(
        f"{metrics['video']}  |  mean={metrics['mean_luma']:.1f}  "
        f"CV={metrics['luma_cv_pct']:.2f}%  drift={metrics['luma_drift_pct_p95_p05']:.2f}%"
    )
    ax.axhline(metrics["mean_luma"], color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Detrended luma (AC component)
    ax = axes[1]
    ax.plot(t, series["luma_detrended"], linewidth=0.5, color="steelblue")
    ax.set_ylabel("Detrended Luma")
    ax.set_title("AC component (slow trend removed)")
    ax.grid(True, alpha=0.3)

    # Row banding
    ax = axes[2]
    ax.plot(t, series["row_banding"], linewidth=0.5, color="crimson")
    ax.set_ylabel("Row Banding (std of row means)")
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"Row Banding  |  mean={metrics['row_banding_mean']:.2f}  "
        f"p95={metrics['row_banding_p95']:.2f}"
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lighting_fft(
    metrics: dict, series: dict, output_path: str, expected_fps: float
) -> None:
    """Generate brightness FFT / PSD plot."""
    import matplotlib.pyplot as plt

    freqs = series["freqs"]
    psd = series["psd"]
    fs = series["fs_actual"]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(freqs, psd, linewidth=0.8, color="navy")

    # Highlight physiological band
    ax.axvspan(0.6, 4.0, alpha=0.1, color="red", label="rPPG band (0.6–4 Hz)")

    # Mark flicker peak
    if metrics["flicker_peak_ratio"] > 3.0:
        ax.axvline(
            metrics["flicker_peak_hz"],
            color="orange",
            linestyle="--",
            label=f"Flicker peak: {metrics['flicker_peak_hz']:.1f} Hz "
            f"(ratio={metrics['flicker_peak_ratio']:.1f}x)",
        )

    # Mark alias frequencies
    alias_50 = abs(50 - round(50 / fs) * fs)
    alias_60 = abs(60 - round(60 / fs) * fs)
    if alias_50 > 0.1:
        ax.axvline(
            alias_50, color="purple", linestyle=":", alpha=0.6, label=f"50Hz alias → {alias_50:.1f} Hz"
        )
    if alias_60 > 0.1 and abs(alias_60 - alias_50) > 0.3:
        ax.axvline(
            alias_60, color="green", linestyle=":", alpha=0.6, label=f"60Hz alias → {alias_60:.1f} Hz"
        )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (power/Hz)")
    ax.set_title(
        f"{metrics['video']}  |  Brightness PSD  "
        f"|  phys_band_ratio={metrics['brightness_phys_band_ratio']:.4f}"
    )
    ax.legend(loc="upper right")
    ax.set_xlim(0, min(fs / 2, 15))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frame_timing(
    metrics: dict, series: dict, output_path: str, expected_fps: float
) -> None:
    """Generate aligned frame interval + brightness jump plot."""
    import matplotlib.pyplot as plt

    dt_ms = series["dt_ms"]
    if dt_ms is None:
        return  # No timestamps available

    n_dt = len(dt_ms)
    fs = series["fs_actual"]
    # Time axis for intervals (midpoints between frames)
    ts = series["timestamps"]
    if ts is not None and len(ts) > 1:
        t_mid = (ts[1:] + ts[:-1]) / 2.0
        t_mid -= t_mid[0]
    else:
        t_mid = np.arange(n_dt) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    expected_dt = 1000.0 / expected_fps

    # Frame intervals
    ax = axes[0]
    ax.plot(t_mid[:n_dt], dt_ms[:n_dt], linewidth=0.5, color="steelblue", alpha=0.8)
    ax.axhline(expected_dt, color="green", linestyle="--", alpha=0.7, label=f"Expected {expected_dt:.1f} ms")
    ax.axhline(
        1.5 * expected_dt,
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"1.5× threshold ({1.5 * expected_dt:.1f} ms)",
    )
    ax.set_ylabel("Interval (ms)")
    ax.set_title(
        f"{metrics['video']}  |  Frame Intervals  "
        f"|  mean={metrics['interval_mean_ms']:.1f} ms  "
        f"std={metrics['interval_std_ms']:.1f} ms  "
        f"spikes={metrics['interval_spike_pct_1p5x']:.1f}%"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Brightness (luma) aligned
    ax = axes[1]
    luma = series["luma"]
    n_luma = len(luma)
    if ts is not None and len(ts) >= n_luma:
        t_luma = ts[:n_luma] - ts[0]
    else:
        t_luma = np.arange(n_luma) / fs
    ax.plot(t_luma, luma, linewidth=0.5, color="goldenrod")
    ax.set_ylabel("Mean Luma")
    ax.set_title("ROI Brightness (aligned with frame timing)")
    ax.grid(True, alpha=0.3)

    # Delta luma vs interval spike overlay
    ax = axes[2]
    dl = series["delta_luma"]
    min_len = min(len(dl), n_dt)
    ax.scatter(
        t_mid[:min_len],
        dl[:min_len],
        s=2,
        alpha=0.5,
        c="goldenrod",
        label="|ΔLuma|",
    )
    ax2 = ax.twinx()
    ax2.scatter(
        t_mid[:min_len],
        dt_ms[:min_len],
        s=2,
        alpha=0.3,
        c="steelblue",
        label="Δt (ms)",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|ΔLuma|", color="goldenrod")
    ax2.set_ylabel("Δt (ms)", color="steelblue")
    ax.set_title(
        f"Brightness Jump vs Frame Interval  "
        f"|  correlation={metrics['abs_delta_luma_interval_corr']:.3f}"
    )
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Variant generation (sensitivity test)
# ---------------------------------------------------------------------------


def _apply_exposure(frame: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _apply_gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    table = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, table)


def _apply_flicker(
    frame: np.ndarray, amplitude: float, freq_hz: float, frame_idx: int, fps: float
) -> np.ndarray:
    """Simulate sinusoidal flicker at freq_hz."""
    t = frame_idx / fps
    modulation = 1.0 + amplitude * np.sin(2 * np.pi * freq_hz * t)
    return np.clip(frame.astype(np.float32) * modulation, 0, 255).astype(np.uint8)


def write_variant(
    video_path: str,
    output_path: str,
    variant_type: str,
    variant_params: str,
    expected_fps: float = 30.0,
) -> None:
    """Generate a lighting-perturbed copy of the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or expected_fps
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if variant_type == "exposure":
            frame = _apply_exposure(frame, float(variant_params))
        elif variant_type == "gamma":
            frame = _apply_gamma(frame, float(variant_params))
        elif variant_type == "flicker":
            parts = variant_params.split(",")
            amp = float(parts[0])
            freq = float(parts[1])
            frame = _apply_flicker(frame, amp, freq, idx, fps)
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    print(f"  Written variant: {output_path} ({idx} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def find_videos(input_path: str, pattern: str) -> List[str]:
    """Find video files matching pattern in input_path."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    videos = []
    for ext_pattern in pattern.split(","):
        ext_pattern = ext_pattern.strip()
        videos.extend(str(f) for f in p.rglob(ext_pattern))
    videos.sort()
    return videos


def find_landmarks_csv(video_path: str, landmarks_dir: Optional[str]) -> Optional[str]:
    """Find matching OpenFace landmarks CSV."""
    if landmarks_dir is None:
        return None
    stem = Path(video_path).stem
    for candidate in Path(landmarks_dir).rglob("*.csv"):
        if stem in candidate.stem:
            return str(candidate)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lighting & frame-timing audit for rPPG videos"
    )
    parser.add_argument(
        "input",
        help="Path to a single video or directory of videos",
    )
    parser.add_argument(
        "--pattern",
        default="*.avi,*.mp4,*.mov,*.mkv",
        help="Glob pattern(s) for video files (comma-separated)",
    )
    parser.add_argument(
        "--roi",
        choices=["center", "full", "openface"],
        default="center",
        help="ROI selection mode",
    )
    parser.add_argument(
        "--landmarks-dir",
        default=None,
        help="Directory containing OpenFace landmark CSVs (for --roi openface)",
    )
    parser.add_argument(
        "--timestamps-dir",
        default=None,
        help="Directory containing frame timestamp CSVs",
    )
    parser.add_argument(
        "--expected-fps",
        type=float,
        default=30.0,
        help="Expected capture FPS (for jitter threshold calculation)",
    )
    parser.add_argument(
        "--output-dir",
        default="./lighting_audit_output",
        help="Output directory for CSVs and plots",
    )
    parser.add_argument(
        "--write-variants",
        action="store_true",
        help="Generate lighting-perturbed video variants",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant spec: exposure=0.7, gamma=1.35, flicker=0.06,10",
    )

    args = parser.parse_args()

    # Create output directories
    out_dir = Path(args.output_dir)
    series_dir = out_dir / "series"
    plots_dir = out_dir / "plots"
    variants_dir = out_dir / "variants"
    for d in [out_dir, series_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find videos
    videos = find_videos(args.input, args.pattern)
    if not videos:
        print(f"ERROR: No videos found matching '{args.pattern}' in {args.input}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to analyze")
    print(f"ROI mode: {args.roi}")
    print(f"Expected FPS: {args.expected_fps}")
    print(f"Output: {out_dir}")
    print()

    # Process each video
    all_metrics: List[dict] = []

    for i, vpath in enumerate(videos):
        stem = Path(vpath).stem
        print(f"[{i + 1}/{len(videos)}] {vpath}")

        # Find associated files
        lm_path = find_landmarks_csv(vpath, args.landmarks_dir)
        ts_path = _find_timestamp_csv(vpath, args.timestamps_dir)

        if args.roi == "openface" and lm_path:
            print(f"  Landmarks: {lm_path}")
        if ts_path:
            print(f"  Timestamps: {ts_path}")

        try:
            metrics, series = analyze_video(
                vpath,
                roi_mode=args.roi,
                landmarks_path=lm_path,
                timestamps_path=ts_path,
                expected_fps=args.expected_fps,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        all_metrics.append(metrics)

        # Save per-video series CSV
        series_path = series_dir / f"{stem}_lighting_series.csv"
        n = len(series["luma"])
        with open(series_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["frame", "time_s", "luma", "luma_detrended", "row_banding", "dark_pct", "bright_pct"]
            )
            fs = series["fs_actual"]
            ts_arr = series["timestamps"]
            for j in range(n):
                t = (ts_arr[j] - ts_arr[0]) if ts_arr is not None and j < len(ts_arr) else j / fs
                writer.writerow([
                    j,
                    round(t, 4),
                    round(series["luma"][j], 3),
                    round(series["luma_detrended"][j], 4),
                    round(series["row_banding"][j], 3),
                    round(series["dark_pct"][j], 2),
                    round(series["bright_pct"][j], 2),
                ])

        # Generate plots
        try:
            plot_lighting(metrics, series, str(plots_dir / f"{stem}_lighting.png"), args.expected_fps)
            plot_lighting_fft(metrics, series, str(plots_dir / f"{stem}_lighting_fft.png"), args.expected_fps)
            if series["dt_ms"] is not None:
                plot_frame_timing(metrics, series, str(plots_dir / f"{stem}_frame_timing.png"), args.expected_fps)
        except Exception as e:
            print(f"  Plot error: {e}")

        # Print key metrics
        print(f"  Luma: mean={metrics['mean_luma']:.1f}, CV={metrics['luma_cv_pct']:.2f}%, "
              f"drift={metrics['luma_drift_pct_p95_p05']:.2f}%")
        print(f"  Flicker: peak={metrics['flicker_peak_hz']:.1f}Hz, "
              f"ratio={metrics['flicker_peak_ratio']:.1f}x, "
              f"phys_band={metrics['brightness_phys_band_ratio']:.4f}")
        print(f"  Row banding: mean={metrics['row_banding_mean']:.2f}, p95={metrics['row_banding_p95']:.2f}")
        if metrics["interval_mean_ms"] > 0:
            print(f"  Timing: mean={metrics['interval_mean_ms']:.1f}ms, "
                  f"std={metrics['interval_std_ms']:.1f}ms, "
                  f"spikes={metrics['interval_spike_pct_1p5x']:.1f}%, "
                  f"luma-interval corr={metrics['abs_delta_luma_interval_corr']:.3f}")
        print()

    # Write summary CSV
    if all_metrics:
        summary_path = out_dir / "lighting_summary.csv"
        fieldnames = list(all_metrics[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"Summary written: {summary_path} ({len(all_metrics)} videos)")

    # Generate variants if requested
    if args.write_variants and args.variant:
        variants_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating {len(args.variant)} variant(s) for {len(videos)} video(s)...")
        for vpath in videos:
            stem = Path(vpath).stem
            for vspec in args.variant:
                parts = vspec.split("=", 1)
                if len(parts) != 2:
                    print(f"  Invalid variant spec: {vspec}")
                    continue
                vtype, vparams = parts[0].strip(), parts[1].strip()
                if vtype not in ("exposure", "gamma", "flicker"):
                    print(f"  Unknown variant type: {vtype}")
                    continue
                out_name = f"{stem}_{vtype}_{vparams.replace(',', '_').replace('.', 'p')}.avi"
                out_path = str(variants_dir / out_name)
                try:
                    write_variant(vpath, out_path, vtype, vparams, args.expected_fps)
                except Exception as e:
                    print(f"  Variant error ({vspec}): {e}")

    # Print diagnostic summary
    if all_metrics:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        n_total = len(all_metrics)

        # Flag potential issues
        issues = {
            "high_luma_cv": [m for m in all_metrics if m["luma_cv_pct"] > 3.0],
            "high_phys_band": [m for m in all_metrics if m["brightness_phys_band_ratio"] > 0.3],
            "flicker_detected": [m for m in all_metrics if m["flicker_peak_ratio"] > 5.0],
            "row_banding": [m for m in all_metrics if m["row_banding_p95"] > 5.0],
            "high_jitter": [m for m in all_metrics if m["interval_spike_pct_1p5x"] > 5.0],
            "luma_interval_corr": [m for m in all_metrics if abs(m["abs_delta_luma_interval_corr"]) > 0.3],
        }

        labels = {
            "high_luma_cv": "Lighting instability (CV > 3%)",
            "high_phys_band": "Brightness in rPPG band (ratio > 0.3)",
            "flicker_detected": "Flicker detected (peak ratio > 5×)",
            "row_banding": "Row banding (p95 > 5)",
            "high_jitter": "Frame timing jitter (spikes > 5%)",
            "luma_interval_corr": "Brightness-timing correlation (|r| > 0.3)",
        }

        for key, flagged in issues.items():
            count = len(flagged)
            print(f"  {labels[key]}: {count}/{n_total} videos")
            if count > 0 and count <= 5:
                for m in flagged:
                    print(f"    - {m['video']}")

        print("\n" + "-" * 70)
        print("Interpretation guide:")
        print("  • high_luma_cv → auto-exposure active or unstable lighting")
        print("  • high_phys_band → lighting changes may contaminate rPPG signal")
        print("  • flicker_detected → LED/PWM/power-line flicker present")
        print("  • row_banding → rolling shutter + PWM interaction")
        print("  • high_jitter → camera pipeline dropping/duplicating frames")
        print("  • luma_interval_corr → brightness jumps coincide with timing issues")
        print("    (suggests auto-exposure triggered by frame skip)")
        print("-" * 70)


if __name__ == "__main__":
    main()
