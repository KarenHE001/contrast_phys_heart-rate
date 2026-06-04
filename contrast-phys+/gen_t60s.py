#!/usr/bin/env python3
"""Generate t60s summary plots for ALL sessions using per-camera clip results directly.
No cross-camera resampling - just plot each camera's own pred/GT clips side by side."""
import csv, sys, os, re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_COLOR_MAP = {
    "gt": "#111111",
    "video_RAW_YUV420": "#d1495b",
    "android_311YJP3P3080D200020": "#00798c",
    "android_RFCN3050F7T": "#edae49",
}
_SHORT = {
    "video_RAW_YUV420": "C920",
    "android_311YJP3P3080D200020": "311",
    "android_RFCN3050F7T": "RFCN",
}

def parse_summary_clips(summary_path):
    """Parse clip pred/gt from summary.txt"""
    clips = []
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("clip ") and "pred=" in line and "gt=" in line:
                parts = line.split()
                pred = float(parts[2].split("=")[1])
                gt = float(parts[4].split("=")[1])
                clips.append({"pred": pred, "gt": gt})
    return clips

def main():
    run_root = Path(sys.argv[1])
    viz_dir = run_root / "common_gt" / "viz_t60s"
    viz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_root / "camera_session_results.csv"
    rows = list(csv.DictReader(open(csv_path)))

    camera_order = ["video_RAW_YUV420", "android_311YJP3P3080D200020", "android_RFCN3050F7T"]

    grouped = defaultdict(dict)
    for r in rows:
        if r.get("status") == "ok":
            grouped[(r["subject"], r["session"])][r["camera_key"]] = r

    generated = 0
    for (subj, sess), by_cam in sorted(grouped.items()):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)

        all_clips_by_cam = {}
        max_clips = 0
        for cam in camera_order:
            if cam not in by_cam:
                continue
            summary_path = by_cam[cam].get("live_summary_path", "")
            if not summary_path or not os.path.isfile(summary_path):
                continue
            clips = parse_summary_clips(summary_path)
            if clips:
                all_clips_by_cam[cam] = clips
                max_clips = max(max_clips, len(clips))

        if not all_clips_by_cam:
            plt.close(fig)
            continue

        # ── Top: HR per clip ──
        # Use clip index as x (each clip ≈ 10s)
        scale_sec = 10
        for cam in camera_order:
            if cam not in all_clips_by_cam:
                continue
            clips = all_clips_by_cam[cam]
            n = len(clips)
            x_mid = np.array([(i + 0.5) * scale_sec for i in range(n)])
            preds = [c["pred"] for c in clips]
            gts = [c["gt"] for c in clips]
            short = _SHORT.get(cam, cam)
            color = _COLOR_MAP.get(cam)

            # Plot GT (only once, use first available camera's GT)
            if cam == list(all_clips_by_cam.keys())[0]:
                axes[0].plot(x_mid, gts, marker="o", linewidth=2.2,
                            color=_COLOR_MAP["gt"], label="GT", zorder=10)

            mae = np.mean(np.abs(np.array(preds) - np.array(gts)))
            axes[0].plot(x_mid, preds, marker="s", linewidth=1.6,
                        color=color, label=f"{short} (MAE={mae:.1f})", alpha=0.85)

        total_dur = max_clips * scale_sec
        axes[0].set_title(f"{subj}/{sess} | {max_clips} clips | ~{total_dur}s")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("HR (BPM)")
        axes[0].set_xlim(0, total_dur)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)

        # ── Bottom: per-clip error bars ──
        bar_width = 0.25
        n_cams = len(all_clips_by_cam)
        cam_list = [c for c in camera_order if c in all_clips_by_cam]

        for ci, cam in enumerate(cam_list):
            clips = all_clips_by_cam[cam]
            n = len(clips)
            x = np.arange(1, n + 1) + (ci - n_cams / 2 + 0.5) * bar_width
            errs = [abs(c["pred"] - c["gt"]) for c in clips]
            short = _SHORT.get(cam, cam)
            color = _COLOR_MAP.get(cam)
            axes[1].bar(x, errs, width=bar_width, color=color, alpha=0.8, label=short)

        axes[1].set_title("Per-clip absolute error")
        axes[1].set_xlabel("Clip index")
        axes[1].set_ylabel("|pred - GT| (BPM)")
        axes[1].set_xticks(np.arange(1, max_clips + 1))
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].legend(loc="best", fontsize=8)

        out_path = viz_dir / f"{subj}__{sess}__t60_summary.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        generated += 1
        print(f"  {subj}/{sess}: {len(all_clips_by_cam)} cameras, {max_clips} clips")

    print(f"\nGenerated {generated} t60s plots in {viz_dir}")

if __name__ == "__main__":
    main()
