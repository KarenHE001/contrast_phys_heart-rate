#!/usr/bin/env python3
"""Debug: check why partial t60s generation isn't working for skipped sessions."""
import csv, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "EfficientPhysNet", "evaluation"))
sys.path.insert(0, os.path.dirname(__file__) or ".")
from common_gt_compare import load_clip_payload

run_root = sys.argv[1] if len(sys.argv) > 1 else "results/EfficientPhysNet/label_ratio_0/camera_compare/20260415_095210"
csv_path = os.path.join(run_root, "camera_session_results.csv")
rows = list(csv.DictReader(open(csv_path)))

# Group by subject/session
from collections import defaultdict
grouped = defaultdict(dict)
for r in rows:
    if r.get("status") == "ok":
        grouped[(r["subject"], r["session"])][r["camera_key"]] = r

for (subj, sess), by_cam in sorted(grouped.items()):
    cams = list(by_cam.keys())
    for cam in cams:
        r = by_cam[cam]
        try:
            p = load_clip_payload(r["live_run_dir"], 10, subj, sess, cam)
            n_clips = len(p["clips"])
            has_timing = all("time_start_wall" in c for c in p["clips"]) if p["clips"] else False
            print(f"  {subj:15s} {sess} {cam:35s} clips={n_clips} timing={has_timing}")
        except Exception as e:
            print(f"  {subj:15s} {sess} {cam:35s} ERROR: {e}")
