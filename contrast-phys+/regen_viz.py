#!/usr/bin/env python3
"""Re-generate common_gt visualizations for an existing benchmark run."""
import csv, sys, os

run_root = sys.argv[1] if len(sys.argv) > 1 else "results/EfficientPhysNet/label_ratio_0/camera_compare/20260409_095210"

# Add paths
eval_dir = os.path.join(os.path.dirname(__file__), "EfficientPhysNet", "evaluation")
sys.path.insert(0, eval_dir)
sys.path.insert(0, os.path.dirname(__file__))

from common_gt_compare import build_common_gt_outputs

csv_path = os.path.join(run_root, "camera_session_results.csv")
with open(csv_path) as f:
    detail_rows = list(csv.DictReader(f))

camera_order = ["video_RAW_YUV420", "android_311YJP3P3080D200020", "android_RFCN3050F7T"]
scale_sec = 10

# Delete old viz
import shutil
for d in ["common_gt/viz_t10s", "common_gt/viz_t60s"]:
    p = os.path.join(run_root, d)
    if os.path.isdir(p):
        shutil.rmtree(p)
        print(f"Cleared {p}")

result = build_common_gt_outputs(detail_rows, camera_order, run_root, scale_sec)
print("Done. Issues:", result.get("issues", []))
