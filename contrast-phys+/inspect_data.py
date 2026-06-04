#!/usr/bin/env python3
"""Inspect data structures needed for viz generation."""
import json, numpy as np, os

base = "results/EfficientPhysNet/label_ratio_0/camera_compare/20260415_095210"
live_run = f"{base}/per_camera/Alvin/v01/video_RAW_YUV420/live_runs/2026-04-09_09-53-05"

# inference_by_scale.json
d = json.load(open(f"{live_run}/inference_by_scale.json"))
c = d["t10s"]["clips"][0]
print("=== inference_by_scale.json clip[0] keys ===")
for k, v in c.items():
    print(f"  {k}: {v}")

# session.npy
npy = np.load(f"{live_run}/eval_bundle_t10s/session.npy", allow_pickle=True).item()
print("\n=== eval_bundle_t10s/session.npy keys ===")
for k, v in npy.items():
    if hasattr(v, "__len__"):
        print(f"  {k}: len={len(v)}, type={type(v[0]) if len(v) else '?'}")
    else:
        print(f"  {k}: {v}")

# Android 311 - check time_start_wall vs BVP range
live_run_311 = f"{base}/per_camera/Alvin/v01/android_311YJP3P3080D200020/live_runs/2026-04-09_09-52-13"
d311 = json.load(open(f"{live_run_311}/inference_by_scale.json"))
c311 = d311["t10s"]["clips"][0]
print("\n=== Android 311 clip[0] time_start_wall ===")
print(f"  time_start_wall: {c311.get('time_start_wall')}")
print(f"  time_end_wall: {c311.get('time_end_wall')}")

# BVP range
import pandas as pd
bvp = pd.read_csv("results/EfficientPhysNet/label_ratio_0/camera_compare/20260415_095210/gt_proxy/Alvin/v01/video_RAW_YUV420/BVP.csv")
print(f"\n=== BVP range ===")
print(f"  t_start: {bvp.iloc[0,0]:.2f}  t_end: {bvp.iloc[-1,0]:.2f}")
