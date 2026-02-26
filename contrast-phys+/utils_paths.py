# -*- coding: utf-8 -*-
"""统一路径逻辑：label_ratio -> 实验目录名，避免硬编码"""
import os
import glob

RESULTS_ROOT = "results"
TRAINING_LOGS_ROOT = "training_logs"


def format_label_ratio(r):
    """0->label_ratio_0, 0.5->label_ratio_0.5, 1->label_ratio_1"""
    s = f"{float(r):.2f}".rstrip("0").rstrip(".")
    return f"label_ratio_{s}"


def get_exp_root(label_ratio):
    """results/label_ratio_X"""
    return os.path.join(RESULTS_ROOT, format_label_ratio(label_ratio))


def get_train_exp_dir(label_ratio, run_id):
    """results/label_ratio_X/<run_id>"""
    return os.path.join(get_exp_root(label_ratio), str(run_id))


def _latest_run_in_root(root):
    dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.isdigit()]
    if not dirs:
        return None
    return os.path.join(root, max(dirs, key=int))


def find_latest_run(label_ratio):
    """返回 label_ratio 下最新 run 的完整路径，无则 None"""
    root = get_exp_root(label_ratio)
    if not os.path.isdir(root):
        return None
    return _latest_run_in_root(root)


def get_live_runs_subdir(train_exp_dir):
    """从 train_exp_dir 推断 label_ratio，返回 live_runs 下子目录名。如 results/label_ratio_1/2 -> label_ratio_1"""
    if not train_exp_dir:
        return "label_ratio_0"
    path = os.path.normpath(train_exp_dir)
    for part in path.replace("\\", "/").split("/"):
        if part.startswith("label_ratio_"):
            return part
    return "label_ratio_0"


def find_latest_run_any():
    """遍历所有 label_ratio_*，返回最新 run（按 mtime）"""
    if not os.path.isdir(RESULTS_ROOT):
        return None
    candidates = []
    for name in os.listdir(RESULTS_ROOT):
        if name.startswith("label_ratio_"):
            root = os.path.join(RESULTS_ROOT, name)
            p = _latest_run_in_root(root)
            if p and os.path.exists(p):
                candidates.append((p, os.path.getmtime(p)))
    return max(candidates, key=lambda x: x[1])[0] if candidates else None
