# -*- coding: utf-8 -*-
"""
EfficientPhysNet 推理 — 使用 pretrained 权重（多尺度训练 curriculum 表现最佳）

运行: cd contrast-phys+ && python EfficientPhysNet/test/test_epn.py with strategy=curriculum time_interval=10
strategy: curriculum(默认) | equal | loss_prop | hybrid | inv_loss
time_interval: clip 长度(秒)，10=T=300 与 96x96 部署目标一致
"""
import os
import sys
import json
import time

_EPN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CP = os.path.dirname(_EPN)  # contrast-phys+
_PRETRAINED = os.path.join(_CP, "pretrained", "EfficientPhysNet")
sys.path.insert(0, _CP)
sys.path.insert(0, os.path.join(_CP, "PhysNet 2D"))

import numpy as np
import h5py
import torch
from EfficientPhysNet import EfficientPhysNet
from utils_inference import dl_model
from utils_paths import format_label_ratio
from sacred import Experiment

ex = Experiment('epn_pred', save_git_info=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')


@ex.config
def my_config():
    strategy       = 'curriculum'   # curriculum(最佳) | equal | loss_prop | hybrid | inv_loss
    time_interval  = 10             # T=300 与 96x96 部署目标一致
    e              = None          # None -> use best_model.pt
    pretrained_dir = os.path.join(_PRETRAINED, strategy)


@ex.automain
def my_main(_run, strategy, time_interval, e, pretrained_dir):
    infer_t0 = time.perf_counter()
    config_path = os.path.join(pretrained_dir, 'config.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    weight_path = os.path.join(pretrained_dir, 'best_model.pt') if e is None else os.path.join(pretrained_dir, f'epoch{e}.pt')
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weight not found: {weight_path}")

    test_list_path = os.path.join(pretrained_dir, 'test_list.npy')
    if not os.path.isfile(test_list_path):
        raise FileNotFoundError(f"test_list not found: {test_list_path}")
    test_list_raw = np.load(test_list_path, allow_pickle=True)
    # 路径相对 contrast-phys+
    test_list = [os.path.normpath(os.path.join(_CP, str(p))) for p in test_list_raw]

    input_size = config.get('input_size', 128)
    model = EfficientPhysNet(config['S'], in_ch=config['in_ch'], input_size=input_size).to(device).eval()
    model.load_state_dict(torch.load(weight_path, map_location=device))

    label_ratio = config.get('label_ratio', 0)
    run_id = int(_run._id) if _run._id is not None else 1
    pred_dir = os.path.join(_CP, "results", "EfficientPhysNet", format_label_ratio(label_ratio),
                            "inference", strategy, f"t{time_interval}", str(run_id))
    os.makedirs(pred_dir, exist_ok=True)

    fs = config.get('fs', 30)
    n_subjects_saved = 0
    n_clips_total = 0
    for h5_path in test_list:
        if not os.path.isfile(h5_path):
            print(f"  Skip (not found): {h5_path}")
            continue
        h5_path = str(h5_path)
        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs']
            bvp = f['bvp']
            duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
            num_blocks = int(duration // time_interval)
            rppg_list, bvp_list = [], []
            for b in range(num_blocks):
                imgs_clip = imgs[b * time_interval * fs:(b + 1) * time_interval * fs]
                rppg_clip = dl_model(model, np.array(imgs_clip), device)
                rppg_list.append(rppg_clip)
                bvp_list.append(np.array(bvp[b * time_interval * fs:(b + 1) * time_interval * fs]))
            results = {'rppg_list': np.array(rppg_list), 'bvp_list': np.array(bvp_list)}
            out_name = os.path.basename(h5_path).replace('.h5', '')
            np.save(os.path.join(pred_dir, out_name), results)
            print(f"  Saved: {out_name}.npy ({len(rppg_list)} clips)")
            n_subjects_saved += 1
            n_clips_total += len(rppg_list)

    infer_sec = time.perf_counter() - infer_t0
    meta = {
        "strategy": strategy,
        "time_interval_sec": int(time_interval),
        "fs": int(fs),
        "input_size": int(input_size),
        "label_ratio": float(label_ratio),
        "run_id": int(run_id),
        "weight_path": weight_path,
        "device": str(device),
        "subjects_saved": int(n_subjects_saved),
        "clips_total": int(n_clips_total),
        "inference_seconds": float(infer_sec),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(pred_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nPred saved to: {pred_dir}")
    print(f"Inference time: {infer_sec:.2f}s | input_size: {input_size} | clips: {n_clips_total}")
    print(f"Next: python EfficientPhysNet/evaluation/evaluate.py {pred_dir}")
