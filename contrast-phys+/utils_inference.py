# -*- coding: utf-8 -*-
"""
共享推理逻辑 - 与 test.py 完全一致

供 test.py 与 live_predict_webcam.py 共同使用，保证两者推理流程完全对齐。
输入格式、数据转换、模型调用方式与 H5Dataset / test.py 一致。
"""
import numpy as np
import torch


def dl_model(model, imgs_clip, device=None):
    """
    模型推理（与 test.py 完全一致）

    输入: imgs_clip [T, H, W, C] - 与 H5 中 imgs 格式一致
      - T: 帧数
      - H, W: 128, 128
      - C: 3 (RGB)
      - 值域: 0-255 (uint8 或 float32 均可，内部会转 float32)

    输出: rppg [T] - 模型输出的 rPPG 波形

    与 utils_data.H5Dataset 及 test.py 保持一致:
      - transpose(3,0,1,2) -> [C,T,H,W]
      - [np.newaxis] -> [1,C,T,H,W]
      - astype float32，不除以 255
      - model(img_batch)[:,-1,:] 取空间平均后的 rPPG
    """
    if device is None:
        device = next(model.parameters()).device
    img_batch = imgs_clip.transpose((3, 0, 1, 2))[np.newaxis].astype("float32")
    img_batch = torch.tensor(img_batch).to(device)
    with torch.no_grad():
        rppg = model(img_batch)[:, -1, :]
    rppg = rppg[0].detach().cpu().numpy()
    return rppg
