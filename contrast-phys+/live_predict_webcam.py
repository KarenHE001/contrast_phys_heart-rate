#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================
实时/离线视频 rPPG 预测 + HR 估计
============================================================

功能：
1. 从摄像头或视频文件采集人脸视频
2. 使用训练好的模型预测 rPPG 信号
3. 从 rPPG 估计心率（FFT 方法）
4. 诊断信号质量和峰值选择问题

使用场景：
- 实时摄像头测试：验证模型在真实场景下的表现
- 离线视频测试：分析已录制的视频

主要流程：
Step 1: 采集视频（人脸检测/裁剪 -> 128x128）
Step 2: 模型推理（rPPG 信号提取）
Step 3: 心率估计（FFT）
Step 4: 诊断分析（信号质量、峰值分析）

=============================================================================
【训练一致性检查清单】用于排查 live/recorded 效果差的原因
=============================================================================
训练数据来源: H5 文件，由 preprocess_ubfc.py 生成
  - 人脸: OpenFace landmarks -> bbox_size=1.5*(maxy-miny), y_ext=0.2
  - 颜色: BGR2RGB
  - 尺寸: 128x128
  - 存储: uint8 [0-255]
  - 帧率: 30 fps（数据集中已固定）

训练 DataLoader (utils_data.py H5Dataset):
  - 读取: imgs[idx_start:idx_end] [T,128,128,3]
  - 转置: (3,0,1,2) -> [C,T,H,W]
  - 类型: float32，值域 0-255（不除以255）
  - T: 300 帧 (10秒@30fps)

本脚本需保持一致:
  [1.1] 颜色: BGR2RGB
  [1.2] 裁剪: bbox 计算、np.take、resize 策略
  [1.3] 帧率: 实际 fps 应与 fs=30 接近，否则需重采样或警告
  [2.1] 输入格式: [1,C,T,H,W], float32, 0-255
  [2.2] 帧数: 最多 300 帧
=============================================================================
"""
import os
import sys
import time
import argparse
import json
import subprocess
import glob
import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免显示窗口
import matplotlib.pyplot as plt

# 允许从项目根目录导入（当以 `python evaluation/xxx.py` 方式运行时）
# 或者从 live_test/ 目录直接运行（所有文件在同一目录，Python会自动查找）
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EVAL_DIR)
# OpenFace 默认路径（与 prep 脚本一致，项目根目录下的 OpenFace）
_DEFAULT_OPENFACE_DIR = os.path.join(_PROJECT_ROOT, "OpenFace")
# 如果当前目录已经有需要的模块，就不需要添加父目录
if _PROJECT_ROOT not in sys.path and not os.path.exists(os.path.join(_EVAL_DIR, "PhysNetModel.py")):
    sys.path.insert(0, _PROJECT_ROOT)

from utils_paths import find_latest_run, find_latest_run_any, get_live_runs_subdir
from PhysNetModel import PhysNet
from utils_inference import dl_model
from utils_sig import butter_bandpass, hr_fft_parabolic, hr_fft, SNR_get
from scipy import signal
from scipy.fft import fft
from scipy.interpolate import interp1d


def parse_args():
    parser = argparse.ArgumentParser(description="Live/video rPPG predictor")
    parser.add_argument("--train-exp-dir", default=None,
                        help="训练实验目录。默认 results/label_ratio_0 下最新 run")
    parser.add_argument("--epoch", type=int, default=None,
                        help="指定epoch权重。默认用 best_model.pt")
    parser.add_argument("--source", default="0",
                        help="视频源：摄像头索引(0/1/2)或视频文件路径。--live-ubfc 时可为视频文件跳过录制")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="采集时长（秒）")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="采集目标帧率（与训练fs保持一致）")
    parser.add_argument("--face", action="store_true",
                        help="启用人脸检测裁剪（默认关闭，失败会回退中心裁剪）")
    parser.add_argument("--harmonics", action="store_true",
                        help="启用谐波去除（默认为关闭，与--no-harmonics一致）")
    parser.add_argument("--use-hr-fft", action="store_true",
                        help="使用原始hr_fft（默认使用parabolic高分辨率）")
    parser.add_argument("--save-video", default=None,
                        help="可选：保存采集视频到指定路径")
    parser.add_argument("--record-only", action="store_true",
                        help="仅录制视频并保存，不运行模型。录制后用 OpenFace 处理，再用 --source 视频 --landmarks CSV 运行")
    parser.add_argument("--landmarks", default=None,
                        help="OpenFace 生成的 landmark CSV 路径。当 source 为视频文件时，用 OpenFace 裁剪替代 Haar，与训练一致")
    parser.add_argument("--no-show", action="store_true",
                        help="关闭实时窗口显示")
    parser.add_argument("--save-waveform", action="store_true",
                        help="保存rPPG波形和可视化图（用于检查）")
    parser.add_argument("--full-waveform", action="store_true",
                        help="保存完整采集时长的rPPG波形（按10秒窗口拼接）")
    parser.add_argument("--output-dir", default=None,
                        help="结果输出目录。默认 results/live_runs/label_ratio_X/YYYY-MM-DD_HH-MM-SS")
    parser.add_argument("--openface-dir", default=_DEFAULT_OPENFACE_DIR,
                        help="OpenFace 安装目录（默认: 项目根/OpenFace）")
    parser.add_argument("--live-ubfc", action="store_true",
                        help="UBFC 全流程：录制 -> OpenFace -> 推理 -> viz。--source 视频文件时可跳过录制")
    parser.add_argument("--camera-index", type=int, default=0,
                        help="摄像头索引，0 失败时可试 1 或 2")
    return parser.parse_args()


def load_last_epoch(train_exp_dir):
    """训练目录中最大的 epoch 编号"""
    epoch_files = [f for f in os.listdir(train_exp_dir) if f.startswith("epoch") and f.endswith(".pt")]
    if not epoch_files:
        raise FileNotFoundError(f"未找到权重文件: {train_exp_dir}/epoch*.pt")
    nums = []
    for f in epoch_files:
        try:
            nums.append(int(f.replace("epoch", "").replace(".pt", "")))
        except Exception:
            continue
    if not nums:
        raise ValueError("无法解析epoch编号")
    return max(nums)


def resolve_weight_path(train_exp_dir, epoch_arg):
    """
    解析权重路径：优先 best_model.pt，否则用 epoch{N}.pt
    epoch_arg 指定时强制用 epoch{N}.pt
    """
    best_path = os.path.join(train_exp_dir, "best_model.pt")
    if epoch_arg is not None:
        path = os.path.join(train_exp_dir, f"epoch{epoch_arg}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到权重: {path}")
        return path
    if os.path.exists(best_path):
        return best_path
    last_epoch = load_last_epoch(train_exp_dir)
    path = os.path.join(train_exp_dir, f"epoch{last_epoch}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到权重: {path}")
    return path


def run_openface(video_path, out_dir, openface_dir):
    """
    调用 OpenFace FeatureExtraction 提取 landmarks，与 preprocess_ubfc 流程一致。

    Args:
        video_path: 视频文件路径
        out_dir: CSV 输出目录
        openface_dir: OpenFace 安装目录

    Returns:
        landmark_csv_path: 生成的 CSV 路径
    """
    video_path = os.path.abspath(video_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 查找 FeatureExtraction（常见路径: build/bin/, bin/, local/bin/）
    candidates = [
        "build/bin/FeatureExtraction",
        "bin/FeatureExtraction",
        "FeatureExtraction",
        "local/bin/FeatureExtraction",
    ]
    exe = None
    for name in candidates:
        p = os.path.join(openface_dir, name.replace("/", os.sep))
        if os.path.isfile(p):
            exe = p
            break
    if exe is None:
        raise FileNotFoundError(
            "未找到 FeatureExtraction。请指定 --openface-dir（OpenFace 根目录或含 build/bin/ 的目录）"
        )

    cmd = [exe, "-f", video_path, "-out_dir", out_dir, "-2Dfp"]
    print("运行 OpenFace: {}".format(" ".join(cmd)))
    # 部分 OpenFace 构建需从可执行文件所在目录运行以加载模型
    cwd = os.path.dirname(exe)
    ret = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError("OpenFace 运行失败: {}".format(ret.stderr or ret.stdout))

    # 查找输出 CSV（OpenFace 命名: <video_basename>.csv 或 <video_basename>_0.csv）
    base = os.path.splitext(os.path.basename(video_path))[0]
    candidates = glob.glob(os.path.join(out_dir, base + "*.csv")) + \
                 glob.glob(os.path.join(out_dir, base + "_*.csv"))
    if not candidates:
        raise FileNotFoundError("OpenFace 未生成 CSV: {}/{}*.csv".format(out_dir, base))
    return candidates[0]


def center_crop(img, size=128):
    """
    中心裁剪图像到指定尺寸（fallback 方法，人脸检测失败时使用）

    【检查点 1.1】与训练预处理一致:
      - 颜色: BGR2RGB（训练 H5 存储为 RGB）
      - 尺寸: 128x128
      - 注意: 训练用 OpenFace 从中心点裁剪，此处用画面中心，裁剪区域不同
    
    Args:
        img: 输入图像 (BGR)
        size: 目标尺寸（正方形）
    
    Returns:
        裁剪后的图像 (size x size) RGB
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        img = cv2.resize(img, (max(size, w), max(size, h)))
        h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    cropped = img[y0:y0+size, x0:x0+size]
    # 【检查点 1.1】BGR2RGB，与训练 H5 存储格式一致
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return cropped


def detect_and_crop_face(img, face_cascade, size=128):
    """
    人脸检测并裁剪（完全匹配 preprocess_ubfc.py 的 bbox 计算和裁剪方式）

    【检查点 1.2】与训练预处理一致:
      - bbox_size: 1.5*(maxy-miny) 与 preprocess_ubfc 一致
      - y 扩展: 0.2*(maxy-miny) 与 preprocess_ubfc 一致
      - 裁剪: np.take(..., mode='clip') 与 preprocess_ubfc 一致
      - 颜色: BGR2RGB 与 preprocess_ubfc 一致
      - 差异: 训练用 OpenFace landmarks，此处用 Haar Cascade（检测方法不同）
    
    Step 1: 灰度化图像
    Step 2: Haar Cascade 人脸检测
    Step 3: 选择最大的人脸区域
    Step 4: 转换为类似 OpenFace landmarks 的方式计算 bbox
    Step 5: 计算 1.5倍人脸高度的正方形 bbox
    Step 6: 向上扩展20%（y_range_ext）
    Step 7: 使用 np.take 从中心点裁剪正方形区域（带 clip 模式）
    Step 8: 只有当 store_size != bbox_size 时才 resize
    Step 9: 转换为RGB
    
    Args:
        img: 输入图像 (BGR)
        face_cascade: OpenCV 人脸检测器
        size: 目标尺寸（store_size，默认128）
    
    Returns:
        face: 裁剪后的人脸 (size x size)
        box: 人脸框坐标 (x, y, w, h)，失败时返回 None
    """
    # Step 1: 转灰度图（Haar 需要灰度图）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: 检测人脸
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None, None
    
    # Step 3: 选择最大人脸（按面积）
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    
    # Step 4-6: 转换为类似 OpenFace landmarks 的方式计算 bbox
    # 从检测框得到边界
    minx = float(x)
    maxx = float(x + w)
    miny = float(y)
    maxy = float(y + h)
    
    # 【与原本预处理一致】向上扩展20%
    y_range_ext = (maxy - miny) * 0.2
    miny = miny - y_range_ext
    
    # 【与原本预处理一致】计算中心点
    cnt_x = np.round((minx + maxx) / 2).astype('int')
    cnt_y = np.round((maxy + miny) / 2).astype('int')
    
    # 【与原本预处理一致】计算 bbox_size = 1.5倍人脸高度（正方形）
    bbox_size = np.round(1.5 * (maxy - miny)).astype('int')
    
    # Step 7: 【检查点 1.2】BGR2RGB，与 preprocess_ubfc 一致（裁剪前转换）
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Step 8: 使用 np.take 从中心点裁剪正方形区域（带 clip 模式，与原本预处理一致）
    bbox_half_size = int(bbox_size / 2)
    face = np.take(frame_rgb, 
                   range(cnt_y - bbox_half_size, cnt_y - bbox_half_size + bbox_size), 
                   0, mode='clip')
    face = np.take(face, 
                   range(cnt_x - bbox_half_size, cnt_x - bbox_half_size + bbox_size), 
                   1, mode='clip')
    
    # Step 9: 只有当 store_size != bbox_size 时才 resize（与原本预处理一致）
    if size != bbox_size:
        face = cv2.resize(face, (size, size))
    
    return face, (x, y, w, h)


def load_frames_from_video_with_openface(video_path, landmark_path, store_size=128):
    """
    从视频文件 + OpenFace landmarks 读取并裁剪人脸帧（与 preprocess_ubfc 完全一致）

    【Record-then-OpenFace 流程】:
      1. webcam 录制: python live_predict_webcam.py --source 0 --record-only --save-video out.mp4 --duration 60
      2. OpenFace: ./FeatureExtraction -f out.mp4 -out_dir landmarks -2Dfp
      3. 本函数: 读取 out.mp4 + landmarks/*.csv，输出与训练一致的 128x128 人脸帧

    Returns:
        frames: [T, 128, 128, 3] uint8 RGB
        timestamps: 每帧时间戳（用于计算实际 fps）
    """
    landmark = pd.read_csv(landmark_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    total_num_frame = len(landmark)
    success_col = 'success' if 'success' in landmark.columns else ' success'
    x_prefix = 'x_' if 'x_0' in landmark.columns else ' x_'
    y_prefix = 'y_' if 'y_0' in landmark.columns else ' y_'

    bbox_size = None
    for frame_num in range(total_num_frame):
        if landmark[success_col][frame_num]:
            lm_x = [landmark[f'{x_prefix}{i}'][frame_num] for i in range(68)]
            lm_y = [landmark[f'{y_prefix}{i}'][frame_num] for i in range(68)]
            minx, maxx = np.min(lm_x), np.max(lm_x)
            miny, maxy = np.min(lm_y), np.max(lm_y)
            y_range_ext = (maxy - miny) * 0.2
            miny = miny - y_range_ext
            bbox_size = int(np.round(1.5 * (maxy - miny)))
            break
    if bbox_size is None:
        raise ValueError("landmark 中无有效帧")

    frames = []
    timestamps = []
    lm_x_prev = lm_y_prev = None
    cnt_x = cnt_y = None
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for frame_num in range(total_num_frame):
        if landmark[success_col][frame_num]:
            lm_x_ = np.array([landmark[f'{x_prefix}{i}'][frame_num] for i in range(68)])
            lm_y_ = np.array([landmark[f'{y_prefix}{i}'][frame_num] for i in range(68)])
            lm_x = 0.9 * lm_x_prev + 0.1 * lm_x_ if lm_x_prev is not None else lm_x_
            lm_y = 0.9 * lm_y_prev + 0.1 * lm_y_ if lm_y_prev is not None else lm_y_
            lm_x_prev, lm_y_prev = lm_x, lm_y
            minx, maxx = np.min(lm_x), np.max(lm_x)
            miny, maxy = np.min(lm_y), np.max(lm_y)
            y_range_ext = (maxy - miny) * 0.2
            miny = miny - y_range_ext
            cnt_x = int(np.round((minx + maxx) / 2))
            cnt_y = int(np.round((maxy + miny) / 2))
        if cnt_x is None:
            ret, _ = cap.read()
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox_half = int(bbox_size / 2)
        face = np.take(frame, range(cnt_y - bbox_half, cnt_y - bbox_half + bbox_size), 0, mode='clip')
        face = np.take(face, range(cnt_x - bbox_half, cnt_x - bbox_half + bbox_size), 1, mode='clip')
        if store_size != bbox_size:
            face = cv2.resize(face, (store_size, store_size))
        frames.append(face)
        timestamps.append(frame_num / video_fps)
    cap.release()
    return np.array(frames), np.array(timestamps)


def capture_frames(source, duration, fps, use_face, show, save_video):
    """
    采集视频帧并预处理

    【检查点 1.3】帧率与训练一致性:
      - 训练: H5 数据固定 30fps
      - 此处: 通过 time.sleep 控制目标 fps，实际 fps 依赖摄像头/文件
      - 若实际 fps != 30，会导致时间尺度错误，影响 rPPG 频率估计
      - 不做重采样: 与 preprocess_ubfc 一致（训练时 H5 已预处理到 30fps）
    
    Step 1: 打开视频源（摄像头或文件）
    Step 2: 按目标帧率采集帧
    Step 3: 每帧预处理（人脸检测/中心裁剪 -> 128x128）
    Step 4: 实时显示采集进度
    
    Args:
        source: 视频源（摄像头索引或文件路径）
        duration: 采集时长（秒）
        fps: 目标帧率
        use_face: 是否启用人脸检测
        show: 是否显示实时窗口
        save_video: 可选，保存视频路径
    
    Returns:
        frames: numpy array [T, H, W, C]，预处理后的帧序列
        timestamps: 每帧的时间戳
    """
    # Step 1: 打开视频源
    # 【检查点 1.3a】训练用 H5 固定 30fps；此处摄像头/文件实际 fps 可能 != 30
    cap = None
    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            for fallback in [0, 1, 2]:
                if fallback == idx:
                    continue
                cap = cv2.VideoCapture(fallback)
                if cap.isOpened():
                    print("  摄像头 {} 不可用，已切换至索引 {}".format(idx, fallback))
                    break
        if not cap.isOpened():
            raise RuntimeError(
                "无法打开摄像头。可尝试: 1) sudo usermod -aG video $USER 后重新登录  2) --camera-index 1 或 2  "
                "3) 使用已有视频: --source 视频路径"
            )
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频源: {}".format(source))

    # 【UBFC 对齐】摄像头模式下设定分辨率和帧率，与 UBFC-rPPG 一致 (Logitech C920: 640x480, 30fps)
    if source.isdigit():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  摄像头设置: {actual_w}x{actual_h} @ {actual_fps:.1f} fps (目标: 640x480 @ 30)")

    face_cascade = None
    if use_face:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    frames = []
    timestamps = []
    writer = None
    start = time.time()
    frame_interval = 1.0 / fps
    next_t = start
    last_print_time = start

    print(f"开始采集视频 ({duration}秒)...")
    print("提示: 按 'q' 键可提前退出")

    while True:
        now = time.time()
        elapsed = now - start
        remaining = duration - elapsed
        
        if elapsed >= duration:
            break
        if now < next_t:
            time.sleep(max(0.0, next_t - now))
        ret, frame = cap.read()
        if not ret:
            print("警告: 无法读取摄像头帧")
            break
        next_t += frame_interval

        # 预处理：人脸裁剪或中心裁剪 -> 128x128
        # 【检查点 1.2/1.1】训练用 OpenFace；此处 Haar 失败时 fallback 到 center_crop
        if use_face and face_cascade is not None:
            face, box = detect_and_crop_face(frame, face_cascade, size=128)
            if face is None:
                box = None
                face = center_crop(frame, size=128)
        else:
            box = None
            face = center_crop(frame, size=128)

        # 实时显示与保存
        if show:
            display = frame.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 在画面上显示进度信息
            progress_text = f"采集中: {elapsed:.1f}s / {duration:.1f}s (剩余 {remaining:.1f}s)"
            cv2.putText(display, progress_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "Press 'q' to quit", (10, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Webcam rPPG Capture", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户提前退出")
                break

        # 每5秒打印一次进度
        if now - last_print_time >= 5.0:
            print(f"  进度: {elapsed:.1f}s / {duration:.1f}s (已采集 {len(frames)} 帧)")
            last_print_time = now

        if save_video:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))
            writer.write(frame)

        frames.append(face)
        timestamps.append(time.time())

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        # Mac上需要等待一下才能关闭窗口
        cv2.waitKey(100)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # 确保窗口关闭
    
    print(f"采集完成: 共 {len(frames)} 帧")
    return np.array(frames), np.array(timestamps)


def run_model(model, frames, max_frames=300):
    """
    运行模型推理，从视频帧提取 rPPG 信号。

    【与 test.py 完全对齐】使用 utils_inference.dl_model，与 test.py 推理逻辑一致。
    输入格式与 H5 imgs 一致: [T, 128, 128, 3] RGB, 0-255。

    Args:
        model: 训练好的 PhysNet 模型
        frames: 输入帧 [T, H, W, C]
        max_frames: 最大输入帧数（默认 300，与训练 T=10s@30fps 一致；test.py 用 30s）

    Returns:
        rppg: rPPG 信号 [T]
    """
    import threading
    import sys

    if len(frames) > max_frames:
        print(f"  警告: 输入帧数({len(frames)})超过模型最大支持({max_frames})，将使用最后{max_frames}帧")
        frames = frames[-max_frames:]
    print(f"  实际输入帧数: {len(frames)}")
    sys.stdout.flush()

    device = next(model.parameters()).device
    print(f"  使用设备: {device}")
    sys.stdout.flush()

    progress_stop = threading.Event()
    def show_progress():
        dots = 0
        while not progress_stop.is_set():
            print(f"\r  正在推理" + "." * (dots % 4) + " " * (3 - dots % 4), end="", flush=True)
            dots += 1
            time.sleep(0.5)

    progress_thread = threading.Thread(target=show_progress, daemon=True)
    progress_thread.start()
    start_time = time.time()
    try:
        rppg = dl_model(model, frames, device)
    finally:
        progress_stop.set()
        progress_thread.join(timeout=1.0)
        elapsed = time.time() - start_time
        print(f"\r  推理完成 (耗时 {elapsed:.2f}秒)")
        sys.stdout.flush()

    print(f"  rPPG信号长度: {len(rppg)}")
    sys.stdout.flush()
    return rppg


def analyze_peaks(rppg_filtered, fs, top_n=5):
    """
    频域峰值分析：找出频谱中功率最强的多个峰值
    
    Step 1: 加窗（Hann 窗）
    Step 2: FFT 变换
    Step 3: 频率掩码（只保留 [0.6, 4] Hz = [36, 240] BPM 范围）
    Step 4: 找峰值
    Step 5: 按功率排序，返回 Top-N
    
    Args:
        rppg_filtered: 滤波后的 rPPG 信号
        fs: 采样率
        top_n: 返回前 N 个峰值
    
    Returns:
        peaks: list of (hr_bpm, power)，按功率降序
    """
    sig = rppg_filtered.reshape(-1)
    N = sig.shape[0]
    
    # Step 1: 加 Hann 窗（减少频谱泄露）
    sig_windowed = sig * signal.windows.hann(N)
    
    # Step 2: FFT 变换
    sig_f = np.abs(fft(sig_windowed))
    
    # Step 3: 频率掩码 [0.6, 4] Hz = [36, 240] BPM
    low_idx = int(np.round(0.6 / fs * N))
    high_idx = int(np.round(4 / fs * N))
    
    sig_f_masked = sig_f.copy()
    sig_f_masked[:low_idx] = 0
    sig_f_masked[high_idx:] = 0
    
    # Step 4: 找峰值
    peak_idx, _ = signal.find_peaks(sig_f_masked)
    if len(peak_idx) == 0:
        return []
    
    # Step 5: 转换为 BPM 并按功率排序
    peak_powers = sig_f_masked[peak_idx]
    peak_hrs = peak_idx / N * fs * 60
    
    sort_idx = np.argsort(peak_powers)[::-1]
    peaks = [(peak_hrs[i], peak_powers[i]) for i in sort_idx[:top_n]]
    
    return peaks


def compute_fft_peaks(rppg_filtered, fs):
    """
    频谱峰值计算 + 抛物线插值，提高频率估计精度
    返回 peaks 列表: [{hr_bpm, power, idx, idx_refined}]
    """
    sig = rppg_filtered.reshape(-1)
    N = sig.shape[0]
    sig_windowed = sig * signal.windows.hann(N)
    sig_f = np.abs(fft(sig_windowed))

    low_idx = int(np.round(0.6 / fs * N))
    high_idx = int(np.round(4 / fs * N))
    sig_f_masked = sig_f.copy()
    sig_f_masked[:low_idx] = 0
    sig_f_masked[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f_masked)
    if len(peak_idx) == 0:
        return []

    peaks = []
    for idx in peak_idx:
        power = sig_f_masked[idx]
        idx_refined = float(idx)
        if 1 <= idx < (N - 1):
            y0 = sig_f_masked[idx - 1]
            y1 = sig_f_masked[idx]
            y2 = sig_f_masked[idx + 1]
            denom = (y0 - 2 * y1 + y2)
            if denom != 0:
                delta = 0.5 * (y0 - y2) / denom
                idx_refined = idx + delta
        hr_bpm = (idx_refined / N) * fs * 60
        peaks.append({
            "hr_bpm": hr_bpm,
            "power": power,
            "idx": idx,
            "idx_refined": idx_refined
        })

    peaks.sort(key=lambda p: p["power"], reverse=True)
    return peaks


def select_hr_from_peaks(peaks, fs, n_samples):
    """
    基于频率分辨率与谐波关系的峰值选择（不硬编码固定心率阈值）
    """
    if not peaks:
        return None, {"reason": "no_peaks"}

    resolution_bpm = fs / n_samples * 60
    tol_bpm = 1.5 * resolution_bpm

    for p in peaks:
        p["harm_penalty"] = 0.0

    for hi in peaks:
        for lo in peaks:
            if lo["hr_bpm"] <= 0:
                continue
            ratio = hi["hr_bpm"] / lo["hr_bpm"]
            if abs(ratio - 2.0) < (tol_bpm / max(lo["hr_bpm"], 1e-6)) or \
               abs(ratio - 3.0) < (tol_bpm / max(lo["hr_bpm"], 1e-6)):
                hi["harm_penalty"] += lo["power"] / (hi["power"] + 1e-9)

    for p in peaks:
        # 功率 + 谐波惩罚的加权评分
        p["score"] = p["power"] / (1.0 + p["harm_penalty"])

    peaks_sorted = sorted(peaks, key=lambda p: p["score"], reverse=True)
    best = peaks_sorted[0]
    return best["hr_bpm"], {
        "reason": "peak_score",
        "resolution_bpm": resolution_bpm,
        "selected_hr": best["hr_bpm"],
        "selected_power": best["power"],
        "selected_score": best["score"]
    }


def adaptive_peak_counting(rppg_filtered, fs, hr_ref=None):
    """
    自适应 Peak Counting：
    - 最小峰距基于参考心率（来自 FFT 选峰）
    - prominence 采用稳健统计（MAD）
    """
    sig = rppg_filtered.reshape(-1)
    if hr_ref is None:
        hr_ref = 75.0
    hr_ref = float(np.clip(hr_ref, 36.0, 240.0))

    min_interval = 0.5 * 60.0 / hr_ref
    min_distance = max(1, int(min_interval * fs))

    med = np.median(sig)
    mad = np.median(np.abs(sig - med)) + 1e-9
    std = np.std(sig) + 1e-9
    prominence = max(0.75 * std, 2.0 * mad)

    peak_locations, _ = signal.find_peaks(sig, distance=min_distance, prominence=prominence)
    return peak_locations, {"min_distance": min_distance, "prominence": prominence}


def check_signal_quality(rppg, rppg_filtered, fs):
    """
    检查rPPG信号质量
    
    Returns:
        dict: 包含信号质量指标
    """
    quality = {}
    
    # 基本统计
    quality['mean'] = np.mean(rppg_filtered)
    quality['std'] = np.std(rppg_filtered)
    quality['min'] = np.min(rppg_filtered)
    quality['max'] = np.max(rppg_filtered)
    quality['range'] = quality['max'] - quality['min']
    
    # 信噪比估计（使用峰值功率/平均功率作为近似）
    sig_windowed = rppg_filtered * signal.windows.hann(len(rppg_filtered))
    sig_f = np.abs(fft(sig_windowed))
    low_idx = int(np.round(0.6 / fs * len(rppg_filtered)))
    high_idx = int(np.round(4 / fs * len(rppg_filtered)))
    sig_f_masked = sig_f.copy()
    sig_f_masked[:low_idx] = 0
    sig_f_masked[high_idx:] = 0
    
    peak_power = np.max(sig_f_masked)
    avg_power = np.mean(sig_f_masked[low_idx:high_idx])
    quality['snr_approx'] = peak_power / avg_power if avg_power > 0 else 0
    
    # 信号稳定性（变异系数）
    quality['cv'] = quality['std'] / abs(quality['mean']) if quality['mean'] != 0 else float('inf')
    
    return quality


def plot_waveform(time_axis, rppg_raw, rppg_filtered, peak_locs, 
                  hr_fft, output_path="rppg_waveform.png"):
    """
    生成 rPPG 波形可视化图
    
    包含：
    - 子图1: 原始信号
    - 子图2: 滤波信号 + 检测到的峰值标记
    - 子图3: FFT 频谱
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 子图1: 原始 rPPG 信号
    axes[0].plot(time_axis, rppg_raw, 'b-', linewidth=0.8, alpha=0.7)
    axes[0].set_title('rPPG Raw Signal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([time_axis[0], time_axis[-1]])
    
    # 子图2: 滤波后的 rPPG 信号 + 峰值标记
    axes[1].plot(time_axis, rppg_filtered, 'g-', linewidth=1.2, label='Filtered Signal')
    if len(peak_locs) > 0:
        axes[1].plot(time_axis[peak_locs], rppg_filtered[peak_locs], 
                     'ro', markersize=6, label=f'Peaks (n={len(peak_locs)})')
    axes[1].set_title(f'Filtered rPPG Signal (0.6-4 Hz) | HR: {hr_fft:.1f} BPM', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([time_axis[0], time_axis[-1]])
    
    # 子图3: FFT 频谱
    N = len(rppg_filtered)
    fs = len(rppg_filtered) / (time_axis[-1] - time_axis[0])
    sig_windowed = rppg_filtered * signal.windows.hann(N)
    sig_f = np.abs(fft(sig_windowed))
    freqs = np.fft.fftfreq(N, 1/fs)
    
    # 只显示正频率部分，且限制在 [0.6, 4] Hz = [36, 240] BPM
    positive_freqs = freqs[:N//2]
    positive_power = sig_f[:N//2]
    hr_range_mask = (positive_freqs >= 0.6) & (positive_freqs <= 4.0)
    
    axes[2].plot(positive_freqs[hr_range_mask] * 60, positive_power[hr_range_mask], 
                 'purple', linewidth=1.5)
    axes[2].axvline(hr_fft, color='red', linestyle='--', linewidth=2, 
                    label=f'FFT HR: {hr_fft:.1f} BPM')
    axes[2].set_title('FFT Spectrum (Heart Rate Range)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Heart Rate (BPM)')
    axes[2].set_ylabel('Power')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([36, 240])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def estimate_hr(rppg, fs, use_harmonics_removal, use_hr_fft, duration=10):
    """
    心率估计主函数（FFT 方法）

    【注意】此函数与训练无关，仅用于从 rPPG 信号估计心率。
    若 duration/fs 与训练不一致（如实际 fps != 30），会影响心率估计精度。
    
    Step 1: 带通滤波（0.6-4 Hz，保留心率相关频段）
    Step 2: FFT 方法估计心率（频域取主峰）
    Step 3: 频域峰值分析（Top-5）
    Step 4: 信号质量检查（SNR、CV 等）
    峰值位置用于波形图可视化标记。
    
    Args:
        rppg: 原始 rPPG 信号
        fs: 采样率
        use_harmonics_removal: 是否启用谐波去除
        use_hr_fft: 使用原始 hr_fft（否则用 parabolic）
        duration: 信号时长（秒）
    
    Returns:
        hr: FFT 方法的心率（BPM）
        peaks: Top-5 峰值列表
        quality: 信号质量指标
        rppg_filtered: 滤波后的信号
        peak_locs: 峰值位置（用于波形图标记）
    """
    # Step 1: 带通滤波（保留 0.6-4 Hz = 36-240 BPM）
    rppg_filtered = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fs)
    
    # Step 2: FFT 方法估计心率（原始 + 智能选峰）
    if use_hr_fft:
        hr_fft_raw, _, _ = hr_fft(rppg_filtered, fs=fs, harmonics_removal=use_harmonics_removal)
    else:
        hr_fft_raw, _, _ = hr_fft_parabolic(rppg_filtered, fs=fs, harmonics_removal=use_harmonics_removal)

    peaks_struct = compute_fft_peaks(rppg_filtered, fs)
    hr_fft_selected, selection_info = select_hr_from_peaks(peaks_struct, fs, len(rppg_filtered))
    if hr_fft_selected is None:
        hr_fft_selected = hr_fft_raw

    # 峰值位置（用于波形图标记）
    peak_locs, _ = adaptive_peak_counting(rppg_filtered, fs, hr_ref=hr_fft_selected)

    # Step 3: 频域峰值分析（用于可视化）
    peaks = analyze_peaks(rppg_filtered, fs, top_n=5)
    
    # Step 4: 信号质量检查
    quality = check_signal_quality(rppg, rppg_filtered, fs)
    quality["hr_fft_raw"] = float(hr_fft_raw)
    quality["hr_fft_selected"] = float(hr_fft_selected)
    quality["selection_info"] = selection_info

    return hr_fft_selected, peaks, quality, rppg_filtered, peak_locs


def load_gt_hr_from_bvp_csv(video_path, fs, use_harmonics, use_hr_fft, n_frames=300):
    """
    当视频来自 RPPG_data_benny_eric/subject/v01/ 时，从同目录的 BVP.csv 和 frames_timestamp.csv
    加载 GT，对齐最后 n_frames 帧对应时间戳的 BVP，滤波后估计 HR。
    Returns: (hr_gt, bvp_aligned) 或 (None, None) 若 BVP 不存在
    """
    if not video_path or not os.path.isfile(video_path):
        return None, None
    dir_path = os.path.dirname(os.path.abspath(video_path))
    bvp_path = os.path.join(dir_path, "BVP.csv")
    ts_path = os.path.join(dir_path, "frames_timestamp.csv")
    if not os.path.isfile(bvp_path) or not os.path.isfile(ts_path):
        return None, None

    bvp_df = pd.read_csv(bvp_path)
    ts_df = pd.read_csv(ts_path)
    t_bvp = bvp_df.iloc[:, 0].values.astype(float)
    bvp_raw = bvp_df.iloc[:, 1].values.astype(float)
    frame_col = ts_df.columns[0]
    ts_col = ts_df.columns[1]
    frame_indices = ts_df[frame_col].values
    timestamps = ts_df[ts_col].values.astype(float)

    # 最后 n_frames 帧的时间戳
    if len(frame_indices) < n_frames:
        return None, None
    last_n_ts = timestamps[-n_frames:]
    # 插值 BVP
    interp_fn = interp1d(t_bvp, bvp_raw, kind="linear", bounds_error=False, fill_value="extrapolate")
    bvp_at_frames = interp_fn(last_n_ts).astype(np.float32)

    hr_gt, _, _, _, _ = estimate_hr(
        bvp_at_frames, fs, use_harmonics, use_hr_fft, duration=n_frames / fs
    )
    return hr_gt, bvp_at_frames


def _default_live_output_dir(train_exp_dir):
    """results/live_runs/label_ratio_X/YYYY-MM-DD_HH-MM-SS"""
    subdir = get_live_runs_subdir(train_exp_dir)
    return os.path.join("results", "live_runs", subdir, time.strftime("%Y-%m-%d_%H-%M-%S"))


def main():
    args = parse_args()

    # 提前解析 train_exp_dir，供 output_dir 推断 label_ratio
    if args.train_exp_dir is None and not args.record_only:
        args.train_exp_dir = find_latest_run(0) or find_latest_run_any()
        if args.train_exp_dir and not args.live_ubfc:
            print("使用训练目录: {}".format(args.train_exp_dir))

    # 解析 openface_dir（参数默认已设为项目根/OpenFace）
    openface_dir = args.openface_dir

    # 【--live-ubfc 全流程】录制 -> OpenFace -> 推理 -> 保存 viz（与 test 保持一致）
    if args.live_ubfc:
        if not openface_dir or not os.path.isdir(openface_dir):
            raise ValueError(
                "--live-ubfc 需要指定有效的 --openface-dir。默认: 项目根/OpenFace"
            )
        output_dir = args.output_dir or _default_live_output_dir(args.train_exp_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 若 --source 为已有视频文件，跳过录制
        if not args.source.isdigit() and os.path.isfile(args.source):
            video_path = os.path.abspath(args.source)
            print("=" * 60)
            print("【live-ubfc】使用已有视频 -> OpenFace -> 推理 -> viz")
            print("=" * 60)
            print("  视频: {}".format(video_path))
        else:
            video_path = os.path.join(output_dir, "recorded_for_openface.mp4")
            print("=" * 60)
            print("【live-ubfc】全流程：录制 -> OpenFace -> 推理 -> viz")
            print("=" * 60)
            show = not args.no_show
            cam_src = str(args.camera_index)
            capture_frames(cam_src, args.duration, args.fps, False, show, video_path)
            print("\n✓ 录制完成: {}".format(video_path))

        landmarks_dir = os.path.join(output_dir, "landmarks")
        landmark_path = run_openface(video_path, landmarks_dir, openface_dir)
        print("✓ OpenFace 完成: {}".format(landmark_path))

        # 覆盖为视频+landmarks 流程，继续推理
        args.source = video_path
        args.landmarks = landmark_path
        args.save_waveform = True  # 全流程默认保存 viz
        args.full_waveform = True  # 完整时长波形，方便 viz 展示
        args.record_only = False
        args.output_dir = output_dir
        # 不 return，继续执行下方推理逻辑

    # 【Record-only 模式】仅录制，不运行模型
    if args.record_only:
        if args.save_video:
            save_path = args.save_video
        else:
            rec_dir = args.output_dir or _default_live_output_dir(args.train_exp_dir)
            os.makedirs(rec_dir, exist_ok=True)
            save_path = os.path.join(rec_dir, "recorded_for_openface.mp4")
        print("=" * 60)
        print("【Record-only】录制视频用于后续 OpenFace 处理")
        print("=" * 60)
        show = not args.no_show
        frames, ts = capture_frames(args.source, args.duration, args.fps, args.face, show, save_path)
        print("\n✓ 录制完成: {}".format(save_path))
        print("\n下一步:")
        print("  1. 运行 OpenFace: ./FeatureExtraction -f {} -out_dir <landmark_dir> -2Dfp".format(save_path))
        print("  2. 再运行本脚本: python live_predict_webcam.py --source {} --landmarks <landmark_dir>/<video_basename>.csv".format(save_path))
        return

    if args.train_exp_dir is None:
        args.train_exp_dir = find_latest_run(0) or find_latest_run_any()
        if not args.train_exp_dir:
            raise FileNotFoundError("未找到训练实验目录，请指定 --train-exp-dir")
        print("使用训练目录: {}".format(args.train_exp_dir))

    # 读取训练配置
    config_path = os.path.join(args.train_exp_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到config.json: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    fs = config.get("fs", 30)
    in_ch = config.get("in_ch", 3)
    S = config.get("S", 2)

    if in_ch != 3:
        raise ValueError("当前脚本仅支持RGB输入（in_ch=3）")

    # 权重选择：默认 best_model.pt，--epoch 指定时用 epoch{N}.pt
    weight_path = resolve_weight_path(args.train_exp_dir, args.epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysNet(S, in_ch=in_ch).to(device).eval()
    model.load_state_dict(torch.load(weight_path, map_location=device))

    print(f"使用权重: {weight_path}")
    print(f"采集参数: duration={args.duration}s, fps={args.fps} (训练fs={fs})")
    # 结果输出目录：results/live_runs/label_ratio_X/YYYY-MM-DD_HH-MM-SS
    output_dir = args.output_dir or _default_live_output_dir(args.train_exp_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("结果保存目录: {}".format(output_dir))
    print("=" * 60)

    show = not args.no_show
    # 【视频 + OpenFace landmarks】与训练 preprocessing 完全一致
    if not args.source.isdigit() and os.path.isfile(args.source):
        landmark_path = args.landmarks
        # 若未提供 landmarks 但指定了 openface_dir，自动运行 OpenFace
        if not landmark_path and openface_dir:
            landmarks_dir = os.path.join(output_dir, "landmarks")
            landmark_path = run_openface(args.source, landmarks_dir, openface_dir)
            print("✓ OpenFace 完成: {}".format(landmark_path))
        if landmark_path:
            print("使用 OpenFace landmarks 裁剪（与训练一致）")
            frames, ts = load_frames_from_video_with_openface(args.source, landmark_path, store_size=128)
            max_frames = int(args.duration * fs)
            if len(frames) > max_frames:
                frames = frames[:max_frames]
                ts = ts[:max_frames]
            print(f"从视频加载 {len(frames)} 帧 (OpenFace 裁剪)")
        else:
            frames, ts = capture_frames(args.source, args.duration, args.fps, args.face, show, None)
            print(f"从视频加载 {len(frames)} 帧 (Haar 裁剪，无 landmarks)")
    else:
        frames, ts = capture_frames(args.source, args.duration, args.fps, args.face, show, args.save_video)
    
    print("=" * 60)
    print("开始处理视频并计算心率...")
    if frames.shape[0] < fs * 10:
        raise RuntimeError("采集帧数过少，至少需要约10秒以上的帧数")

    # 【检查点 1.3b】帧率一致性
    # 训练: fs=30，H5 中已固定；此处实际 fps 可能 != 30
    # 若差异大，会导致时间尺度错误，进而影响 rPPG 频率估计
    actual_fps = len(frames) / (ts[-1] - ts[0]) if len(ts) > 1 else args.fps
    print(f"  实际采集帧率: {actual_fps:.2f} fps (目标: {args.fps:.1f} fps, 训练fs: {fs})")
    
    if abs(actual_fps - fs) > 1.0:
        print(f"  ⚠️  警告: 实际帧率({actual_fps:.2f})与训练fs({fs})差异较大")
        print(f"     原本GitHub repo方式：不做重采样，直接使用采集帧")
        print(f"     如果帧率差异大，可能导致时间尺度错误")
    
    # 注意：live streaming 保持不做重采样（与 process_video_dataset 一致）
    # 如果摄像头帧率不稳定，这是数据质量问题，不是代码问题

    print("  运行模型推理...")
    import sys
    sys.stdout.flush()  # 确保输出立即显示
    
    inference_start = time.time()
    # 【检查点 2.2】max_frames=300 与训练 T=fs*10 一致
    rppg = run_model(model, frames, max_frames=fs * 10)  # 最多300帧（10秒@30fps）
    sys.stdout.flush()
    
    # 可选：生成完整时长的rPPG波形（按10秒窗口拼接）
    rppg_full = None
    rppg_full_filtered = None
    peak_locs_full = None
    if args.full_waveform:
        # 完整波形：按 10 秒窗口滑动推理，与训练 T=300 一致
        window_len = fs * 10
        if len(frames) >= window_len:
            print("  生成完整波形：按10秒窗口逐段推理（带重叠拼接）...")
            stride = window_len // 2  # 5秒重叠，提升连续性
            starts = list(range(0, len(frames) - window_len + 1, stride))
            last_start = len(frames) - window_len
            if last_start not in starts:
                starts.append(last_start)

            rppg_accum = np.zeros(len(frames), dtype=np.float32)
            rppg_count = np.zeros(len(frames), dtype=np.float32)

            for start in starts:
                chunk = frames[start:start + window_len]
                rppg_chunk = run_model(model, chunk, max_frames=window_len)
                rppg_accum[start:start + window_len] += rppg_chunk
                rppg_count[start:start + window_len] += 1.0

            rppg_full = rppg_accum / np.maximum(rppg_count, 1e-6)
            rppg_full_filtered = butter_bandpass(rppg_full, lowcut=0.6, highcut=4, fs=fs)
            peak_locs_full, _ = adaptive_peak_counting(rppg_full_filtered, fs, hr_ref=None)
            print(f"  完整波形长度: {len(rppg_full)} 样本 ({len(rppg_full)/fs:.1f}s)")
        else:
            print("  ⚠️  帧数不足10秒，无法生成完整波形")

    print("  计算心率...")
    sys.stdout.flush()
    hr_start = time.time()
    # 有完整波形时用 60s 估计 HR，否则用 10s
    rppg_for_hr = rppg_full if rppg_full is not None else rppg
    rppg_for_hr_len = len(rppg_for_hr)
    hr, peaks, quality, rppg_filtered, peak_locs = estimate_hr(
        rppg_for_hr, fs=fs, use_harmonics_removal=args.harmonics, use_hr_fft=args.use_hr_fft, 
        duration=rppg_for_hr_len / fs)
    hr_elapsed = time.time() - hr_start
    print(f"  心率计算完成 (耗时 {hr_elapsed:.2f}秒) [基于 {rppg_for_hr_len/fs:.1f}s]")
    sys.stdout.flush()

    # 若视频来自 RPPG_data/subject/v01/，从 BVP.csv 计算 GT HR 并对比
    hr_gt = None
    if not args.source.isdigit() and os.path.isfile(args.source):
        n_used = min(len(frames), rppg_for_hr_len)
        hr_gt, _ = load_gt_hr_from_bvp_csv(args.source, fs, args.harmonics, args.use_hr_fft, n_frames=n_used)
    if hr_gt is not None:
        err_bpm = hr - hr_gt
        print("")
        print("=" * 60)
        print("【GT 对比】BVP.csv 真值")
        print("=" * 60)
        print(f"  HR (预测): {hr:.2f} BPM")
        print(f"  HR (GT):   {hr_gt:.2f} BPM")
        print(f"  误差:      {err_bpm:+.2f} BPM")
        print("=" * 60)

    # 保存波形（如果指定）到 output_dir
    rppg_raw_plot = rppg_for_hr  # 用于绘图的 raw（60s 或 10s）
    if args.save_waveform:
        waveform_data = {
            'rppg_raw': rppg,
            'rppg_filtered': rppg_filtered,
            'fs': fs,
            'duration': len(frames)/fs,
            'peak_locations': peak_locs
        }
        if rppg_full is not None:
            waveform_data['rppg_full_raw'] = rppg_full
            waveform_data['rppg_full_filtered'] = rppg_full_filtered
            waveform_data['peak_locations_full'] = peak_locs_full
        waveform_file = os.path.join(output_dir, 'rppg_waveform.npz')
        np.savez(waveform_file, **waveform_data)
        print(f"\n波形已保存到: {waveform_file}")

        time_axis = np.arange(len(rppg_filtered)) / fs
        txt_path = os.path.join(output_dir, 'rppg_waveform.txt')
        with open(txt_path, 'w') as f:
            f.write("时间(秒)\trPPG原始\trPPG滤波\n")
            for i in range(len(rppg_filtered)):
                t = i / fs
                f.write(f"{t:.3f}\t{rppg_raw_plot[i]:.6f}\t{rppg_filtered[i]:.6f}\n")
        print(f"波形文本已保存到: {txt_path}")

        png_path = os.path.join(output_dir, 'rppg_waveform.png')
        plot_waveform(time_axis, rppg_raw_plot, rppg_filtered, peak_locs,
                      hr_fft=hr, output_path=png_path)
        print(f"波形图已保存到: {png_path}")
        print("所有结果保存在: {}".format(output_dir))

    print("=" * 60)
    print(f"✅ 预测心率 (FFT): {hr:.2f} BPM")
    print(f"   原始FFT估计: {quality.get('hr_fft_raw', hr):.2f} BPM")
    print("=" * 60)

    # 保存本次运行摘要到 output_dir（每次运行都有记录）
    run_summary_path = os.path.join(output_dir, "run_summary.txt")
    with open(run_summary_path, "w", encoding="utf-8") as f:
        f.write("Live rPPG Run Summary\n")
        f.write("=" * 50 + "\n")
        f.write("Time: {}\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
        f.write("Source: {}\n".format(args.source))
        f.write("Duration: {:.1f}s, Frames: {}\n".format(len(frames)/fs, len(frames)))
        f.write("HR (FFT): {:.2f} BPM\n".format(hr))
        if hr_gt is not None:
            f.write("HR (GT from BVP.csv): {:.2f} BPM\n".format(hr_gt))
            f.write("Error (Pred - GT): {:.2f} BPM\n".format(hr - hr_gt))
        f.write("Weight: {}\n".format(weight_path))
    print("\n运行摘要已保存到: {}".format(run_summary_path))
    
    # ======================================================================
    # 诊断1: 信号质量检查
    # ======================================================================
    print("\n【诊断1】rPPG信号质量检查:")
    print("-" * 60)
    print(f"  信号均值: {quality['mean']:.4f}")
    print(f"  信号标准差: {quality['std']:.4f}")
    print(f"  信号范围: [{quality['min']:.4f}, {quality['max']:.4f}]")
    print(f"  变异系数 (CV): {quality['cv']:.4f} {'⚠️ 高变异' if quality['cv'] > 0.5 else '✓ 正常'}")
    print(f"  近似SNR: {quality['snr_approx']:.2f} {'⚠️ 低SNR' if quality['snr_approx'] < 3.0 else '✓ 正常'}")
    print("-" * 60)
    
    # ======================================================================
    # 诊断2: 峰值分析 + 备选峰值评估
    # ======================================================================
    if peaks:
        print("\n【诊断2】频谱峰值分析 (Top-5):")
        print("-" * 60)
        print(f"{'排名':<6} {'心率 (BPM)':<15} {'功率':<15} {'合理性':<15}")
        print("-" * 60)
        
        # 合理性检查：正常心率范围 [50, 150] BPM
        reasonable_range = (50, 150)
        
        for i, (peak_hr, peak_power) in enumerate(peaks, 1):
            marker = "✓" if i == 1 else " "
            is_reasonable = reasonable_range[0] <= peak_hr <= reasonable_range[1]
            reasonableness = "✓ 合理" if is_reasonable else "⚠️ 异常"
            print(f"{marker} {i:<5} {peak_hr:<15.2f} {peak_power:<15.2f} {reasonableness:<15}")
        print("-" * 60)
        
        # 检查峰值选择问题
        if len(peaks) >= 2:
            peak1_hr, peak1_power = peaks[0]
            peak2_hr, peak2_power = peaks[1]
            power_ratio = peak1_power / peak2_power if peak2_power > 0 else float('inf')
            
            print(f"\n【诊断3】峰值选择分析:")
            print("-" * 60)
            print(f"  当前选择: 峰值1 ({peak1_hr:.2f} BPM, 功率={peak1_power:.2f})")
            print(f"  功率比 (峰值1/峰值2): {power_ratio:.2f}")
            
            if power_ratio < 1.5:
                print(f"  ⚠️  警告: 峰值1和峰值2功率接近，可能选错峰值")
            
            # 检查是否有更合理的峰值
            reasonable_peaks = [(hr, pwr) for hr, pwr in peaks if reasonable_range[0] <= hr <= reasonable_range[1]]
            if reasonable_peaks and peak1_hr not in [p[0] for p in reasonable_peaks[:1]]:
                best_reasonable = reasonable_peaks[0]
                print(f"  💡 建议: 峰值{peaks.index(best_reasonable)+1} ({best_reasonable[0]:.2f} BPM) 在合理范围内且功率较高")
            
            # 显示备选峰值的影响
            print(f"\n  备选峰值评估:")
            for i, (peak_hr, peak_power) in enumerate(peaks[:3], 1):  # 只显示前3个
                if i == 1:
                    continue  # 跳过已选择的
                power_ratio_to_best = peak_power / peak1_power if peak1_power > 0 else 0
                is_reasonable = reasonable_range[0] <= peak_hr <= reasonable_range[1]
                print(f"    峰值{i}: {peak_hr:.2f} BPM (功率比={power_ratio_to_best:.2f}, {'合理' if is_reasonable else '异常'})")
            print("-" * 60)

        # 智能选峰信息
        selection_info = quality.get("selection_info", {})
        if selection_info:
            print("\n【诊断3b】智能选峰结果:")
            print("-" * 60)
            print(f"  选峰原因: {selection_info.get('reason', 'N/A')}")
            print(f"  频率分辨率: {selection_info.get('resolution_bpm', 0):.2f} BPM")
            print(f"  选中HR: {selection_info.get('selected_hr', 0):.2f} BPM")
            print("-" * 60)
    
    # ======================================================================
    # 诊断4: 可能的问题原因
    # ======================================================================
    print("\n【诊断4】可能的问题原因:")
    print("-" * 60)
    issues = []
    suggestions = []
    
    if quality['snr_approx'] < 3.0:
        issues.append("信号SNR较低")
        suggestions.append("改善光照条件、保持静止、确保人脸正对摄像头")
    
    if quality['cv'] > 0.5:
        issues.append("信号变异较大")
        suggestions.append("减少运动、确保稳定采集环境")
    
    if peaks and len(peaks) >= 2:
        peak1_hr, peak1_power = peaks[0]
        peak2_hr, peak2_power = peaks[1]
        power_ratio = peak1_power / peak2_power if peak2_power > 0 else float('inf')
        if power_ratio < 1.5:
            issues.append("多个峰值功率接近")
            suggestions.append("考虑使用更智能的峰值选择策略（结合合理性检查）")
    
    if not issues:
        print("  ✓ 未发现明显问题")
    else:
        for issue in issues:
            print(f"  ⚠️  {issue}")
        print("\n  改进建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"    {i}. {suggestion}")
    
    print("-" * 60)
    print("=" * 60)


if __name__ == "__main__":
    main()

