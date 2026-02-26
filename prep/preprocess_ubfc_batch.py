#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量预处理UBFC数据集：从原始视频和OpenFace landmarks生成h5文件

使用方法:
    python preprocess_ubfc_batch.py --input_dir datasets/UBFC_raw --landmarks_dir landmarks --output_dir datasets/UBFC_h5
"""

import argparse
import os
import sys
import cv2
import numpy as np
import h5py
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def load_landmarks(csv_path):
    """从OpenFace输出的CSV文件加载landmarks"""
    try:
        df = pd.read_csv(csv_path)
        # OpenFace的2D landmarks列名通常是 x_0, y_0, x_1, y_1, ..., x_67, y_67
        # 或者可能是其他格式，需要根据实际情况调整
        landmarks = []
        for i in range(68):  # 68个landmark点
            x_col = f'x_{i}'
            y_col = f'y_{i}'
            if x_col in df.columns and y_col in df.columns:
                landmarks.append(np.array([df[x_col].values, df[y_col].values]).T)
            else:
                # 尝试其他可能的列名格式
                x_col = f'X_{i}'
                y_col = f'Y_{i}'
                if x_col in df.columns and y_col in df.columns:
                    landmarks.append(np.array([df[x_col].values, df[y_col].values]).T)
                else:
                    raise ValueError(f"无法找到landmark列: x_{i}, y_{i}")
        
        if len(landmarks) == 0:
            raise ValueError("未找到任何landmark数据")
        
        # 合并所有landmarks
        landmarks = np.stack(landmarks, axis=1)  # [N_frames, 68, 2]
        return landmarks
    except Exception as e:
        print(f"⚠️  加载landmarks失败 {csv_path}: {e}")
        return None

def get_face_bbox(landmarks_frame, frame_idx, first_frame_bbox=None):
    """
    根据landmarks计算人脸边界框
    
    Args:
        landmarks_frame: 当前帧的landmarks [68, 2]
        frame_idx: 帧索引
        first_frame_bbox: 第一帧的边界框（用于固定大小）
    
    Returns:
        bbox: (cx, cy, w, h) 中心点和宽高
    """
    if landmarks_frame is None or len(landmarks_frame) == 0:
        return None
    
    # 计算landmarks的边界
    x_min = np.min(landmarks_frame[:, 0])
    x_max = np.max(landmarks_frame[:, 0])
    y_min = np.min(landmarks_frame[:, 1])
    y_max = np.max(landmarks_frame[:, 1])
    
    # 计算中心点和尺寸
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = x_max - x_min
    h = y_max - y_min
    
    # 第一帧：固定边界框大小（1.2倍垂直范围）
    if frame_idx == 0:
        bbox_size = h * 1.2
        first_frame_bbox = (cx, cy, bbox_size, bbox_size)
        return first_frame_bbox
    
    # 后续帧：使用第一帧的大小，但更新中心点
    if first_frame_bbox is not None:
        _, _, w_fixed, h_fixed = first_frame_bbox
        return (cx, cy, w_fixed, h_fixed)
    
    return (cx, cy, w * 1.2, h * 1.2)

def crop_face_frame(frame, bbox):
    """根据边界框裁剪人脸"""
    if bbox is None:
        return None
    
    cx, cy, w, h = bbox
    
    # 计算裁剪区域
    x1 = int(max(0, cx - w/2))
    y1 = int(max(0, cy - h/2))
    x2 = int(min(frame.shape[1], cx + w/2))
    y2 = int(min(frame.shape[0], cy + h/2))
    
    # 裁剪
    cropped = frame[y1:y2, x1:x2]
    
    # 调整大小到128x128
    if cropped.size > 0:
        cropped = cv2.resize(cropped, (128, 128))
        return cropped
    return None

def load_bvp(ppg_file):
    """加载PPG信号（BVP）"""
    # UBFC数据集的PPG文件格式可能是.txt或.csv
    # 需要根据实际格式调整
    try:
        if ppg_file.endswith('.txt'):
            bvp = np.loadtxt(ppg_file)
        elif ppg_file.endswith('.csv'):
            df = pd.read_csv(ppg_file)
            # 假设第一列是BVP信号
            bvp = df.iloc[:, 0].values
        else:
            raise ValueError(f"不支持的PPG文件格式: {ppg_file}")
        return bvp
    except Exception as e:
        print(f"⚠️  加载BVP失败 {ppg_file}: {e}")
        return None

def process_video(video_path, landmarks_path, bvp_path, output_path):
    """
    处理单个视频，生成h5文件
    
    Args:
        video_path: 视频文件路径
        landmarks_path: OpenFace landmarks CSV文件路径
        bvp_path: PPG信号文件路径
        output_path: 输出h5文件路径
    """
    print(f"\n处理: {video_path.name}")
    
    # 1. 加载landmarks
    landmarks = load_landmarks(landmarks_path)
    if landmarks is None:
        print(f"  ❌ 跳过：无法加载landmarks")
        return False
    
    # 2. 加载BVP信号
    bvp = load_bvp(bvp_path)
    if bvp is None:
        print(f"  ❌ 跳过：无法加载BVP信号")
        return False
    
    # 3. 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ 跳过：无法打开视频")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  视频信息: {total_frames} 帧, {fps:.1f} fps")
    print(f"  Landmarks: {landmarks.shape}")
    print(f"  BVP: {len(bvp)} 采样点")
    
    # 4. 处理每一帧
    frames = []
    first_frame_bbox = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 获取当前帧的landmarks
        if frame_idx < len(landmarks):
            landmarks_frame = landmarks[frame_idx]
            bbox = get_face_bbox(landmarks_frame, frame_idx, first_frame_bbox)
            if frame_idx == 0:
                first_frame_bbox = bbox
            
            # 裁剪人脸
            cropped = crop_face_frame(frame, bbox)
            if cropped is not None:
                frames.append(cropped)
        
        frame_idx += 1
    
    cap.release()
    
    if len(frames) == 0:
        print(f"  ❌ 跳过：未提取到有效帧")
        return False
    
    # 5. 对齐BVP和视频帧数
    frames = np.array(frames)
    min_length = min(len(frames), len(bvp))
    frames = frames[:min_length]
    bvp = bvp[:min_length]
    
    print(f"  最终数据: {len(frames)} 帧, {len(bvp)} BVP采样点")
    
    # 6. 保存为h5文件
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('imgs', data=frames, compression='gzip')
        f.create_dataset('bvp', data=bvp, compression='gzip')
    
    print(f"  ✅ 保存到: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='批量预处理UBFC数据集')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='原始视频目录')
    parser.add_argument('--landmarks_dir', type=str, required=True,
                        help='OpenFace landmarks目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出h5文件目录')
    parser.add_argument('--video_ext', type=str, default='.avi',
                        help='视频文件扩展名（默认: .avi）')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    landmarks_dir = Path(args.landmarks_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    if not landmarks_dir.exists():
        print(f"❌ Landmarks目录不存在: {landmarks_dir}")
        return
    
    # 查找所有视频文件
    video_files = list(input_dir.rglob(f"*{args.video_ext}"))
    
    if len(video_files) == 0:
        print(f"❌ 未找到视频文件（扩展名: {args.video_ext}）")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频
    success_count = 0
    for video_path in tqdm(video_files, desc="处理视频"):
        # 查找对应的landmarks文件
        # 假设landmarks文件名与视频文件名相同，只是扩展名不同
        landmarks_name = video_path.stem + '.csv'
        landmarks_path = landmarks_dir / landmarks_name
        
        if not landmarks_path.exists():
            # 尝试其他可能的路径
            landmarks_path = landmarks_dir / video_path.parent.name / landmarks_name
            if not landmarks_path.exists():
                print(f"⚠️  未找到landmarks: {landmarks_name}")
                continue
        
        # 查找BVP文件（可能需要根据实际数据集结构调整）
        bvp_name = video_path.stem + '.txt'
        bvp_path = video_path.parent / bvp_name
        if not bvp_path.exists():
            bvp_path = video_path.parent / 'bvp' / bvp_name
        if not bvp_path.exists():
            print(f"⚠️  未找到BVP文件: {bvp_name}")
            continue
        
        # 输出文件名
        output_name = video_path.stem + '.h5'
        output_path = output_dir / output_name
        
        # 处理视频
        if process_video(video_path, landmarks_path, bvp_path, output_path):
            success_count += 1
    
    print(f"\n处理完成: {success_count}/{len(video_files)} 成功")

if __name__ == "__main__":
    main()
