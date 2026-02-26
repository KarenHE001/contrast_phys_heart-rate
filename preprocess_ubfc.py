"""
UBFC-rPPG 数据集预处理脚本
将原始视频和PPG信号转换为训练所需的.h5格式

使用方法:
1. 首先使用OpenFace提取面部关键点:
   ./FeatureExtraction -f <video_file> -out_dir <landmark_dir> -2Dfp

2. 然后运行此脚本:
   python preprocess_ubfc.py --video_dir <video_dir> --landmark_dir <landmark_dir> --ppg_dir <ppg_dir> --output_dir <output_dir>
"""

import cv2
import numpy as np
import h5py
import pandas as pd
import os
import argparse
from scipy import interpolate
import glob


def load_ppg_signal_ubfc(ppg_path, target_length, original_fps=30, dataset_type=2):
    """
    加载UBFC-rPPG数据集的PPG信号并插值到目标长度
    
    Args:
        ppg_path: PPG信号文件路径
        target_length: 目标长度（视频帧数）
        original_fps: 视频帧率（默认30fps）
        dataset_type: 数据集类型，1=Dataset1, 2=Dataset2
    
    Returns:
        ppg_signal: 插值后的PPG信号，长度=target_length
    """
    try:
        if dataset_type == 1:
            # Dataset1: gtdump.xmp文件格式
            # Column 1: Timestep (ms)
            # Column 2: Heart rate (HR)
            # Column 3: SpO2
            # Column 4: PPG signal
            try:
                data = pd.read_csv(ppg_path, sep='\t', header=None)
                if data.shape[1] >= 4:
                    timestep_ms = data.iloc[:, 0].values  # 毫秒
                    ppg_data = data.iloc[:, 3].values  # PPG信号（第4列）
                    original_time = timestep_ms / 1000.0  # 转换为秒
                else:
                    raise ValueError(f"Unexpected format in {ppg_path}")
            except:
                # 尝试其他分隔符
                data = pd.read_csv(ppg_path, sep=' ', header=None)
                if data.shape[1] >= 4:
                    timestep_ms = data.iloc[:, 0].values
                    ppg_data = data.iloc[:, 3].values
                    original_time = timestep_ms / 1000.0
                else:
                    raise ValueError(f"Cannot parse Dataset1 format: {ppg_path}")
        
        elif dataset_type == 2:
            # Dataset2: ground_truth.txt文件格式
            # Line 1: PPG signal (space-separated values)
            # Line 2: Heart rate (HR)
            # Line 3: Timestep (seconds, scientific notation)
            with open(ppg_path, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 3:
                raise ValueError(f"Dataset2 format requires at least 3 lines, got {len(lines)}")
            
            # Line 1: PPG signal
            ppg_data = np.array([float(x) for x in lines[0].strip().split()])
            
            # Line 3: Timestep (seconds)
            timestep_str = lines[2].strip()
            # 处理科学计数法
            timestep_values = np.array([float(x) for x in timestep_str.split()])
            
            # 如果timestep只有一个值，假设是采样间隔
            if len(timestep_values) == 1:
                # 假设是采样间隔（秒）
                sampling_interval = timestep_values[0]
                original_time = np.arange(len(ppg_data)) * sampling_interval
            else:
                # 如果timestep是时间序列
                original_time = timestep_values[:len(ppg_data)]
        
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        if len(ppg_data) == 0:
            print(f"Warning: Empty PPG file {ppg_path}, creating dummy signal")
            return np.zeros(target_length, dtype=np.float32)
        
        # 创建目标时间点（视频帧率）
        target_time = np.arange(target_length) / original_fps
        
        # 确保目标时间不超过原始时间范围
        if len(original_time) > 0:
            max_time = original_time[-1]
            target_time = np.clip(target_time, 0, max_time)
        
        # 使用线性插值
        if len(ppg_data) > 1 and len(original_time) == len(ppg_data):
            f = interpolate.interp1d(original_time, ppg_data, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
            ppg_signal = f(target_time)
        elif len(ppg_data) > 1:
            # 如果时间信息不完整，假设均匀采样
            original_time = np.arange(len(ppg_data)) / (len(ppg_data) / original_time[-1] if len(original_time) > 0 else 1.0)
            f = interpolate.interp1d(original_time, ppg_data, kind='linear', 
                                     bounds_error=False, fill_value='extrapolate')
            ppg_signal = f(target_time)
        else:
            # 如果只有一个值，复制到所有帧
            ppg_signal = np.full(target_length, ppg_data[0])
        
        return ppg_signal.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading PPG signal from {ppg_path}: {e}")
        import traceback
        traceback.print_exc()
        print(f"Creating dummy PPG signal of length {target_length}")
        return np.zeros(target_length, dtype=np.float32)


def openface_h5_with_ppg(video_path, landmark_path, h5_path, ppg_path=None, 
                        store_size=128, video_fps=30, dataset_type=2):
    """
    从OpenFace landmarks裁剪人脸并保存为.h5文件，同时包含PPG信号
    
    Args:
        video_path: 视频文件路径
        landmark_path: OpenFace生成的landmark .csv文件路径
        h5_path: 输出的.h5文件路径
        ppg_path: PPG信号文件路径（可选，用于测试集）
        store_size: 裁剪后的人脸大小（默认128x128）
        video_fps: 视频帧率（默认30fps）
        ppg_fps: PPG信号采样率（默认4Hz）
    """
    
    # 读取landmark文件
    try:
        landmark = pd.read_csv(landmark_path)
    except Exception as e:
        print(f"Error reading landmark file {landmark_path}: {e}")
        return False
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return False
    
    total_num_frame = len(landmark)
    
    # 初始化bounding box（从第一帧）
    # 检查列名格式（OpenFace可能生成'success'或' success'）
    success_col = 'success' if 'success' in landmark.columns else ' success'
    x_prefix = 'x_' if 'x_0' in landmark.columns else ' x_'
    y_prefix = 'y_' if 'y_0' in landmark.columns else ' y_'
    
    bbox_size = None
    for frame_num in range(total_num_frame):
        if landmark[success_col][frame_num]:
            lm_x = []
            lm_y = []
            for lm_num in range(68):
                lm_x.append(landmark[f'{x_prefix}{lm_num}'][frame_num])
                lm_y.append(landmark[f'{y_prefix}{lm_num}'][frame_num])
            
            lm_x = np.array(lm_x)
            lm_y = np.array(lm_y)
            
            minx = np.min(lm_x)
            maxx = np.max(lm_x)
            miny = np.min(lm_y)
            maxy = np.max(lm_y)
            
            y_range_ext = (maxy-miny)*0.2
            miny = miny - y_range_ext
            
            cnt_x = np.round((minx+maxx)/2).astype('int')
            cnt_y = np.round((maxy+miny)/2).astype('int')
            
            bbox_size = np.round(1.5*(maxy-miny)).astype('int')
            break
    
    if bbox_size is None:
        print(f"Error: No valid landmarks found in {landmark_path}")
        cap.release()
        return False
    
    # 创建.h5文件
    with h5py.File(h5_path, 'w') as f:
        # 创建imgs数据集
        imgs = f.create_dataset('imgs', shape=(total_num_frame, store_size, store_size, 3), 
                               dtype='uint8', chunks=(1, store_size, store_size, 3),
                               compression="gzip", compression_opts=4)
        
        # 处理每一帧
        lm_x_prev = None
        lm_y_prev = None
        cnt_x = None
        cnt_y = None
        
        for frame_num in range(total_num_frame):
            if landmark[success_col][frame_num]:
                lm_x_ = []
                lm_y_ = []
                for lm_num in range(68):
                    lm_x_.append(landmark[f'{x_prefix}{lm_num}'][frame_num])
                    lm_y_.append(landmark[f'{y_prefix}{lm_num}'][frame_num])
                
                lm_x_ = np.array(lm_x_)
                lm_y_ = np.array(lm_y_)
                
                # 使用平滑（90%前一个+10%当前）
                if lm_x_prev is not None:
                    lm_x = 0.9*lm_x_prev + 0.1*lm_x_
                    lm_y = 0.9*lm_y_prev + 0.1*lm_y_
                else:
                    lm_x = lm_x_
                    lm_y = lm_y_
                
                lm_x_prev = lm_x
                lm_y_prev = lm_y
                
                minx = np.min(lm_x)
                maxx = np.max(lm_x)
                miny = np.min(lm_y)
                maxy = np.max(lm_y)
                
                y_range_ext = (maxy-miny)*0.2
                miny = miny - y_range_ext
                
                cnt_x = np.round((minx+maxx)/2).astype('int')
                cnt_y = np.round((maxy+miny)/2).astype('int')
            
            # 如果这一帧没有成功检测，使用上一帧的位置
            if cnt_x is None or cnt_y is None:
                print(f"Warning: Frame {frame_num} has no valid landmarks, skipping")
                # 读取但不处理这一帧
                ret, frame = cap.read()
                if not ret:
                    break
                continue
            
            # 读取视频帧
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Can't read frame {frame_num} from {video_path}")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 裁剪人脸
            bbox_half_size = int(bbox_size/2)
            face = np.take(frame, 
                          range(cnt_y-bbox_half_size, cnt_y-bbox_half_size+bbox_size), 
                          0, mode='clip')
            face = np.take(face, 
                          range(cnt_x-bbox_half_size, cnt_x-bbox_half_size+bbox_size), 
                          1, mode='clip')
            
            # 调整大小
            if store_size != bbox_size:
                face = cv2.resize(face, (store_size, store_size))
            
            imgs[frame_num] = face
        
        # 处理PPG信号（如果提供）
        if ppg_path and os.path.exists(ppg_path):
            ppg_signal = load_ppg_signal_ubfc(ppg_path, total_num_frame, video_fps, dataset_type)
            f.create_dataset('bvp', data=ppg_signal, compression="gzip")
            print(f"  Added PPG signal from {ppg_path} (Dataset{dataset_type} format)")
        else:
            # 如果没有PPG信号，创建占位数组（训练时可能不需要）
            dummy_bvp = np.zeros(total_num_frame, dtype=np.float32)
            f.create_dataset('bvp', data=dummy_bvp, compression="gzip")
            print(f"  Created dummy BVP signal (no PPG file provided)")
    
    cap.release()
    return True


def process_ubfc_dataset(video_dir, landmark_dir, output_dir, ppg_dir=None, 
                        video_pattern="*.avi", ppg_pattern="*.txt", dataset_type=2):
    """
    批量处理UBFC-rPPG数据集
    
    Args:
        video_dir: 视频文件目录
        landmark_dir: OpenFace landmark文件目录
        output_dir: 输出.h5文件目录
        ppg_dir: PPG信号文件目录（可选）
        video_pattern: 视频文件匹配模式
        ppg_pattern: PPG文件匹配模式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有视频文件
    video_files = glob.glob(os.path.join(video_dir, video_pattern))
    video_files.sort()
    
    print(f"Found {len(video_files)} video files")
    
    success_count = 0
    fail_count = 0
    
    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 从视频路径提取subject名称（例如：DATASET_2/subject1/vid_converted.avi -> subject1）
        # 如果视频名包含"converted"，尝试从路径提取subject名
        subject_name = None
        if 'converted' in video_name.lower():
            path_parts = os.path.normpath(video_path).split(os.sep)
            for part in path_parts:
                if part.startswith('subject'):
                    subject_name = part
                    break
        
        # 查找对应的landmark文件
        landmark_path = os.path.join(landmark_dir, f"{video_name}.csv")
        if not os.path.exists(landmark_path):
            # 尝试其他可能的命名方式
            landmark_files = glob.glob(os.path.join(landmark_dir, f"*{video_name}*.csv"))
            if landmark_files:
                landmark_path = landmark_files[0]
            elif subject_name:
                # 如果找到了subject名，尝试用subject名查找
                subject_csv = glob.glob(os.path.join(landmark_dir, f"{subject_name}*.csv"))
                if subject_csv:
                    landmark_path = subject_csv[0]
                else:
                    print(f"Warning: No landmark file found for {video_name} (subject: {subject_name}), skipping...")
                    fail_count += 1
                    continue
            else:
                print(f"Warning: No landmark file found for {video_name}, skipping...")
                fail_count += 1
                continue
        
        # 查找对应的PPG文件（如果提供）
        ppg_path = None
        if ppg_dir:
            # 优先使用subject_name，如果没有则使用video_name
            search_name = subject_name if subject_name else video_name
            
            if dataset_type == 1:
                # Dataset1: 查找gtdump.xmp文件
                ppg_candidates = [
                    os.path.join(ppg_dir, f"{search_name}", "gtdump.xmp"),
                    os.path.join(ppg_dir, f"{search_name}_gtdump.xmp"),
                    os.path.join(ppg_dir, f"{search_name}", "gtdump.txt"),
                ]
            else:  # dataset_type == 2
                # Dataset2: 查找ground_truth.txt文件
                ppg_candidates = [
                    os.path.join(ppg_dir, f"{search_name}", "ground_truth.txt"),
                    os.path.join(ppg_dir, f"{search_name}_ground_truth.txt"),
                    os.path.join(ppg_dir, f"{search_name}", "ground_truth.csv"),
                ]
            
            for candidate in ppg_candidates:
                if os.path.exists(candidate):
                    ppg_path = candidate
                    break
        
        # 输出文件路径（使用subject_name如果可用，否则使用video_name）
        output_name = subject_name if subject_name else video_name
        output_path = os.path.join(output_dir, f"{output_name}.h5")
        
        print(f"\nProcessing: {video_name}")
        print(f"  Video: {video_path}")
        print(f"  Landmark: {landmark_path}")
        if ppg_path:
            print(f"  PPG: {ppg_path}")
        print(f"  Output: {output_path}")
        
        # 处理视频
        if openface_h5_with_ppg(video_path, landmark_path, output_path, ppg_path, dataset_type=dataset_type):
            success_count += 1
            print(f"  ✓ Success")
        else:
            fail_count += 1
            print(f"  ✗ Failed")
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess UBFC-rPPG dataset')
    parser.add_argument('--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('--landmark_dir', type=str, required=True,
                       help='Directory containing OpenFace landmark CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for .h5 files')
    parser.add_argument('--ppg_dir', type=str, default=None,
                       help='Directory containing PPG signal files (optional, for test set)')
    parser.add_argument('--video_pattern', type=str, default='*.avi',
                       help='Video file pattern (default: *.avi)')
    parser.add_argument('--ppg_pattern', type=str, default='*.txt',
                       help='PPG file pattern (default: *.txt)')
    parser.add_argument('--dataset_type', type=int, default=2, choices=[1, 2],
                       help='Dataset type: 1=Dataset1 (gtdump.xmp), 2=Dataset2 (ground_truth.txt)')
    
    args = parser.parse_args()
    
    process_ubfc_dataset(
        video_dir=args.video_dir,
        landmark_dir=args.landmark_dir,
        output_dir=args.output_dir,
        ppg_dir=args.ppg_dir,
        video_pattern=args.video_pattern,
        ppg_pattern=args.ppg_pattern,
        dataset_type=args.dataset_type
    )
