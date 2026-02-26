#!/bin/bash
# 完整的UBFC数据集预处理流程
# 按照contrast-phys README的指示：
# 1. 使用OpenFace提取landmarks
# 2. 使用内嵌Python脚本生成h5文件（包含imgs和bvp）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 配置路径
OPENFACE_DIR="$PROJECT_DIR/OpenFace"
# 优先使用conda环境中的OpenFace（如果已安装）
CONDA_ENV="/home/vt_ai_test1/mamba-envs/ml"
CONDA_OPENFACE_BIN="$CONDA_ENV/local/bin/FeatureExtraction"
BUILD_OPENFACE_BIN="$OPENFACE_DIR/build/bin/FeatureExtraction"

if [ -f "$CONDA_OPENFACE_BIN" ]; then
    OPENFACE_BIN="$CONDA_OPENFACE_BIN"
elif [ -f "$BUILD_OPENFACE_BIN" ]; then
    OPENFACE_BIN="$BUILD_OPENFACE_BIN"
else
    OPENFACE_BIN="$BUILD_OPENFACE_BIN"  # 用于错误检查
fi

VIDEO_DIR="$PROJECT_DIR/datasets/UBFC_raw/DATASET_2"
LANDMARKS_DIR="$PROJECT_DIR/landmarks"
H5_OUTPUT_DIR="$PROJECT_DIR/datasets/UBFC_h5"

echo "============================================================"
echo "UBFC数据集完整预处理流程"
echo "============================================================"

# 检查OpenFace是否安装
if [ ! -f "$OPENFACE_BIN" ]; then
    echo "❌ OpenFace未安装或未编译"
    echo "请先安装OpenFace:"
    echo "1. 等待git clone完成"
    echo "2. 运行: bash prep/install_openface_step_by_step.sh"
    echo "3. 按照指示编译OpenFace"
    exit 1
fi

echo "✅ OpenFace已安装: $OPENFACE_BIN"

# 创建输出目录
mkdir -p "$LANDMARKS_DIR"
mkdir -p "$H5_OUTPUT_DIR"

# 步骤1: 使用OpenFace提取landmarks
echo "\n============================================================"
echo "步骤1: 使用OpenFace提取landmarks"
echo "============================================================"

# 查找所有视频文件
video_files=($(find "$VIDEO_DIR" -name "vid.avi" | sort))

if [ ${#video_files[@]} -eq 0 ]; then
    echo "❌ 未找到视频文件"
    exit 1
fi

echo "找到 ${#video_files[@]} 个视频文件"

# 提取landmarks（批量处理）
for video_file in "${video_files[@]}"; do
    subject_name=$(basename $(dirname "$video_file"))
    landmark_csv="$LANDMARKS_DIR/${subject_name}.csv"
    
    # 检查是否已处理
    if [ -f "$landmark_csv" ]; then
        echo "⏭️  跳过 $subject_name (landmarks已存在)"
        continue
    fi
    
    echo "处理: $subject_name"
    echo "  视频: $video_file"
    echo "  输出: $landmark_csv"
    
    # 运行OpenFace FeatureExtraction
    # 根据README，使用 -of 参数明确指定输出文件名
    # -2Dfp: 只输出2D landmarks（节省空间和时间）
    # -of: 指定输出文件名（不含扩展名，OpenFace会自动添加.csv）
    "$OPENFACE_BIN" \
        -f "$video_file" \
        -out_dir "$LANDMARKS_DIR" \
        -of "$subject_name" \
        -2Dfp
    
    # 检查输出文件
    if [ ! -f "$landmark_csv" ]; then
        # 如果使用-of参数后文件名不对，尝试查找默认输出文件名
        video_basename=$(basename "$video_file" .avi)
        default_csv="$LANDMARKS_DIR/${video_basename}.csv"
        if [ -f "$default_csv" ]; then
            mv "$default_csv" "$landmark_csv"
        fi
    fi
    
    if [ -f "$landmark_csv" ]; then
        echo "  ✅ 完成"
    else
        echo "  ⚠️  警告: 输出文件不存在，请检查OpenFace输出"
    fi
done

echo "\n✅ Landmarks提取完成"

# 步骤2: 生成h5文件
echo "\n============================================================"
echo "步骤2: 生成h5文件（包含imgs和bvp）"
echo "============================================================"

# 使用Python脚本处理
python3 << 'PYTHON_SCRIPT'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import h5py

project_dir = Path("/home/vt_ai_test1/contrast-phys")
video_dir = project_dir / "datasets/UBFC_raw/DATASET_2"
landmarks_dir = project_dir / "landmarks"
h5_output_dir = project_dir / "datasets/UBFC_h5"

def load_bvp(gt_file):
    """加载BVP信号（从ground_truth.txt）"""
    try:
        # UBFC Dataset2的ground_truth.txt格式：
        # Line 1: PPG signal (BVP信号，空格分隔的1801个值)
        # Line 2: Heart rate (心率)
        # Line 3: Timestep (时间戳)
        with open(gt_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 1:
            raise ValueError(f"文件为空: {gt_file}")
        
        # 只读取第一行（BVP信号）
        bvp = np.array([float(x) for x in lines[0].strip().split()])
        return bvp.astype('float32')
    except Exception as e:
        print(f"⚠️  加载BVP失败 {gt_file}: {e}")
        return None

def openface_h5_with_bvp(video_path, landmark_path, bvp_path, h5_path, store_size=128):
    """
    从OpenFace landmarks裁剪人脸并保存为.h5文件，包含BVP信号
    按照contrast-phys README的指示：
    - 边界框大小是landmarks垂直范围的1.2倍（从第一帧确定）
    - 裁剪后调整到128x128
    """
    try:
        landmark = pd.read_csv(landmark_path)
    except Exception as e:
        print(f"❌ 读取landmark失败 {landmark_path}: {e}")
        return False
    
    # 加载BVP信号
    bvp = load_bvp(bvp_path)
    if bvp is None:
        print(f"⚠️  警告: 无法加载BVP信号，将只保存imgs")
        bvp = None
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return False
    
    total_num_frame = len(landmark)
    
    # 检查列名格式（OpenFace可能生成'success'或' success'）
    success_col = 'success' if 'success' in landmark.columns else ' success'
    x_prefix = 'x_' if 'x_0' in landmark.columns else ' x_'
    y_prefix = 'y_' if 'y_0' in landmark.columns else ' y_'
    
    # 从第一帧确定边界框大小（1.2倍垂直范围）
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
            
            miny = np.min(lm_y)
            maxy = np.max(lm_y)
            y_range = maxy - miny
            # 按照contrast-phys README: 边界框大小是landmarks垂直范围的1.2倍
            bbox_size = int(np.round(1.2 * y_range))
            break
    
    if bbox_size is None:
        print(f"❌ 无法确定边界框大小")
        cap.release()
        return False
    
    # 创建h5文件
    with h5py.File(h5_path, 'w') as f:
        imgs = f.create_dataset('imgs', shape=(total_num_frame, store_size, store_size, 3), 
                                dtype='uint8', chunks=(1, store_size, store_size, 3),
                                compression="gzip", compression_opts=4)
        
        if bvp is not None:
            # 对齐BVP和视频帧数
            min_length = min(total_num_frame, len(bvp))
            bvp_aligned = bvp[:min_length]
            bvp_dataset = f.create_dataset('bvp', data=bvp_aligned, compression='gzip')
        
        # 处理每一帧
        lm_x_prev = None
        lm_y_prev = None
        
        for frame_num in range(total_num_frame):
            if landmark[success_col][frame_num]:
                lm_x_ = []
                lm_y_ = []
                for lm_num in range(68):
                    lm_x_.append(landmark[f'{x_prefix}{lm_num}'][frame_num])
                    lm_y_.append(landmark[f'{y_prefix}{lm_num}'][frame_num])
                
                lm_x_ = np.array(lm_x_)
                lm_y_ = np.array(lm_y_)
                
                # 平滑处理（90%前一个+10%当前）
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
                
                # 计算中心点
                cnt_x = int(np.round((minx+maxx)/2))
                cnt_y = int(np.round((miny+maxy)/2))
            else:
                # 如果这一帧没有成功检测，使用上一帧的位置
                if lm_x_prev is None:
                    continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 裁剪人脸（使用固定的边界框大小）
            bbox_half_size = bbox_size // 2
            y_start = max(0, cnt_y - bbox_half_size)
            y_end = min(frame.shape[0], cnt_y - bbox_half_size + bbox_size)
            x_start = max(0, cnt_x - bbox_half_size)
            x_end = min(frame.shape[1], cnt_x - bbox_half_size + bbox_size)
            
            face = frame[y_start:y_end, x_start:x_end]
            
            # 调整大小到128x128
            if face.size > 0:
                face_resized = cv2.resize(face, (store_size, store_size))
                imgs[frame_num] = face_resized
    
    cap.release()
    return True

# 处理所有subjects
video_files = sorted(video_dir.glob("*/vid.avi"))
print(f"找到 {len(video_files)} 个视频文件")

success_count = 0
for video_file in video_files:
    subject_name = video_file.parent.name
    landmark_csv = landmarks_dir / f"{subject_name}.csv"
    gt_file = video_file.parent / "ground_truth.txt"
    h5_file = h5_output_dir / f"{subject_name}.h5"
    
    # 检查是否已处理
    if h5_file.exists():
        print(f"⏭️  跳过 {subject_name} (h5文件已存在)")
        continue
    
    if not landmark_csv.exists():
        print(f"⚠️  跳过 {subject_name} (landmark文件不存在)")
        continue
    
    print(f"\n处理: {subject_name}")
    print(f"  视频: {video_file}")
    print(f"  Landmark: {landmark_csv}")
    print(f"  BVP: {gt_file}")
    print(f"  输出: {h5_file}")
    
    if openface_h5_with_bvp(video_file, landmark_csv, gt_file, h5_file):
        print(f"  ✅ 完成")
        success_count += 1
    else:
        print(f"  ❌ 失败")

print(f"\n✅ 处理完成: {success_count}/{len(video_files)} 成功")

PYTHON_SCRIPT

echo "\n============================================================"
echo "预处理完成！"
echo "============================================================"
echo "h5文件保存在: $H5_OUTPUT_DIR"
echo "可以开始训练了！"
