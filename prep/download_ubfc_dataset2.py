#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从Google Drive下载UBFC-rPPG Dataset 2的脚本
使用方法：
    python download_ubfc_dataset2.py
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def check_gdown():
    """检查是否安装了gdown"""
    try:
        import gdown
        return True
    except ImportError:
        print("gdown未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        return True

def check_downloaded_files(download_dir):
    """
    检查已下载的文件，返回下载状态
    
    Returns:
        dict: {
            'completed': [已完整下载的subject列表],
            'partial': [部分下载的subject列表（有.part文件）],
            'missing': [缺失的subject列表],
            'total_size': 已下载的总大小（GB）
        }
    """
    download_dir = Path(download_dir)
    
    # UBFC Dataset 2应该有42个subjects（subject1-subject49，但有些缺失）
    # 根据实际下载情况，我们检查所有存在的subject文件夹
    expected_subjects = set()
    completed_subjects = []
    partial_subjects = []
    
    total_size = 0
    
    # 检查所有subject文件夹
    if download_dir.exists():
        for subject_dir in download_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('subject'):
                subject_name = subject_dir.name
                expected_subjects.add(subject_name)
                
                # 检查必需文件
                vid_file = subject_dir / "vid.avi"
                gt_file = subject_dir / "ground_truth.txt"
                
                # 检查是否有.part文件（未完成的下载）
                part_files = list(subject_dir.glob("*.part"))
                
                if part_files:
                    # 有.part文件，说明下载中断
                    partial_subjects.append(subject_name)
                    print(f"  ⚠️  {subject_name}: 部分下载（有.part文件）")
                    # 计算已下载大小
                    for part_file in part_files:
                        total_size += part_file.stat().st_size / (1024**3)
                elif vid_file.exists() and gt_file.exists():
                    # 检查文件大小是否合理（vid.avi应该>100MB）
                    vid_size = vid_file.stat().st_size / (1024**3)
                    if vid_size > 0.1:  # 至少100MB
                        completed_subjects.append(subject_name)
                        total_size += vid_size
                        total_size += gt_file.stat().st_size / (1024**3)
                    else:
                        partial_subjects.append(subject_name)
                        print(f"  ⚠️  {subject_name}: 文件不完整（vid.avi太小: {vid_size:.2f}GB）")
                else:
                    partial_subjects.append(subject_name)
                    print(f"  ⚠️  {subject_name}: 文件缺失")
    
    # 根据实际发现的subjects，推断应该有哪些subjects
    # UBFC Dataset 2通常有: subject1, subject3, subject4, subject5, subject8-subject49（部分缺失）
    # 但我们只检查实际存在的文件夹
    
    return {
        'completed': completed_subjects,
        'partial': partial_subjects,
        'total_size': total_size,
        'found_subjects': list(expected_subjects)
    }

def clean_partial_files(download_dir):
    """清理未完成的.part文件"""
    download_dir = Path(download_dir)
    cleaned = []
    
    if download_dir.exists():
        for part_file in download_dir.rglob("*.part"):
            # 检查对应的完整文件是否存在
            # .part文件名格式可能是: filename.part 或 filename.extxxxxx.part
            part_name = part_file.name
            
            # 尝试找到对应的完整文件
            # 如果是 vid.avi7idub7yy.part，对应的文件应该是 vid.avi
            parent_dir = part_file.parent
            possible_names = []
            
            # 如果.part文件名包含原始文件名，提取它
            if 'vid.avi' in part_name:
                possible_names.append('vid.avi')
            if 'ground_truth.txt' in part_name:
                possible_names.append('ground_truth.txt')
            
            # 也检查所有可能的文件名（去掉.part和随机后缀）
            base_name = part_name.replace('.part', '')
            # 尝试去掉可能的随机后缀（如 vid.avi7idub7yy -> vid.avi）
            if '.' in base_name:
                parts = base_name.split('.')
                if len(parts) >= 2:
                    # 假设最后一部分是扩展名，前面可能有随机后缀
                    ext = parts[-1]
                    # 尝试找到原始文件名
                    for f in parent_dir.glob(f"*.{ext}"):
                        if f.name != part_name and not f.name.endswith('.part'):
                            possible_names.append(f.name)
            
            # 如果找到对应的完整文件，删除.part文件
            should_delete = False
            if possible_names:
                for name in possible_names:
                    full_file = parent_dir / name
                    if full_file.exists() and full_file.stat().st_size > 1000:  # 至少1KB
                        should_delete = True
                        print(f"  清理.part文件（完整文件已存在）: {part_file.name} -> {name}")
                        break
            else:
                # 如果没有找到对应文件，也删除.part文件（可能是旧的下载残留）
                should_delete = True
                print(f"  清理残留.part文件: {part_file.name}")
            
            if should_delete:
                part_file.unlink()
                cleaned.append(str(part_file))
    
    return cleaned

def download_dataset2(workers=4, resume=True):
    """
    下载UBFC-rPPG Dataset 2，支持断点续传
    
    Args:
        workers: 并发下载线程数（默认4，可以增加到8-16以加速）
        resume: 是否启用断点续传（默认True）
    """
    # Google Drive文件夹ID - DATASET_2的直接ID
    folder_id = "1q4vWuF2GJvKP5xyeX8dxaJ2fmq97-4ai"
    
    # 下载目录 - 直接下载到DATASET_2文件夹
    download_dir = Path(__file__).parent.parent / "datasets" / "UBFC_raw" / "DATASET_2"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n下载目录: {download_dir}")
    
    # 检查已下载的文件
    if resume:
        print("\n" + "="*60)
        print("检查已下载的文件...")
        print("="*60)
        status = check_downloaded_files(download_dir)
        
        print(f"\n📊 下载状态:")
        print(f"  ✅ 已完成: {len(status['completed'])} 个subjects")
        if status['completed']:
            print(f"     列表: {', '.join(sorted(status['completed']))}")
        
        print(f"  ⚠️  部分下载: {len(status['partial'])} 个subjects")
        if status['partial']:
            print(f"     列表: {', '.join(sorted(status['partial']))}")
        
        print(f"  📦 已下载总大小: {status['total_size']:.2f} GB")
        
        if status['partial']:
            print(f"\n⚠️  发现 {len(status['partial'])} 个未完成的下载")
            print("  将清理.part文件并重新下载这些subjects...")
            cleaned = clean_partial_files(download_dir)
            if cleaned:
                print(f"  已清理 {len(cleaned)} 个.part文件")
        
        if len(status['completed']) > 0:
            print(f"\n✅ 将跳过已完成的 {len(status['completed'])} 个subjects")
            print("  gdown会自动跳过已存在的文件，只下载缺失的文件")
    
    print(f"\n开始下载UBFC-rPPG Dataset 2...")
    print(f"⚠️  注意: 只下载DATASET_2，不包括DATASET_1")
    print(f"📥 并发下载线程数: {workers} (可以通过--workers参数调整)")
    print(f"🔄 断点续传: {'启用' if resume else '禁用'}")
    
    import gdown
    
    # 重要：gdown的download_folder可能不会自动跳过已存在的文件
    # 我们需要在下载前手动检查并临时重命名已完成的文件
    # 这样gdown会跳过它们，下载完成后再恢复
    
    if resume:
        print(f"\n准备跳过已完成的文件...")
        completed_files = []
        for subject_dir in download_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('subject'):
                vid_file = subject_dir / "vid.avi"
                gt_file = subject_dir / "ground_truth.txt"
                
                # 检查视频文件是否完整（至少100MB）
                if vid_file.exists() and vid_file.stat().st_size > 100 * 1024 * 1024:
                    # 临时重命名，让gdown认为文件不存在
                    temp_name = vid_file.with_suffix('.avi.tmp_skip')
                    vid_file.rename(temp_name)
                    completed_files.append(('vid', temp_name, vid_file))
                    print(f"  跳过: {subject_dir.name}/vid.avi")
                
                # 检查ground_truth文件
                if gt_file.exists() and gt_file.stat().st_size > 1000:
                    temp_name = gt_file.with_suffix('.txt.tmp_skip')
                    gt_file.rename(temp_name)
                    completed_files.append(('gt', temp_name, gt_file))
        
        print(f"  已标记 {len([f for f in completed_files if f[0]=='vid'])} 个视频文件跳过")
    
    # 下载DATASET_2文件夹
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    # gdown的download_folder默认会跳过已存在的文件，实现断点续传
    try:
        gdown.download_folder(
            url, 
            output=str(download_dir), 
            quiet=False, 
            use_cookies=False,
            remaining_ok=True  # 允许断点续传，跳过已存在的文件
        )
    except Exception as e:
        print(f"⚠️  下载过程中出现错误: {e}")
        print("尝试使用基本下载方式...")
        gdown.download_folder(
            url, 
            output=str(download_dir), 
            quiet=False, 
            use_cookies=False
        )
    finally:
        # 恢复临时重命名的文件
        if resume and 'completed_files' in locals():
            print(f"\n恢复已完成的文件...")
            for file_type, temp_path, original_path in completed_files:
                if temp_path.exists():
                    temp_path.rename(original_path)
                    print(f"  恢复: {original_path.parent.name}/{original_path.name}")
    
    # 再次检查下载状态
    print("\n" + "="*60)
    print("下载完成，检查最终状态...")
    print("="*60)
    final_status = check_downloaded_files(download_dir)
    print(f"\n📊 最终状态:")
    print(f"  ✅ 已完成: {len(final_status['completed'])} 个subjects")
    print(f"  ⚠️  部分下载: {len(final_status['partial'])} 个subjects")
    print(f"  📦 总大小: {final_status['total_size']:.2f} GB")
    
    if final_status['partial']:
        print(f"\n⚠️  仍有 {len(final_status['partial'])} 个subjects未完成下载")
        print("  可以重新运行此脚本继续下载")
    
    print(f"\n文件保存在: {download_dir}")
    return download_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载UBFC-rPPG Dataset 2（支持断点续传）')
    parser.add_argument('--workers', type=int, default=4,
                        help='并发下载线程数（默认4，可以增加到8-16以加速，但受网络带宽限制）')
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用断点续传，重新下载所有文件')
    args = parser.parse_args()
    
    print("="*60)
    print("UBFC-rPPG Dataset 2 下载脚本")
    print("="*60)
    
    print("\n📌 关于下载速度的说明:")
    print("  - GPU不能用于下载加速（GPU只用于计算，不用于网络传输）")
    print("  - 下载速度主要受限于:")
    print("    1. 网络带宽（你的Ubuntu服务器到Google Drive的连接速度）")
    print("    2. Google Drive的下载速度限制")
    print("    3. 磁盘I/O速度")
    print("  - 可以通过增加并发线程数来加速（但不要超过网络带宽）")
    print("  - 建议: 如果网络带宽充足，可以设置 --workers 8 或 16")
    
    # 检查磁盘空间
    import shutil
    total, used, free = shutil.disk_usage("/home/vt_ai_test1")
    print(f"\n磁盘空间检查:")
    print(f"  总空间: {total // (1024**3)} GB")
    print(f"  已使用: {used // (1024**3)} GB")
    print(f"  可用空间: {free // (1024**3)} GB")
    
    if free < 50 * (1024**3):  # 至少需要50GB
        print("\n⚠️  警告: 可用空间可能不足，建议至少保留50GB空间")
        response = input("是否继续下载? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # 检查并安装gdown
    check_gdown()
    
    # 下载数据集
    try:
        download_dir = download_dataset2(workers=args.workers, resume=not args.no_resume)
        print(f"\n✅ 下载完成！")
        print(f"请检查下载的文件，然后运行预处理脚本生成h5文件")
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n如果gdown下载失败，可以尝试手动下载:")
        print("1. 访问: https://drive.google.com/drive/folders/1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk")
        print("2. 下载DATASET_2文件夹中的所有subjects")
        print("3. 将文件解压到 datasets/UBFC_raw/ 目录")
        sys.exit(1)
