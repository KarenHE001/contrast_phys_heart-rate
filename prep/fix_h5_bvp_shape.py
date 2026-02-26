#!/usr/bin/env python3
"""
修复h5文件中bvp的形状问题
将(3, N)或(4, N)的bvp改为(N,)的一维数组
"""

import h5py
import numpy as np
import sys
from pathlib import Path

def fix_h5_bvp(h5_path):
    """修复单个h5文件的bvp形状"""
    try:
        with h5py.File(h5_path, 'r+') as f:
            if 'bvp' not in f:
                print(f"⚠️  {h5_path}: 没有bvp数据集")
                return False
            
            bvp = f['bvp']
            bvp_shape = bvp.shape
            
            # 检查是否需要修复
            if len(bvp_shape) == 1:
                print(f"✓ {h5_path}: bvp形状正确 {bvp_shape}")
                return True
            
            if len(bvp_shape) == 2 and bvp_shape[0] > 1:
                # bvp是(3, N)或(4, N)，需要只取第一行
                print(f"🔧 修复 {h5_path}: bvp形状 {bvp_shape} -> ({bvp_shape[1]},)")
                
                # 读取第一行（BVP信号）
                bvp_data = bvp[0, :]
                
                # 删除旧数据集
                del f['bvp']
                
                # 创建新的一维数据集
                f.create_dataset('bvp', data=bvp_data, compression='gzip')
                
                print(f"✅ 修复完成: bvp形状现在是 ({bvp_data.shape[0]},)")
                return True
            else:
                print(f"⚠️  {h5_path}: 未知的bvp形状 {bvp_shape}")
                return False
                
    except Exception as e:
        print(f"❌ 修复失败 {h5_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python fix_h5_bvp_shape.py <h5_dir>")
        print("示例: python fix_h5_bvp_shape.py datasets/UBFC_h5")
        sys.exit(1)
    
    h5_dir = Path(sys.argv[1])
    if not h5_dir.exists():
        print(f"❌ 目录不存在: {h5_dir}")
        sys.exit(1)
    
    h5_files = list(h5_dir.glob('*.h5'))
    if not h5_files:
        print(f"⚠️  未找到h5文件: {h5_dir}")
        sys.exit(1)
    
    print(f"找到 {len(h5_files)} 个h5文件")
    print("="*60)
    
    fixed = 0
    skipped = 0
    failed = 0
    
    for h5_file in sorted(h5_files):
        if fix_h5_bvp(h5_file):
            fixed += 1
        else:
            failed += 1
    
    print("="*60)
    print(f"修复完成: {fixed} 个成功, {failed} 个失败")
    print("="*60)
