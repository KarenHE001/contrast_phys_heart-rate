"""
快速测试预处理脚本是否正常工作
"""

import h5py
import numpy as np
import os

def test_h5_file(h5_path):
    """测试.h5文件格式是否正确"""
    if not os.path.exists(h5_path):
        print(f"❌ 文件不存在: {h5_path}")
        return False
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"\n✓ 成功打开文件: {h5_path}")
            print(f"  包含的键: {list(f.keys())}")
            
            if 'imgs' in f:
                imgs = f['imgs']
                print(f"  imgs shape: {imgs.shape}")
                print(f"  imgs dtype: {imgs.dtype}")
                print(f"  imgs 数据范围: [{imgs[:].min()}, {imgs[:].max()}]")
            
            if 'bvp' in f:
                bvp = f['bvp']
                print(f"  bvp shape: {bvp.shape}")
                print(f"  bvp dtype: {bvp.dtype}")
                print(f"  bvp 数据范围: [{bvp[:].min():.4f}, {bvp[:].max():.4f}]")
                
                # 检查是否是占位数组（全零）
                if np.allclose(bvp[:], 0):
                    print(f"  ⚠️  警告: bvp 是全零数组（可能是占位数据）")
                else:
                    print(f"  ✓ bvp 包含有效数据")
            
            # 检查长度是否匹配
            if 'imgs' in f and 'bvp' in f:
                if f['imgs'].shape[0] == f['bvp'].shape[0]:
                    print(f"  ✓ imgs 和 bvp 长度匹配")
                else:
                    print(f"  ❌ 错误: imgs 和 bvp 长度不匹配!")
                    return False
            
            return True
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("使用方法: python test_preprocess.py <h5_file_path>")
        print("示例: python test_preprocess.py ./datasets/UBFC_h5/1.h5")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    success = test_h5_file(h5_path)
    
    if success:
        print("\n✓ 文件格式验证通过！")
    else:
        print("\n❌ 文件格式验证失败！")
