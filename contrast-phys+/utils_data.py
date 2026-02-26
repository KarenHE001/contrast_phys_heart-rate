import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt
import glob
from numpy.random import default_rng

def MR_NIRP_split(val_num=1):

    h5_dir = '../datasets/MR-NIRP'
    train_list = []
    val_list = []

    for subject in range(1,9):
        for task in ['motion', 'still']:
            if os.path.isfile(h5_dir+'/sub%d_%s.h5'%(subject, task)):
                if subject == val_num:
                    val_list.append(h5_dir+'/sub%d_%s.h5'%(subject, task))
                else:
                    train_list.append(h5_dir+'/sub%d_%s.h5'%(subject, task))

    return train_list, val_list
    
def OBF_split(k=10, idx=0):
    h5_dir = '../OBF/h5_align'
    if idx>=k:
        raise(ValueError('invalid idx'))

    train_list = []
    val_list = []

    val_len = 100//k
    val_subject = list(range(idx*val_len+1, (idx+1)*val_len+1))

    for subject in range(1,101):
        for sess in [1,2]:
            if os.path.isfile(h5_dir+'/%03d_RGB_%d.h5'%(subject, sess)):
                if subject in val_subject:
                    val_list.append(h5_dir+'/%03d_RGB_%d.h5'%(subject, sess))
                else:
                    train_list.append(h5_dir+'/%03d_RGB_%d.h5'%(subject, sess))
    return train_list, val_list  

def UBFC_LU_split(test_mode=False):
    # ======================================================================
    # 数据集划分（UBFC）
    # ======================================================================
    # split UBFC dataset into training, validation, and testing parts
    # returns: (train_list, val_list, test_list)
    # TODO (README): if you train on another dataset, define a new split function.
    # 注意：为了向后兼容，如果只需要train和test，可以忽略val_list
    
    # 修改路径：从相对路径改为相对于项目根目录的路径
    # 注意：train.py在contrast-phys+目录下运行，需要返回到项目根目录
    h5_dir = '../datasets/UBFC_h5'  # 修改：从contrast-phys+目录返回到项目根目录
    train_list = []
    val_list = []
    test_list = []

    # 查找所有.h5文件
    all_h5_files = sorted(glob.glob(h5_dir + '/*.h5'))
    
    # ======================================================================
    # Test Mode: use a few files for complete test
    # ======================================================================

    if test_mode:
        # 测试模式：使用少量文件进行完整流程测试
        # 训练集：前4个文件
        # 验证集：第5个文件
        # 测试集：第6个文件（如果有的话）
        print("⚠️  测试模式：只使用少量数据")
        if len(all_h5_files) >= 6:
            train_list = all_h5_files[:4]  # 前4个作为训练集
            val_list = [all_h5_files[4]]   # 第5个作为验证集
            test_list = [all_h5_files[5]]   # 第6个作为测试集
        elif len(all_h5_files) >= 4:
            train_list = all_h5_files[:2]  # 前2个作为训练集
            val_list = [all_h5_files[2]]   # 第3个作为验证集
            test_list = [all_h5_files[3]]  # 第4个作为测试集
        elif len(all_h5_files) >= 3:
            train_list = all_h5_files[:1]  # 1个训练
            val_list = [all_h5_files[1]]  # 1个验证
            test_list = [all_h5_files[2]]  # 1个测试
        else:
            train_list = all_h5_files
            val_list = []
            test_list = []
    else:
        # 正常模式：70/15/15划分（训练集/验证集/测试集）- my setting
        # 使用固定seed保证可复现
        rng = np.random.RandomState(42)
        shuffled_files = rng.permutation(all_h5_files)
        
        total = len(shuffled_files)
        train_num = int(total * 0.70)  # 70%
        val_num = int(total * 0.15)    # 15%
        # test_num = total - train_num - val_num  # 剩余15%
        
        train_list = sorted(shuffled_files[:train_num].tolist())
        val_list = sorted(shuffled_files[train_num:train_num+val_num].tolist())
        test_list = sorted(shuffled_files[train_num+val_num:].tolist())
        
        # 打印划分信息
        train_subjects = [os.path.basename(f).replace('.h5', '') for f in train_list]
        val_subjects = [os.path.basename(f).replace('.h5', '') for f in val_list]
        test_subjects = [os.path.basename(f).replace('.h5', '') for f in test_list]
        print(f"数据集划分: 训练集={len(train_list)}个 ({len(train_list)/total*100:.1f}%), "
              f"验证集={len(val_list)}个 ({len(val_list)/total*100:.1f}%), "
              f"测试集={len(test_list)}个 ({len(test_list)/total*100:.1f}%)")

    return train_list, val_list, test_list    


def MMSE_split_percentage(k=5, idx=0):

    f_sub_num = list(range(5,20)) + list(range(21,28))
    m_sub_num = list(range(1, 18))

    sub = np.array(['F%03d'%n for n in f_sub_num]+['M%03d'%n for n in m_sub_num])

    rng = np.random.default_rng(12345)
    sub = rng.permutation(sub)

    val_len = len(sub)//k
    sub_val = sub[idx*val_len+1:(idx+1)*val_len+1]
    
    all_files_list = glob.glob('../datasets/MMSE_HR_h5/*h5')

    train_list = []
    val_list = []

    for f_name in all_files_list:
        sub = f_name.split('/')[-1][:4]
        if sub in sub_val:
            val_list.append(f_name)
        else:
            train_list.append(f_name)

    return train_list, val_list
    
def PURE_split():

    h5_dir = '../datasets/PURE_h5'
    train_list = []
    val_list = []

    val_subject = [6, 8, 9, 10]

    for subject in range(1,11):
        for sess in [1,2,3,4,5,6]:
            if os.path.isfile(h5_dir+'/%02d-%02d.h5'%(subject, sess)):
                if subject in val_subject:
                    val_list.append(h5_dir+'/%02d-%02d.h5'%(subject, sess))
                else:
                    train_list.append(h5_dir+'/%02d-%02d.h5'%(subject, sess))
    return train_list, val_list  
    
class H5Dataset_(Dataset):

    def __init__(self, train_list, T, label_ratio):
        # ==================================================================
        # README TODO: 根据你的数据集标签情况调整
        # - 这里默认是“全标注”的数据组织方式
        # - label_ratio 控制使用多少标注样本
        # - 如果你的数据集中只有部分视频有标签：
        #   需要把有标签的视频放在 train_list 前部，并相应设置 label_sample_number
        # ==================================================================
        rng = np.random.RandomState(42)
        self.train_list = rng.permutation(train_list)
        self.T = T # video clip length
        self.label_sample_number = int(len(self.train_list) * label_ratio) # number of samples with labels
         
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        if idx < self.label_sample_number:
            label_flag = np.float32(1) # a flag to indicate whether the sample has a label
        else:
            label_flag = np.float32(0)
        
        h5_f = np.random.choice(glob.glob(self.train_list[idx]+'/*.h5'))

        with h5py.File(h5_f, 'r') as f:
            img_length = np.min([f['imgs'].shape[0], f['bvp'].shape[0]])

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            bvp = f['bvp'][idx_start:idx_end].astype('float32')

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq, bvp, label_flag

class H5Dataset(Dataset):

    def __init__(self, train_list, T, label_ratio):
        # ==================================================================
        # README TODO: 根据你的数据集标签情况调整
        # - 默认支持无监督训练：label_ratio=0
        # - 如果部分视频有标签：
        #   需要把有标签的视频放在 train_list 前部，并相应设置 label_sample_number
        # ==================================================================
        rng = np.random.RandomState(42)
        self.train_list = rng.permutation(train_list)
        self.T = T # video clip length
        self.label_sample_number = int(len(self.train_list) * label_ratio) # number of samples with labels

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        if idx < self.label_sample_number:
            label_flag = np.float32(1) # a flag to indicate whether the sample has a label
        else:
            label_flag = np.float32(0)

        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = np.min([f['imgs'].shape[0], f['bvp'].shape[0]])

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            bvp = f['bvp'][idx_start:idx_end].astype('float32')

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq, bvp, label_flag

if __name__ == "__main__":
    a, b = MMSE_split_percentage(0.4)