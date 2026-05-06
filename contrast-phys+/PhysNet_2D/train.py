# -*- coding: utf-8 -*-
import sys
import os
# PhysNet 2D/ -> contrast-phys+/ (loss, utils_*, IrrelevantPowerRatio)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
from EfficientPhysNet import EfficientPhysNet
from loss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio

from utils_data import *
from utils_sig import *
from utils_paths import format_label_ratio
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('model_train', save_git_info=False)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

@ex.config
def my_config():
    # ======================================================================
    # 参数设置（训练超参数）
    # ======================================================================
    total_epoch = 30 # total number of epochs for training the model
    lr = 1e-5 # learning rate
    in_ch = 3 # TODO (README): number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.

    # ======================================================================
    # 参数设置（ST-rPPG 模块）
    # ======================================================================
    fs = 30 # TODO (README): video frame rate, modify if your fps is not 30 fps.
    T = fs * 10 # temporal dimension of ST-rPPG block, default is 10 seconds.
    S = 2 # spatial dimenion of ST-rPPG block, default is 2x2.

    # ======================================================================
    # hyperparms 参数设置（rPPG 时空采样）
    # ======================================================================
    delta_t = int(T/2) # time length of each rPPG sample
    K = 4 # the number of rPPG samples at each spatial position

    # ======================================================================
    # README 中需要按数据集调整的项
    # ======================================================================
    label_ratio = 0 # TODO (README): labeled data ratio. Set 0 for fully unsupervised.

    # ======================================================================
    # 实验/运行设置
    # ======================================================================
    # 测试模式：设置为True时只使用少量文件快速测试
    test_mode = False

    train_exp_name = format_label_ratio(label_ratio)
    # 绝对路径 -> results 永远写入 PhysNet 2D/results/，与 3D results/ 隔离
    result_dir = os.path.join(_HERE, "results", format_label_ratio(label_ratio))
    os.makedirs(result_dir, exist_ok=True)
    ex.observers.append(FileStorageObserver(result_dir))

@ex.automain
def my_main(_run, total_epoch, T, S, lr, result_dir, fs, delta_t, K, in_ch, label_ratio, test_mode):

    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path
    
    print("\n" + "="*60)
    print("训练实验信息")
    print("="*60)
    print("训练实验ID: {}".format(int(_run._id)))
    print("结果保存目录: {}".format(exp_dir))
    if test_mode:
        print("⚠️  测试模式：使用少量数据快速测试")
    print("="*60 + "\n")

    # ======================================================================
    # 数据集划分（README: need to define your split function) including split ratio and test path
    # ======================================================================
    print("===开始训练初始化===")
    train_list, val_list, test_list = UBFC_LU_split(test_mode=test_mode) # TODO (README): define split for your dataset
    print("训练集: {} 个文件, 验证集: {} 个文件, 测试集: {} 个文件".format(len(train_list), len(val_list), len(test_list)))
    np.save(exp_dir+'/train_list.npy', train_list)
    np.save(exp_dir+'/val_list.npy', val_list)
    np.save(exp_dir+'/test_list.npy', test_list)
    print("已保存训练集、验证集和测试集文件列表到: {}".format(exp_dir))

    # define the dataloader
    print("创建数据集和DataLoader...")
    train_dataset = H5Dataset(train_list, T, label_ratio) # please read the code about H5Dataset when preparing your dataset
    # 创建验证集数据集和DataLoader
    val_dataset = H5Dataset(val_list, T, label_ratio) if len(val_list) > 0 else None
    # 修改：如果训练集文件数少于batch_size，设置drop_last=False以避免没有数据可训练
    # 注意：对比学习需要batch_size=2（两个视频），但如果数据量少，可以设置drop_last=False
    drop_last_setting = len(train_list) >= 2  # 只有当训练集文件数>=2时才drop_last
    # GPU上可以启用多进程和pin_memory加速，CPU上使用单进程
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False
    train_dataloader = DataLoader(train_dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last_setting)
    # 创建验证集DataLoader（如果验证集不为空）
    val_dataloader = None
    if val_dataset is not None and len(val_list) >= 2:
        val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, 
                                   num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    print("训练集DataLoader创建完成，数据集长度: {}".format(len(train_dataset)))
    if val_dataset is not None:
        print("验证集DataLoader创建完成，数据集长度: {}".format(len(val_dataset)))
    
    # define the model and loss
    print("初始化模型和损失函数...")
    # model = PhysNet(S, in_ch=in_ch).to(device).train() 
    model = EfficientPhysNet(S, in_ch=in_ch).to(device).train()
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)
    print("开始训练，共 {} 个epoch".format(total_epoch))
    import sys
    sys.stdout.flush()
    
    # 用于跟踪最佳验证loss和对应的epoch
    best_val_loss = float('inf')
    best_epoch = -1

    for e in range(total_epoch):
        print("\n=== Epoch {}/{} ===".format(e+1, total_epoch))
        sys.stdout.flush()
        # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
        # 修改：我们的视频长度约51-67秒，平均约60秒，使用60秒
        video_length_seconds = 60  # 修改：适配我们的视频长度（约51-67秒，平均60秒）
        num_iterations = np.round(video_length_seconds/(T/fs)).astype('int')
        print("  每个epoch有 {} 个iteration".format(num_iterations))
        sys.stdout.flush()
        
        # 训练阶段
        model.train()
        epoch_train_losses = []
        pbar = tqdm(range(num_iterations), desc=f"Epoch {e+1}/{total_epoch}", leave=False, dynamic_ncols=True)
        for it in pbar:
            batch_count = 0
            for imgs, GT_sig, label_flag in train_dataloader: # dataloader randomly samples a video clip with length T
                batch_count += 1
                if batch_count == 1 and it == 0:
                    print("  开始处理第一个batch...")
                    sys.stdout.flush()
                imgs = imgs.to(device)
                GT_sig = GT_sig.to(device)
                label_flag = label_flag.to(device)
                
                # model forward propagation
                if batch_count == 1 and it == 0:
                    print("  数据已转移到设备，开始模型前向传播...")
                    sys.stdout.flush()
                model_output = model(imgs) 
                rppg = model_output[:,-1] # get rppg
                if batch_count == 1 and it == 0:
                    print("  模型前向传播完成，开始计算损失...")
                    sys.stdout.flush()

                # define the loss functions
                loss, p_loss, n_loss, p_loss_gt, n_loss_gt = loss_func(model_output, GT_sig, label_flag)
                if batch_count == 1 and it == 0:
                    print("  损失计算完成，Train Loss: {:.6f}".format(loss.item()))
                    sys.stdout.flush()

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))

                # 累积训练loss用于计算epoch平均loss
                epoch_train_losses.append(loss.item())
                
                # save loss values and IPR (训练阶段)
                ex.log_scalar("train_loss", loss.item())
                ex.log_scalar("train_p_loss", p_loss.item())
                ex.log_scalar("train_n_loss", n_loss.item())
                ex.log_scalar("train_p_loss_gt", p_loss_gt.item())
                ex.log_scalar("train_n_loss_gt", n_loss_gt.item())
                ex.log_scalar("train_ipr", ipr.item())
                
                if batch_count % 10 == 0:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 计算epoch平均训练loss
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0.0
        print("  Epoch {} 平均训练Loss: {:.6f}".format(e+1, avg_train_loss))
        ex.log_scalar("epoch_train_loss", avg_train_loss, step=e+1)
        sys.stdout.flush()
        
        # 验证阶段
        if val_dataloader is not None:
            model.eval()  # 切换到评估模式
            epoch_val_losses = []
            epoch_val_p_losses = []
            epoch_val_n_losses = []
            epoch_val_p_gt_losses = []
            epoch_val_n_gt_losses = []
            epoch_val_iprs = []
            
            print("  开始验证集评估...")
            sys.stdout.flush()
            with torch.no_grad():  # 验证时不需要计算梯度
                video_length_seconds = 60
                num_val_iterations = np.round(video_length_seconds/(T/fs)).astype('int')
                for it in range(num_val_iterations):
                    for imgs, GT_sig, label_flag in val_dataloader:
                        imgs = imgs.to(device)
                        GT_sig = GT_sig.to(device)
                        label_flag = label_flag.to(device)
                        
                        # model forward propagation
                        model_output = model(imgs)
                        rppg = model_output[:,-1]  # get rppg
                        
                        # 计算损失（不进行反向传播）
                        loss, p_loss, n_loss, p_loss_gt, n_loss_gt = loss_func(model_output, GT_sig, label_flag)
                        
                        # 计算IPR
                        ipr = torch.mean(IPR(rppg.clone().detach()))
                        
                        # 累积验证loss
                        epoch_val_losses.append(loss.item())
                        epoch_val_p_losses.append(p_loss.item())
                        epoch_val_n_losses.append(n_loss.item())
                        epoch_val_p_gt_losses.append(p_loss_gt.item())
                        epoch_val_n_gt_losses.append(n_loss_gt.item())
                        epoch_val_iprs.append(ipr.item())
            
            # 计算epoch平均验证loss
            avg_val_loss = np.mean(epoch_val_losses) if epoch_val_losses else 0.0
            avg_val_p_loss = np.mean(epoch_val_p_losses) if epoch_val_p_losses else 0.0
            avg_val_n_loss = np.mean(epoch_val_n_losses) if epoch_val_n_losses else 0.0
            avg_val_p_gt = np.mean(epoch_val_p_gt_losses) if epoch_val_p_gt_losses else 0.0
            avg_val_n_gt = np.mean(epoch_val_n_gt_losses) if epoch_val_n_gt_losses else 0.0
            avg_val_ipr = np.mean(epoch_val_iprs) if epoch_val_iprs else 0.0
            
            log_fmt = "  Epoch {} 平均验证Loss: {:.6f} (P: {:.6f}, N: {:.6f}, IPR: {:.6f})"
            if label_ratio > 0 and (avg_val_p_gt != 0 or avg_val_n_gt != 0):
                log_fmt = "  Epoch {} 平均验证Loss: {:.6f} (P: {:.6f}, N: {:.6f}, P_GT: {:.6f}, N_GT: {:.6f}, IPR: {:.6f})"
                print(log_fmt.format(e+1, avg_val_loss, avg_val_p_loss, avg_val_n_loss, avg_val_p_gt, avg_val_n_gt, avg_val_ipr))
            else:
                print(log_fmt.format(e+1, avg_val_loss, avg_val_p_loss, avg_val_n_loss, avg_val_ipr))
            sys.stdout.flush()
            
            # 保存验证指标
            ex.log_scalar("val_loss", avg_val_loss, step=e+1)
            ex.log_scalar("val_p_loss", avg_val_p_loss, step=e+1)
            ex.log_scalar("val_n_loss", avg_val_n_loss, step=e+1)
            ex.log_scalar("val_p_loss_gt", avg_val_p_gt, step=e+1)
            ex.log_scalar("val_n_loss_gt", avg_val_n_gt, step=e+1)
            ex.log_scalar("val_ipr", avg_val_ipr, step=e+1)
            
            # 检查是否为最佳模型（验证loss更低）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = e
                print("  ✅ 发现更好的模型！验证Loss: {:.6f} (Epoch {})".format(best_val_loss, best_epoch+1))
                sys.stdout.flush()
        else:
            print("  跳过验证（验证集为空或文件数不足）")
            sys.stdout.flush()

        # save model checkpoints
        print("保存Epoch {}的模型检查点...".format(e+1))
        sys.stdout.flush()
        checkpoint_path = exp_dir+'/epoch%d.pt'%e
        torch.save(model.state_dict(), checkpoint_path)
        print("Epoch {} 完成，模型权重已保存到: {}".format(e+1, checkpoint_path))
        sys.stdout.flush()
        
        # 如果是当前最佳模型，额外保存一份best_model.pt
        if val_dataloader is not None and e == best_epoch:
            best_model_path = exp_dir+'/best_model.pt'
            torch.save(model.state_dict(), best_model_path)
            print("  💾 已保存最佳模型到: {} (验证Loss: {:.6f})".format(best_model_path, best_val_loss))
            sys.stdout.flush()
    
    # 训练完成后的总结信息
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("训练实验ID: {}".format(int(_run._id)))
    print("结果保存目录: {}".format(exp_dir))
    print("\n保存的文件:")
    print("  - train_list.npy: 训练集文件列表")
    print("  - val_list.npy: 验证集文件列表")
    print("  - test_list.npy: 测试集文件列表")
    print("  - epoch0.pt ~ epoch{}.pt: 每个epoch的模型权重".format(total_epoch-1))
    if val_dataloader is not None and best_epoch >= 0:
        print("  - best_model.pt: 最佳模型（Epoch {}, 验证Loss: {:.6f}）".format(best_epoch+1, best_val_loss))
    print("  - config.json: 训练配置（Sacred自动保存）")
    print("  - run.json: 运行信息（Sacred自动保存）")
    print("  - metrics.json: 训练指标（Sacred自动保存）")
    print("\n重要说明:")
    print("  - 每个epoch都会保存模型权重（epoch0.pt ~ epoch{}.pt）".format(total_epoch-1))
    print("  - 训练是连续的：每个epoch在前一个epoch的基础上继续训练")
    print("  - 最后一个epoch（epoch{}.pt）不一定是最佳模型，可能出现过拟合".format(total_epoch-1))
    if val_dataloader is not None and best_epoch >= 0:
        print("  - 建议使用最佳模型（best_model.pt，对应Epoch {}）进行测试".format(best_epoch+1))
        print("  - 或手动指定epoch: python test.py with train_exp_num={} e={}".format(int(_run._id), best_epoch))
    else:
        print("  - 由于没有验证集，默认使用最后一个epoch（epoch{}.pt）".format(total_epoch-1))
    print("\n下一步: 运行测试")
    if val_dataloader is not None and best_epoch >= 0:
        print("  使用最佳模型: python test.py with train_exp_num={} e={}".format(int(_run._id), best_epoch))
    print("  使用默认（最后一个epoch）: python test.py with train_exp_num={}".format(int(_run._id)))
    print("="*60 + "\n")