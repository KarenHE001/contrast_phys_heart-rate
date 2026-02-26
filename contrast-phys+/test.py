# -*- coding: utf-8 -*-
import numpy as np
import h5py
import torch
from PhysNetModel import PhysNet
from utils_data import *
from utils_sig import *
from utils_inference import dl_model
from utils_paths import get_train_exp_dir
from sacred import Experiment
from sacred.observers import FileStorageObserver
import json

ex = Experiment('model_pred', save_git_info=False)

@ex.config
def my_config():
    # ======================================================================
    # 参数设置（测试 / 预测）
    # ======================================================================
    e = None # 模型权重epoch（None表示自动使用最后一个可用epoch）
    label_ratio = 0
    train_exp_num = 12 # 训练 run_id
    train_exp_dir = get_train_exp_dir(label_ratio, train_exp_num)
    # clip长度（秒）: 30s=900帧（与UBFC视频长度匹配，FFT分辨率更高）；10s=300帧（与训练T一致，可与live对齐）
    time_interval = 30  # 用 time_interval=10 可与 live 的 10s 窗口对齐

    # ======================================================================
    # 运行设置（Sacred记录位置）
    # ======================================================================
    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    else:
        device = torch.device('cpu')

@ex.automain
def my_main(_run, e, train_exp_dir, device, time_interval):
    import os
    import glob

    # ======================================================================
    # Step 0: 解析权重epoch（None -> 自动找最后一个）
    # ======================================================================
    if e is None:
        epoch_files = glob.glob(train_exp_dir + '/epoch*.pt')
        if not epoch_files:
            raise FileNotFoundError("未找到模型权重文件: {}/epoch*.pt".format(train_exp_dir))
        # 提取epoch编号并找到最大的
        epoch_nums = []
        for f in epoch_files:
            try:
                epoch_num = int(os.path.basename(f).replace('epoch', '').replace('.pt', ''))
                epoch_nums.append(epoch_num)
            except:
                continue
        if not epoch_nums:
            raise ValueError("无法解析epoch编号: {}".format(epoch_files))
        e = max(epoch_nums)
        print("⚠️  未指定epoch，自动使用最后一个可用的epoch: {}".format(e))

    print("\n" + "="*60)
    print("测试实验信息")
    print("="*60)
    print("测试实验ID: {}".format(int(_run._id)))
    print("加载训练实验目录: {}".format(train_exp_dir))
    print("使用的模型权重: epoch{}.pt".format(e))
    print("="*60 + "\n")

    # ======================================================================
    # Step 1: 读取测试集列表 & 加载模型配置
    # ======================================================================
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory
    print("测试集文件数: {}".format(len(test_list)))
    print("预测结果保存目录: {}".format(pred_exp_dir))
    print("")

    with open(train_exp_dir+'/config.json') as f:
        config_train = json.load(f)

    # ======================================================================
    # Step 2: 构建模型并加载权重
    # ======================================================================
    model = PhysNet(config_train['S'], config_train['in_ch']).to(device).eval()
    model_weight_path = train_exp_dir+'/epoch%d.pt'%(e)
    
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError("模型权重文件不存在: {}".format(model_weight_path))
    
    print("加载模型权重: {}".format(model_weight_path))
    model.load_state_dict(torch.load(model_weight_path, map_location=device)) # load weights to the model
    print("模型加载完成\n")

    # ======================================================================
    # Step 4: 对每个subject切clip并保存预测
    # ======================================================================
    for h5_path in test_list:
        h5_path = str(h5_path)

        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs']
            bvp = f['bvp']
            # bvppeak = f['bvp_peak']
            fs = config_train['fs']

            # 按time_interval（秒）切分为非重叠clip
            duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
            num_blocks = int(duration // time_interval)

            rppg_list = []
            bvp_list = []
            # bvppeak_list = []

            for b in range(num_blocks):
                imgs_clip = imgs[b*time_interval*fs:(b+1)*time_interval*fs]
                rppg_clip = dl_model(model, imgs_clip, device)
                rppg_list.append(rppg_clip)

                bvp_list.append(bvp[b*time_interval*fs:(b+1)*time_interval*fs])
                # bvppeak_list.append(bvppeak[b*time_interval*fs:(b+1)*time_interval*fs])

            rppg_list = np.array(rppg_list)
            bvp_list = np.array(bvp_list)
            # bvppeak_list = np.array(bvppeak_list)
            # results = {'rppg_list': rppg_list, 'bvp_list': bvp_list, 'bvppeak_list':bvppeak_list}
            # 保存格式与README一致：rppg_list + bvp_list
            results = {'rppg_list': rppg_list, 'bvp_list': bvp_list}
            output_file = pred_exp_dir+'/'+h5_path.split('/')[-1][:-3]
            np.save(output_file, results)
            print("  已保存预测结果: {}.npy ({} clips)".format(output_file, len(rppg_list)))
    
    # 测试完成后的总结信息
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("测试实验ID: {}".format(int(_run._id)))
    print("预测结果保存目录: {}".format(pred_exp_dir))
    print("\n保存的文件:")
    for h5_path in test_list:
        subject_name = str(h5_path).split('/')[-1][:-3]
        print("  - {}.npy: {}的预测结果".format(subject_name, subject_name))
    print("\n下一步: 运行评估")
    print("  python evaluate_from_test.py")
    print("  或手动指定路径:")
    print("  python evaluate_from_test.py (修改pred_dir为: {})".format(pred_exp_dir))
    print("="*60 + "\n")