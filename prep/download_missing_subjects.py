#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只下载缺失的subjects，避免重复下载已存在的文件
"""

import sys
from pathlib import Path
import gdown

def get_missing_subjects():
    """获取缺失的subjects列表"""
    video_dir = Path("datasets/UBFC_raw/DATASET_2")
    
    # 先恢复所有.tmp_skip文件
    for tmp_file in video_dir.rglob("*.tmp_skip"):
        original = tmp_file.with_suffix('')
        if tmp_file.exists():
            tmp_file.rename(original)
    
    # UBFC Dataset 2的subjects列表（根据实际数据集）
    all_subjects = []
    for i in range(1, 50):
        if i not in [2, 6, 7, 19, 21, 28, 29]:  # 这些subjects在Dataset 2中不存在
            all_subjects.append(f"subject{i}")
    
    missing = []
    completed = []
    
    for subject_name in all_subjects:
        subject_dir = video_dir / subject_name
        vid_file = subject_dir / "vid.avi"
        
        # 检查视频文件是否完整（至少100MB）
        if vid_file.exists():
            try:
                size = vid_file.stat().st_size
                if size > 100 * 1024 * 1024:  # 至少100MB
                    completed.append(subject_name)
                else:
                    missing.append(subject_name)
            except:
                missing.append(subject_name)
        else:
            missing.append(subject_name)
    
    return completed, missing

# Google Drive文件ID映射（从之前的下载日志中提取）
SUBJECT_FILE_IDS = {
    'subject1': {'vid': '1qBlkbaB8y3-KlWC_A61KsY42Wjo66ss4', 'gt': '1q7LX_8Ggfl43ZmsFXMrUkV2hNnKr-yTE'},
    'subject3': {'vid': '1tL5EX50qFD8n6VO0wX9x_iOrPKyVR2-W', 'gt': '1tKRzecjw14TGFkizo4XvMEWXeefriiCH'},
    'subject4': {'vid': '1w00W_8bNiKKLvZ-p3n3iESLG_gr9AOoz', 'gt': '1vvyDn_3hzw-AvgX69Ct3O2omH_D_erHj'},
    'subject5': {'vid': '1yGaYtOAGYzI4UOIN-XArEl88eRjRarTc', 'gt': '1yD3ng0b2D5gFdFMfmhOphCjunXeuU4IB'},
    'subject8': {'vid': '1yLAa1v0tdL3TZM4CEXkbslI0qyiqyRkH', 'gt': '1yKqVlkTkBp93KXUvmniOZ3JVfK-HxNnP'},
    'subject9': {'vid': '1yZsWIP2f95RMgauo4qJ2VTdnK3DmFqAc', 'gt': '1yW-OciEs79xm3Me5AFqbsCIipZrj4wwZ'},
    'subject10': {'vid': '1qQ4-jO0tSmL69-yF_Qq3B7fwL12wa70l', 'gt': '1qLMw4RCX7_n87XaE_e-LpI8pFZ3h6ErX'},
    'subject11': {'vid': '1qXH_TXTpu0sqYh1JbgZVCBuTOMI18m6n', 'gt': '1qSZgCAC77QgDWMrLtnjyNSOL-9h7dJJ4'},
    'subject12': {'vid': '1qj8bXSS5tTVHD4UC1j2_rogh0OQHGd5l', 'gt': '1qeLIBTBRULL6k283pQMsdco4ZKzG8wDc'},
    'subject13': {'vid': '1qv-awvaCk9cqMh30WW4Xwsw853dyhMgy', 'gt': '1qsUcZOvEd-A_e_Niay5idM62ron1Rbvg'},
    'subject14': {'vid': '1rFo5IUHKu07r49MFYM_E81nB5O7W2HCN', 'gt': '1rFS0-3aOpD2rNvKclr3_G41NAR29mfKn'},
    'subject15': {'vid': '1rITpoMPp71nI9yApstuTL3RWcyNhLuff', 'gt': '1rHZzYZlSt1yST-BDjbGYspwZ_tNlLlU_'},
    'subject16': {'vid': '1riJP8YwngKAHKmyUd43fEXoseWkYrAfq', 'gt': '1rgnJuHy3qik-LrYmpb4-dk84hiXSrzTf'},
    'subject17': {'vid': '1rp3GayS_C5To2XjMkPdReajc9gFP5G5h', 'gt': '1rojrR7jH64wFOlvvgYpxg5BHyQdEungD'},
    'subject18': {'vid': '1s3yk-QMw76ZF0P8d_T_KMvxQuNx9SxN0', 'gt': '1s2XxazUfBaI-QEXhGdwLRBoavAGxfQL0'},
    'subject20': {'vid': '1sGg7Xm5S2GV8LtQ1H8hqQmOVKOP_xexS', 'gt': '1s5YThjp7ZkDjv9BXsnhf1n-xSFCmLzDP'},
    'subject22': {'vid': '1sW3BLS2J0jcyGKXGJEy2GoBPE63jAgUk', 'gt': '1sThbS1zW-7gMw7x9_MAqoq3P7rJ4l-m5'},
    'subject23': {'vid': '1sm2XTMP7SuCAnhz6MFHJY3fpQcdPK2MJ', 'gt': '1skOIpdO3cy8WceTQtdlGZVIqrol4_3hn'},
    'subject24': {'vid': '1szUW64nqzFSjuv2N_7bkN4GWswo4S5Cx', 'gt': '1swEE0XiJzAYwq9rk7TGBVgomZVhr4p4a'},
    'subject25': {'vid': '1t-rZHnzujc8hU9QuIZYiFfGwVCHqykYU', 'gt': '1t-8g3-2tvtgyNLK6v4ubMRRpOQTDdxTI'},
    'subject26': {'vid': '1t6GfzFkZqLLGrmyas-EsWRVq4_10B7w5', 'gt': '1t5v2Q6F38rSZ9gLfBXZ-Uo7UcUe1hvxf'},
    'subject27': {'vid': '1tH59pcyi2Oplws_qHNA3A6KHdCzyaKJf', 'gt': '1tCe_6Gshg-wTeT3UaNnfeC-fWAAZqZY3'},
    'subject30': {'vid': '1taONp19-1wpAazETtGLG2rmylsm6Cjhp', 'gt': '1tXJX_zCrzE92GznFcPSuD5KyqnQFuGo3'},
    'subject31': {'vid': '1tyEacPj-0Zf1sm9CRoQ6lcHnfoZYeuzJ', 'gt': '1twhHHY91hjG2rPiV2IefWXca9VbDVRfS'},
    'subject32': {'vid': '1uBJ1IhsyY-sg_0SOw1sDBW5Bxnh9Xous', 'gt': '1u3ziH9Fmz5CnwtBCG8BQL1Fgo4P8ii6J'},
    'subject33': {'vid': '1uNR2DJgzPO8bD6nm7wpnxfHk5Wzk7JMu', 'gt': '1uMiq9i6ushno754vimuxa3zT1O5RUeLW'},
    'subject34': {'vid': '1uYVAuQD0nus3Hin8Hkh90kPPGME2o3UO', 'gt': '1uXIEHosBTbcfcjDWeQWSquK5Qw9XNGXU'},
    'subject35': {'vid': '1uvVp-DdNT94DElGX5_d9ln6qrGGXkTKk', 'gt': '1uhR8uFjLRg-zoaPK02-3CHUXl4sc7oMP'},
    'subject36': {'vid': '1v94cnI8eOVR0naa9NZqX6PBJklXOji1Z', 'gt': '1v6cU8zGHOlNsdJDOE8657zTVwn35WiBB'},
    'subject37': {'vid': '1vKB3NVl4YPi3N2Q0et19vCcoDofhN66B', 'gt': '1vIjv_lmxPpqz5cJC-gk_oCF2A7zd7EBa'},
    'subject38': {'vid': '1vWA8pdmx2YrO6sFnU1_1aF6t2HoXQZCl', 'gt': '1vRnzklzv1lhfPHNo8LztBF1RJ65Z719b'},
    'subject39': {'vid': '1vfkQ2HMx319yX2sjQKgrLQiaONl-C2eY', 'gt': '1vf8Jl0teyMCDExz4WAD5xrL-PM7UBPp0'},
    'subject40': {'vid': '1w50AlwOrzhddkK_CbX9j8n0djVif5Te7', 'gt': '1w390Xlz-dQKVDxx4otan9pbIwXVOrldR'},
    'subject41': {'vid': '1wRmd9DLW09kcxO8_KKcUivDX3NIouanD', 'gt': '1wHy0jozHq04aJY1orom4zX09ALVn6f6U'},
    'subject42': {'vid': '1wb08DGjJIcmXtlXUkCqDyNZDOK5NoiVz', 'gt': '1w_1W7ilPBCmuJp_xSbUAiielLgB7zxHV'},
    'subject43': {'vid': '1wp0NHRoRK8DqBK-rmkE1LUtPgxeNIgtu', 'gt': '1wl-JPJ2I6kbfOvCEkEHaxqifRSYhWxSN'},
    'subject44': {'vid': '1x75z47xaez5H6R2EvzjrfkuOQDQhXoQq', 'gt': '1x3yxN1b6zjokRW2VoAyAstzlC5izIXcA'},
    'subject45': {'vid': '1xFji9m55jZ02pCCvQGzY1zT5NFJ8DMGi', 'gt': '1xEjWcpunrXDiIPmvLfS1VSEfWEQckFJV'},
    'subject46': {'vid': '1xRpdeqvWZ8QRrn3e1WsFYOVtSpQ13ezh', 'gt': '1xItAgY8Air0onXVK_eM_FK4idoTdPLEJ'},
    'subject47': {'vid': '1xe7CPKHZTKt9K7O7AIElW3ZwuOZjV3ri', 'gt': '1xZGV8XsgMg_XX5cuFUQpLCXfNsHF-2UP'},
    'subject48': {'vid': '1y0-8B5WvxBQEqWCQ0CWDFNrIhi6JsI68', 'gt': '1xzTUqwcVYHRLyApEXEGVOfA0PAtqHDiG'},
    'subject49': {'vid': '1y7cSxcCLD7qpkKfPUxZ2rnOydLH56kib', 'gt': '1y7_UASK0V1-a8dXhsrtLnwhod90Wfzs2'},
}

def download_subject(subject_name, file_ids):
    """下载单个subject的文件"""
    video_dir = Path("datasets/UBFC_raw/DATASET_2")
    subject_dir = video_dir / subject_name
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    vid_file = subject_dir / "vid.avi"
    gt_file = subject_dir / "ground_truth.txt"
    
    success = True
    
    # 下载ground_truth.txt
    if not gt_file.exists() or gt_file.stat().st_size < 1000:
        print(f"  下载 {subject_name}/ground_truth.txt...")
        try:
            url = f"https://drive.google.com/uc?id={file_ids['gt']}"
            gdown.download(url, str(gt_file), quiet=False)
            print(f"  ✅ ground_truth.txt 完成")
        except Exception as e:
            print(f"  ❌ ground_truth.txt 失败: {e}")
            success = False
    else:
        print(f"  ⏭️  跳过 {subject_name}/ground_truth.txt (已存在)")
    
    # 下载vid.avi
    if not vid_file.exists() or vid_file.stat().st_size < 100 * 1024 * 1024:
        print(f"  下载 {subject_name}/vid.avi...")
        try:
            url = f"https://drive.google.com/uc?id={file_ids['vid']}"
            gdown.download(url, str(vid_file), quiet=False)
            print(f"  ✅ vid.avi 完成")
        except Exception as e:
            print(f"  ❌ vid.avi 失败: {e}")
            success = False
    else:
        print(f"  ⏭️  跳过 {subject_name}/vid.avi (已存在)")
    
    return success

if __name__ == "__main__":
    print("="*60)
    print("只下载缺失的subjects")
    print("="*60)
    
    completed, missing = get_missing_subjects()
    
    print(f"\n✅ 已完成: {len(completed)} 个subjects")
    print(f"❌ 缺失: {len(missing)} 个subjects")
    
    if not missing:
        print("\n✅ 所有subjects都已下载完成！")
        sys.exit(0)
    
    print(f"\n需要下载的subjects: {', '.join(missing)}")
    print(f"\n开始下载 {len(missing)} 个subjects...")
    print("="*60)
    
    success_count = 0
    for i, subject_name in enumerate(missing, 1):
        print(f"\n[{i}/{len(missing)}] 处理: {subject_name}")
        
        if subject_name not in SUBJECT_FILE_IDS:
            print(f"  ⚠️  警告: {subject_name} 的文件ID未找到，跳过")
            continue
        
        if download_subject(subject_name, SUBJECT_FILE_IDS[subject_name]):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"下载完成: {success_count}/{len(missing)} 成功")
    print("="*60)
