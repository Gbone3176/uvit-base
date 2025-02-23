import os
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
import glob

def load_npy_file(file_path):
    try:
        # 读取 .npy 文件
        data = np.load(file_path)
        
        # 输出文件内容
        print("文件内容：")
        print(data)
        
        return data
    except Exception as e:
        print(f"读取文件时发生错误: {e}")


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    print("len(files):" + str(len(files)))
    print("len(files_caption):" + str(len(files_caption)))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions

if __name__ == "__main__":
    get_feature_dir_info("/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-ViT/assets/ChestXray14-256_features/val")
    get_feature_dir_info("/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-Vit-new/U-ViT/assets/ChestXray14-256_features/val")