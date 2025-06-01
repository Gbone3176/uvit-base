import torch
import os
import numpy as np

import sys
print(sys.path)

import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm

def main():
    prompts = []
    VIS_NUM = 1000
    target_dir = "/storage/U-ViT/scripts/Vis_caps/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, 
                    help="指定要使用的Bert模型版本, 例如:'StanfordAIMI/RadBERT'")
    args = parser.parse_args()

    version = args.version


    dirlist = os.listdir("/storage/U-ViT/assets/ClipScore/run_vis_txt-1k")
    dirlist = sorted(dirlist, key=lambda x: int(x.split('.')[0]))

    for i, file in enumerate(dirlist):
        if i >= VIS_NUM:
            break
        file_path = os.path.join("/storage/U-ViT/assets/ClipScore/run_vis_txt-1k", file)

        with open(file_path, 'r') as f:
            content = f.read()
            prompts.append(content)

        # output_path = os.path.join(target_dir, f"{i}.txt")
        # with open(output_path, 'w') as out_f:
        #     out_f.write(content)

    device = 'cuda'
    clip = libs.clip.BertEmbedder(version)
    clip.eval()
    clip.to(device)

    # save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-256_features-{version.split("/")[-1]}/run_vis'
    save_dir = f'/storage/U-ViT/assets/ClipScore/{version.split("/")[-1]}/run_vis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    latent = clip.encode(prompts)
    for i in tqdm(range(len(latent)), total=len(latent)):
        c = latent[i].detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'{i}.npy'), {"prompt":prompts[i], "context":c}, allow_pickle=True)
    print("finished: extract test prompt features for vis")

if __name__ == '__main__':
    main()