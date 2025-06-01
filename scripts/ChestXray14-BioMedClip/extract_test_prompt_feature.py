import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main():
    prompts = []
    VIS_NUM = 1000
    dirlist = os.listdir("/storage/U-ViT/assets/ClipScore/run_vis_txt-1k")
    dirlist = sorted(dirlist, key=lambda x: int(x.split('.')[0]))
    for i, file in enumerate(dirlist):
        if i >= VIS_NUM:
            break
        file_path = os.path.join("/storage/U-ViT/assets/ClipScore/run_vis_txt-1k", file)
        with open(file_path, 'r') as f:
            prompts.append(f.read())

    device = 'cuda'
    clip = libs.clip.BioMedClipEmbedder()
    clip.eval()
    clip.to(device)

    # save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-256_features/run_vis'
    save_dir = f'/storage/U-ViT/assets/ClipScore/BiomedClip/run_vis'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    latent = clip.encode(prompts)
    for i in range(len(latent)):
        c = latent[i].detach().cpu().numpy()
        np.save(os.path.join(save_dir, f'{i}.npy'), {"prompt":prompts[i], "context":c}, allow_pickle=True)
    print("finished: extract test prompt features")


if __name__ == '__main__':
    main()