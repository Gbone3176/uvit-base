import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, 
                    help="指定要使用的Bert模型版本, 例如:'StanfordAIMI/RadBERT'")
    args = parser.parse_args()

    version = args.version

    prompts = [
        '',
    ]

    device = 'cuda'
    clip = libs.clip.BertEmbedder(version)
    clip.eval()
    clip.to(device)

    save_dir = f'/storage/U-ViT/assets/datasets/VAP/ChestXray14-{resolution}_features-{version.split("/")[-1]}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    
    latent = clip.encode(prompts)
    c = latent[0].detach().cpu().numpy()
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)
    print("empty_context.npy processes finished")

if __name__ == '__main__':
    main()
