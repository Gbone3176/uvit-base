import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main(resolution=256):
    prompts = [
        '',
    ]

    device = 'cuda'
    clip = libs.clip.PubMedClipEmbedder()
    clip.eval()
    clip.to(device)

    save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-{resolution}_features-pubmedclip'
    latent = clip.encode(prompts)
    c = latent[0].detach().cpu().numpy()
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)
    print("processes finished")

if __name__ == '__main__':
    main()
