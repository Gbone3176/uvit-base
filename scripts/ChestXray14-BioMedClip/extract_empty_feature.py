import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import MSCOCODatabase
import argparse
from tqdm import tqdm


def main():
    prompts = [
        '',
    ]

    device = 'cuda'
    clip = libs.clip.BioMedClipEmbedder()
    clip.eval()
    clip.to(device)

    save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-256_features'
    latent = clip.encode(prompts)['last_hidden_state']
    # print(latent.shape)
    c = latent[0].detach().cpu().numpy()
    np.save(os.path.join(save_dir, f'empty_context.npy'), c)


if __name__ == '__main__':
    main()
