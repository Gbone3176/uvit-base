import einops
import torch
import os
import numpy as np
from PIL import Image
import sys
print(sys.path)

import libs.autoencoder
import libs.clip
from datasets import ChestXray14Database
import argparse
from tqdm import tqdm

def main(resolution=256):
    prompts = []
    VIS_NUM = 1000
    target_dir = "/storage/U-ViT/assets/datasets/VAP/ChestXray14-256_features-BioLinkBERT-base/Vis/"

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, 
                    help="指定要使用的Bert模型版本, 例如:'StanfordAIMI/RadBERT'")
    args = parser.parse_args()

    version = args.version

    datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                            csv_file='/storage/dataset/ChestXray14/reports/valid_11_1.csv',
                            size=resolution,
                            mode='valid')

    device = "cuda"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)

    clip = libs.clip.BertEmbedder(version)
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas), total=VIS_NUM):
            if idx > VIS_NUM:
                print(f"Reached VIS_NUM ({VIS_NUM}), stopping.")
                break # This will exit the loop

            x, captions = data

            # 保存用于测试的图片和文本报告
            IMG_SAVE_DIR = "/storage/U-ViT/assets/datasets/VAP/ChestXray14-256_features-BioLinkBERT-base/Vis/img"
            TXT_SAVE_DIR = "/storage/U-ViT/assets/datasets/VAP/ChestXray14-256_features-BioLinkBERT-base/Vis/txt"

            # 创建文件夹（如果不存在）
            os.makedirs(IMG_SAVE_DIR, exist_ok=True)
            os.makedirs(TXT_SAVE_DIR, exist_ok=True)

            
            image_np = einops.rearrange(x, 'c h w -> h w c')
            image_pil = Image.fromarray((image_np * 127.5 + 127.5).astype(np.uint8))
            image_pil.save(os.path.join(IMG_SAVE_DIR, f"{idx}.jpg"))

            report_text = captions["report"]
            # 保存文本
            txt_path = os.path.join(TXT_SAVE_DIR, f"{idx}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report_text)



            # 隐空间文本和图像编码
            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)

            # 保存图像（假设 x 是张量，形状为 [C, H, W]）
            img_path = os.path.join(IMG_SAVE_DIR, f"{idx}.png")


            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(target_dir, f'{idx}.npy'), moments)

            latent = clip.encode(captions["report"])

            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                l = captions["label"]
                 
                np.save(os.path.join(target_dir, f'{idx}_{i}.npy'), {"caption":c, "label":l})

        print("finished")

if __name__ == '__main__':
    main()