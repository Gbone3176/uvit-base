import torch
import os
import shutil
import numpy as np
import pandas as pd
import libs.autoencoder
import libs.clip
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm
from glob import glob
import einops
import cv2 as cv
import open_clip

from Bio_Con_model import MyCLIPEncoder

import random

random.seed(42)


class MyDatabase(Dataset):
    def __init__(self, root, mode, size=None):
        # from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size
        self.size = size
        # self.ids = glob(self.root + '/caption/*')
        self.paths = []
        self.mode = mode
        # with open(root + f'/final_{self.mode}_filter_cls.txt', 'r', encoding='utf-8') as f:
        self.paths = glob(root + f'{mode}_data/*.jpg')
        if self.mode == 'train':
            self.df = pd.read_csv(
                '/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/TrainingGroundTruth.csv')
        else:
            self.df = pd.read_csv(
                '/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/TestGroundTruth.csv')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        txt_name = img_path.split('/')[-1].split('.')[0]
        ### v3版本只修改训练集的diverse caption
        txt_dir = f'/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/reponse_train918/'
        txt_files = glob(txt_dir + f'{txt_name}_*.txt')
        # with open(f'/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/response_910_{self.mode}/{txt_name}.txt', 'r') as f:
        #     target = f.readlines()
        targets = []
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                target = f.readlines()
                f.close()
                target = [ann.strip() for ann in target]
                target = [ann for ann in target if ann != '']
                if len(target) > 1:
                    target = [' '.join(target)]
                targets += target

        ### token process
        image = cv.cvtColor(cv.resize(cv.imread(img_path), (self.size, self.size)), cv.COLOR_BGR2RGB)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        ### 根据txt_name查询label
        label = self.df.loc[self.df['image'] == txt_name].iloc[0, 1:].values.argmax()

        return image, targets, label


def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    print(args)

    if args.split == "train":
        datas = MyDatabase(root='/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/', mode='train',
                           size=resolution)
        # save_dir = f'/storage/ScientificPrograms/Conditional_Diffusion/U-VIT-G/assets/datasets/ISIC{resolution}_Ori_v3/train'
        save_dir = f'/storage/ScientificPrograms/Conditional_Diffusion/U-VIT-G/assets/datasets/ISIC{resolution}_Ori_v4/train'

    elif args.split == "val":
        datas = MyDatabase(root='/storage/ScientificPrograms/Conditional_Diffusion/ISIC_data/ISIC2018/', mode='test',
                           size=resolution)
        # save_dir = f'/storage/ScientificPrograms/Conditional_Diffusion/U-VIT-G/assets/datasets/ISIC{resolution}_Ori_v3/val'
        save_dir = f'/storage/ScientificPrograms/Conditional_Diffusion/U-VIT-G/assets/datasets/ISIC{resolution}_Ori_v4/val'

    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    os.makedirs(save_dir, exist_ok=True)

    ### image encoding space
    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)
    print('Autoencoder loaded...')

    ### text encoding space
    backbone, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = MyCLIPEncoder(backbone)
    # print('Clip text encoder loaded...')
    model.load_state_dict(torch.load('/storage/ScientificPrograms/Conditional_Diffusion/'
                                     'BiomedCLIP_Finetune/ISIC/Finetune/temp007/Sep21_06-40-35/27-6.7672.pth'))  # 加载新的权重
    print('Clip text encoder loaded with new weights...')
    model.eval()
    model.to(device)
    ###

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas)):
            # x, captions, cls_info = data
            x, captions, label = data

            ### image encoding process
            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)
            ###

            ### text latent process
            texts = tokenizer(captions, context_length=256)
            latent = model(texts.to(device))
            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), {'text': c, 'label': label})
            ###

            ### label process
            # np.save(os.path.join(save_dir, f'{idx}_cls_{label}.npy'), label)


if __name__ == '__main__':
    main()
