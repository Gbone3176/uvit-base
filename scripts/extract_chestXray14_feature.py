import einops
import torch.nn as nn
import numpy as np
import torch
from datasets import ImageNet
from torch.utils.data import DataLoader
from libs.autoencoder import get_model
import argparse
from tqdm import tqdm
torch.manual_seed(0)
np.random.seed(0)
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)

class ChestXray14Database(Dataset):
    def __init__(self, root="/storage/dataset/ChestXray14", csv_file="/storage/dataset/ChestXray14/reports/train_11_1.csv", size=256, mode='train'):
        """
        Args:
            root (str): 数据集根目录，包含图像和报告的子目录。
            csv_file (str): 包含图像和标签对应关系的CSV文件路径。
            size (int, optional): 图像的目标尺寸（高度和宽度）。默认为None。
        """
        self.root = root
        self.height = self.width = size
        self.df = pd.read_csv(csv_file)
        self.keys = self.df['Image Index'].tolist()
        self.mode = mode

    def _load_image(self, key: str):
        path = os.path.join(self.root, 'images', key)
        return Image.open(path).convert("RGB")

    def _load_target(self, key: str):
        # 获取标签
        labels = self.df[self.df['Image Index'] == key].iloc[:, 2:].values.flatten().tolist()
        label = labels.index(1)

        return label

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        
        if self.width and self.height:
            image = center_crop(self.width, self.height, image).astype(np.float32)
        
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')
        target = self._load_target(key)

        return image, target

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    dataset = ChestXray14Database()
    train_dataset_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    model = get_model('assets/stable-diffusion/autoencoder_kl.pth')
    model = nn.DataParallel(model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # features = []
    # labels = []

    save_dir = f'/storage/U-ViT/assets/datasets/chestXray14_{resolution}_ldm_features'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    idx = 0
    for batch in tqdm(train_dataset_loader):
        img, label = batch
        img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        img = img.to(device)
        moments = model(img, fn='encode_moments')
        moments = moments.detach().cpu().numpy()

        label = torch.cat([label, label], dim=0)
        label = label.detach().cpu().numpy()

        for moment, lb in zip(moments, label):
            np.save(os.path.join(save_dir, f'{idx}.npy'), {"moment":moment, "lb":lb})
            idx += 1

    print(f'save {idx} files')

    # features = np.concatenate(features, axis=0)
    # labels = np.concatenate(labels, axis=0)
    # print(f'features.shape={features.shape}')
    # print(f'labels.shape={labels.shape}')
    # np.save(f'imagenet{resolution}_features.npy', features)
    # np.save(f'imagenet{resolution}_labels.npy', labels)


if __name__ == "__main__":
    main()
