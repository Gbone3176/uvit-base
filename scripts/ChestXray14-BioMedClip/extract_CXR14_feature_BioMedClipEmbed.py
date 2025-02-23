import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import ChestXray14Database
import argparse
from tqdm import tqdm
from torchvision import transforms

'''
transfer the imgs from pixel space to this latent space
'''

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='val')
    args = parser.parse_args()
    print(args)

    if args.split == "train":
        # datas = ChestXray14Database(root='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14',
        #                      csv_file ='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14/reports/train_11_1.csv',
        #                      size=resolution)
        datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                             csv_file ='/storage/dataset/ChestXray14/reports/train_11_1.csv',
                             size=resolution,
                             mode='train')
        save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-{resolution}_features/train'
    elif args.split == "val":
        # datas = ChestXray14Database(root='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14',
        #                      csv_file='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14/reports/valid_11_1.csv',
        #                      size=resolution)
        datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                             csv_file='/storage/dataset/ChestXray14/reports/valid_11_1.csv',
                             size=resolution,
                             mode='valid')
        save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-{resolution}_features/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)
    clip = libs.clip.BioMedClipEmbedder()
    clip.eval()
    clip.to(device)

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas), total=len(datas)):
            x, captions = data

            if len(x.shape) == 3:
                x = x[None, ...]
            x = torch.tensor(x, device=device)
            moments = autoencoder(x, fn='encode_moments').squeeze(0)
            moments = moments.detach().cpu().numpy()
            np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            latent = clip.encode(captions["report"])['last_hidden_state']

            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)
            # if idx == 0:
            #     print(latent.shape)
            #     break
        print("finished")

if __name__ == '__main__':
    main()