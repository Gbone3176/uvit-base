import torch
import os
import numpy as np
import libs.autoencoder
import libs.clip
from datasets import ChestXray14Database, MSCOCODatabase
import argparse
from tqdm import tqdm
from spb import SPB


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
        save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-{resolution}_features/train_query'
    elif args.split == "val":
        # datas = ChestXray14Database(root='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14',
        #                      csv_file='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14/reports/valid_11_1.csv',
        #                      size=resolution)
        datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                             csv_file='/storage/dataset/ChestXray14/reports/valid_11_1.csv',
                             size=resolution,
                             mode='valid')
        save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-{resolution}_features/val_query'
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

    # set up SPB
    checkpoint = torch.load('/storage/U-ViT/assets/Prior_SPB/last.ckpt')
    spb_state_dict = {k.replace('sentence_bank.', ''): v 
                    for k, v in checkpoint['state_dict'].items() 
                    if k.startswith('sentence_bank.')}
    spb = SPB(768, 512)
    spb.load_state_dict(spb_state_dict)
    spb.set_temp(epoch=1, max_epoch=1, strategy='fixed')
    spb.eval()

    with torch.no_grad():
        for idx, data in tqdm(enumerate(datas), total=len(datas)):
            x, captions = data

            # if len(x.shape) == 3:
            #     x = x[None, ...]
            # x = torch.tensor(x, device=device)
            # moments = autoencoder(x, fn='encode_moments').squeeze(0)
            # moments = moments.detach().cpu().numpy()
            # np.save(os.path.join(save_dir, f'{idx}.npy'), moments)

            latent = clip.encode(captions["report"])['last_hidden_state']

            for i in range(len(latent)):
                c = latent[i].detach().cpu()
                output_query, embed_ind_query = spb.query(c) # 思路1：匹配与word embedding最相近的sentence prototype
                c = output_query.numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)
        print(f"finished: extract {args.split} datasets text features")

if __name__ == '__main__':
    main()