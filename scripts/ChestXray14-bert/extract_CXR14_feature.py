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
    parser.add_argument('--split', default='train')
    parser.add_argument("--version", type=str, required=True, 
                    help="指定要使用的Bert模型版本, 例如:'StanfordAIMI/RadBERT'")
    
    args = parser.parse_args()
    print(args)

    # 以下是BERT系列的model_name
    # michiyasunaga/BioLinkBERT-base 256
    # michiyasunaga/BioLinkBERT-large 256 hidden_size 1024, 不可行
    # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    # cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    # StanfordAIMI/RadBERT

    version = args.version
    print(version)

    if args.split == "train":
        # datas = ChestXray14Database(root='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14',
        #                      csv_file ='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14/reports/train_11_1.csv',
        #                      size=resolution)
        datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                             csv_file ='/storage/dataset/ChestXray14/reports/train_11_1.csv',
                             size=resolution,
                             mode='train')
        save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-{resolution}_features-{version.split("/")[-1]}/train'
    elif args.split == "val":
        # datas = ChestXray14Database(root='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14',
        #                      csv_file='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/datasets/ChestXray14/reports/valid_11_1.csv',
        #                      size=resolution)
        datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                             csv_file='/storage/dataset/ChestXray14/reports/valid_11_1.csv',
                             size=resolution,
                             mode='valid')
        save_dir = f'/storage/U-ViT/assets/datasets/ChestXray14-{resolution}_features-{version.split("/")[-1]}/val'
    else:
        raise NotImplementedError("ERROR!")

    device = "cuda"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    autoencoder = libs.autoencoder.get_model('assets/stable-diffusion/autoencoder_kl.pth')
    autoencoder.to(device)

    
    clip = libs.clip.BertEmbedder(version)
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

            latent = clip.encode(captions["report"])

            for i in range(len(latent)):
                c = latent[i].detach().cpu().numpy()
                np.save(os.path.join(save_dir, f'{idx}_{i}.npy'), c)
            # if idx == 0:
            #     print(latent.shape)
            #     break
        print("finished")

if __name__ == '__main__':
    main()