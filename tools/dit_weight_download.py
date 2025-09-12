# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
from torchvision.datasets.utils import download_url
import torch
import os


pretrained_models = {'DiT-XL-2-512x512.pt', 'DiT-XL-2-256x256.pt'}


def find_model(model_name, download_dir='pretrained_models'):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    
    Args:
        model_name (str): Name of the model or path to local checkpoint
        download_dir (str): Directory to save downloaded models (default: 'pretrained_models')
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name, download_dir)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name, download_dir='/storage/U-ViT/weights/DiT'):
    """
    Downloads a pre-trained DiT model from the web.
    
    Args:
        model_name (str): Name of the model to download
        download_dir (str): Directory to save the downloaded model (default: 'pretrained_models')
    """
    assert model_name in pretrained_models
    local_path = f'{download_dir}/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs(download_dir, exist_ok=True)
        web_path = f'https://dl.fbaipublicfiles.com/DiT/models/{model_name}'
        download_url(web_path, download_dir)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download DiT pre-trained models')
    parser.add_argument('--download_dir', type=str, default='/storage/U-ViT/weights/DiT',
                        help='Directory to save downloaded models (default: /storage/U-ViT/weights/DiT)')
    parser.add_argument('--model', type=str, choices=list(pretrained_models), 
                        help='Specific model to download (if not specified, downloads all)')
    
    args = parser.parse_args()
    
    # Download specified model or all models
    if args.model:
        download_model(args.model, args.download_dir)
        print(f'Downloaded {args.model} to {args.download_dir}')
    else:
        for model in pretrained_models:
            download_model(model, args.download_dir)
        print(f'Downloaded all models to {args.download_dir}')