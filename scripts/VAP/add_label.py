import torch
import os
import numpy as np
from datasets import ChestXray14Database 
import argparse
from tqdm import tqdm
import re

def main(resolution=256):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'val'], help="Dataset split to process (train or val)")
    parser.add_argument("--version", type=str, required=True,
                        help="Specify the version string used in the save directory, e.g., 'RadBERT'")
    
    args = parser.parse_args()
    print(f"Parsed arguments: {args}")

    version_suffix = args.version.split("/")[-1] # Extracts 'RadBERT' from 'StanfordAIMI/RadBERT'
    print(f"Using version suffix for directory: {version_suffix}")

    # Initialize ChestXray14Database to fetch labels
    if args.split == "train":
        datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                                     csv_file ='/storage/dataset/ChestXray14/reports/train_11_1.csv',
                                     size=resolution,
                                     mode='train')
        save_dir = f'/storage/U-ViT/assets/datasets/VAP/ChestXray14-{resolution}_features-{version_suffix}/train'
    elif args.split == "val":
        datas = ChestXray14Database(root='/storage/dataset/ChestXray14',
                                     csv_file='/storage/dataset/ChestXray14/reports/valid_11_1.csv',
                                     size=resolution,
                                     mode='valid')
        save_dir = f'/storage/U-ViT/assets/datasets/VAP/ChestXray14-{resolution}_features-{version_suffix}/val'
    else:
        raise ValueError(f"Invalid split: {args.split}")

    print(f"Target save directory: {save_dir}")
    if not os.path.exists(save_dir):
        print(f"Error: Directory {save_dir} does not exist. Please check the path and arguments.")
        return

    caption_file_pattern = re.compile(r"(\d+)_0\.npy$") # Simplified regex

    files_to_process = []
    for filename in os.listdir(save_dir):
        if caption_file_pattern.fullmatch(filename): # Use the simplified pattern
            files_to_process.append(filename)
    
    if not files_to_process:
        print(f"No caption files matching the pattern (e.g., idx_0.npy) found in {save_dir}.")
        return

    print(f"Found {len(files_to_process)} caption files to process.")

    for filename in tqdm(files_to_process, desc="Updating npy files"):
        filepath = os.path.join(save_dir, filename)
        
        match = caption_file_pattern.fullmatch(filename)
        if not match: # Should not happen due to pre-filtering
            print(f"Warning: Filename {filename} did not match expected pattern during loop. Skipping.")
            continue
            
        idx_str = match.group(1) # The first captured group is the index 'x'

        try:
            idx = int(idx_str)
        except ValueError:
            print(f"Warning: Could not parse index from filename {filename} (parsed as '{idx_str}'). Skipping.")
            continue

        try:
            latent_data_content = np.load(filepath, allow_pickle=True)

            if isinstance(latent_data_content, dict) and "caption" in latent_data_content and "label" in latent_data_content:
                continue
            elif isinstance(latent_data_content, np.ndarray):
                latent_caption_array = latent_data_content
                
                if idx < len(datas):
                    _, original_captions_dict = datas[idx]
                    label = original_captions_dict.get("label")

                    if label is None:
                        print(f"Warning: Label not found for idx {idx} in captions_dict for file {filename}. Skipping.")
                        continue
                        
                    updated_data = {
                        "caption": latent_caption_array,
                        "label": label
                    }
                    
                    np.save(filepath, updated_data)
                else:
                    print(f"Warning: Index {idx} from file {filename} is out of bounds for the dataset (size {len(datas)}). Skipping.")
            else:
                print(f"Warning: File {filename} contains unexpected data type: {type(latent_data_content)}. Skipping.")

        except FileNotFoundError:
            print(f"Warning: File {filepath} not found during iteration. Skipping.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}. Skipping.")
            
    print("Finished updating npy files.")

if __name__ == '__main__':
    main(resolution=256)