import numpy as np
import os

def check_npy_content(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        data = np.load(file_path, allow_pickle=True)
        print(data)
    except Exception as e:
        print(f"Error loading or reading {file_path}: {e}")


file_to_check = "/storage/U-ViT/assets/datasets/VAP/ChestXray14-256_features-BioLinkBERT-base/val/0_0.npy"
print(f"Attempting to check: {file_to_check}")

check_npy_content(file_to_check)