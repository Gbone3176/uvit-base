# Description: Fetch the image-text pairs from the dataset and save them to the testfiles folder using multiprocessing
import os
from PIL import Image  
import numpy as np  
import pandas as pd  
from multiprocessing import Pool

# Extract the image-text pairs
output_dir_images = '/storage/U-ViT/tools/clip-score/testfiles/images-real'
output_dir_images_gen = '/storage/U-ViT/tools/clip-score/testfiles/images-gen'
output_dir_images_gen_query = '/storage/U-ViT/tools/clip-score/testfiles/images-gen-query'
output_dir_captions = '/storage/U-ViT/tools/clip-score/testfiles/captions-real'
# os.makedirs(output_dir_images, exist_ok=True)
# os.makedirs(output_dir_images_gen, exist_ok=True)
# os.makedirs(output_dir_captions, exist_ok=True)
# os.makedirs(output_dir_images_gen_query, exist_ok=True)

data = pd.read_csv('/storage/dataset/ChestXray14/reports/valid_11_1.csv')
img_list = data['Image Index'].tolist()

def process_image_text_pair(i):
    image_name = img_list[i]
    image_path = os.path.join(output_dir_images, f'image_{i}.png')
    caption_path = os.path.join(output_dir_captions, f'caption_{i}.txt')
    
    # img_pt = f"/storage/U-ViT/sample/ChestXray-t2i-query/{i}.png"   # 生成图像路径
    img_pt = os.path.join('/storage/dataset/ChestXray14/images', image_name) # 真实图像路径
    cap_pt = os.path.join('/storage/dataset/ChestXray14/reports/valid', image_name.replace('.png', '_0.txt'))

    img = Image.open(img_pt)
    img = img.resize((256, 256))  # 调整图像大小
    img.save(image_path)
    with open(cap_pt, 'r') as f:
        caption = f.read()
    with open(caption_path, 'w') as f:
        f.write(caption)
    print(f'pair of {image_name} saved')

if __name__ == '__main__':
    with Pool(processes=os.cpu_count()) as pool:
        pool.map(process_image_text_pair, range(len(img_list)))
