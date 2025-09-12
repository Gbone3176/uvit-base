import os
import numpy as np
import re
from tqdm import tqdm
import pandas as pd
import glob
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_npy_file(file_path):
    try:
        # 读取 .npy 文件
        # data = np.load(file_path, allow_pickle=True).item()
        data = np.load(file_path, allow_pickle=True)
        
        # 输出文件内容和类型信息
        print("文件内容：")
        print(f"数据类型: {type(data)}")

        if isinstance(data, dict):
            print("字典类型数据，包含以下键：")
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"键名: {key}, shape: {value.shape}, dtype: {value.dtype}")
                else:
                    print(f"键名: {key}, 类型: {type(value)}, 值: {value}")
        elif isinstance(data, np.ndarray):
            print(f"数组类型数据，shape: {data.shape}")
            print(f"数据类型: {data.dtype}")
        
        return data
    except Exception as e:
        print(f"读取文件时发生错误: {e}")


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy'))
    print("len(files):" + str(len(files)))
    print("len(files_caption):" + str(len(files_caption)))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions

def calculate_cosine_similarity(embedding1, embedding2):
    """
    计算两个向量或向量组之间的余弦相似度
    
    Args:
        embedding1: 第一个向量或向量组
        embedding2: 第二个向量或向量组
    
    Returns:
        余弦相似度值或矩阵
    """
    # 确保输入是numpy数组
    if not isinstance(embedding1, np.ndarray):
        embedding1 = np.array(embedding1)
    if not isinstance(embedding2, np.ndarray):
        embedding2 = np.array(embedding2)
    
    # 如果是1D向量，重塑为2D
    if embedding1.ndim == 1:
        embedding1 = embedding1.reshape(1, -1)
    if embedding2.ndim == 1:
        embedding2 = embedding2.reshape(1, -1)
    
    # 计算余弦相似度
    similarity = cosine_similarity(embedding1, embedding2)
    
    return similarity

def analyze_zero_values(embedding, name="向量"):
    """
    分析向量中零值的统计信息
    
    Args:
        embedding: 输入的向量或矩阵
        name: 向量的名称，用于显示
    
    Returns:
        dict: 包含零值统计信息的字典
    """
    zero_mask = (embedding == 0)
    zero_count = np.sum(zero_mask)
    total_elements = embedding.size
    zero_percentage = (zero_count / total_elements) * 100
    
    stats = {
        'total_elements': total_elements,
        'zero_count': zero_count,
        'zero_percentage': zero_percentage,
        'non_zero_count': total_elements - zero_count
    }
    
    print(f"\n=== {name}零值统计 ===")
    print(f"总元素数量: {total_elements}")
    print(f"零值数量: {zero_count}")
    print(f"零值百分比: {zero_percentage:.2f}%")
    
    # 如果是2D数组，按行统计零值
    if embedding.ndim == 2:
        zero_rows = np.sum(zero_mask, axis=1)
        completely_zero_rows = np.sum(zero_rows == embedding.shape[1])
        rows_with_zeros = np.sum(zero_rows > 0)
        
        stats.update({
            'zero_rows_count': zero_rows,
            'completely_zero_rows': completely_zero_rows,
            'rows_with_zeros': rows_with_zeros
        })
        
        print(f"每行零值数量: {zero_rows}")
        print(f"完全为零的行数: {completely_zero_rows}")
        print(f"包含零值的行数: {rows_with_zeros}")
    
    # 显示零值位置的前10个索引（如果是1D数组）
    if embedding.ndim == 1:
        zero_indices = np.where(zero_mask)[0]
        stats['zero_indices'] = zero_indices
        if len(zero_indices) > 0:
            print(f"零值位置索引（前10个）: {zero_indices[:10]}")
    
    # 显示非零值的统计信息
    non_zero_values = embedding[~zero_mask]
    if len(non_zero_values) > 0:
        non_zero_stats = {
            'mean': np.mean(non_zero_values),
            'std': np.std(non_zero_values),
            'max': np.max(non_zero_values),
            'min': np.min(non_zero_values)
        }
        stats['non_zero_stats'] = non_zero_stats
        
        print(f"\n=== 非零值统计 ===")
        print(f"非零值数量: {len(non_zero_values)}")
        print(f"非零值均值: {non_zero_stats['mean']:.6f}")
        print(f"非零值标准差: {non_zero_stats['std']:.6f}")
        print(f"非零值最大值: {non_zero_stats['max']:.6f}")
        print(f"非零值最小值: {non_zero_stats['min']:.6f}")
    
    return stats

def visualize_cosine_similarity(similarity_matrix, title="余弦相似度矩阵", save_path=None):
    """
    可视化余弦相似度矩阵
    
    Args:
        similarity_matrix: 余弦相似度矩阵
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn创建热力图
    sns.heatmap(similarity_matrix, 
                annot=False,  # 对于256x256矩阵，不显示数值标注
                cmap='coolwarm', 
                center=0,
                square=True,
                cbar_kws={'label': '余弦相似度'})
    
    plt.title(title, fontsize=16)
    plt.xlabel('向量索引', fontsize=12)
    plt.ylabel('向量索引', fontsize=12)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # get_feature_dir_info("/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-ViT/assets/ChestXray14-256_features/val")
    data1 = load_npy_file("/storage/U-ViT/assets/datasets/VAP/ChestXray14-256_features-BioLinkBERT-base/val/0.npy")
    
    # 加载两个文件的context向量
    # data1 = load_npy_file("/storage/U-ViT/assets/datasets/ChestXray14-256_features-BioLinkBERT-base/run_vis/26.npy")
    # data2 = load_npy_file("/storage/U-ViT/assets/datasets/ChestXray14-256_features-BioLinkBERT-base/run_vis/27.npy")
    
    # # 提取context向量
    # text_embedding1 = data1.item()["context"] if isinstance(data1, np.ndarray) and data1.dtype == 'object' else data1["context"]
    # text_embedding2 = data2.item()["context"] if isinstance(data2, np.ndarray) and data2.dtype == 'object' else data2["context"]
    
    # print(f"向量1形状: {text_embedding1.shape}")
    # print(f"向量2形状: {text_embedding2.shape}")

    # print(text_embedding1[246,:10])   
    # print(text_embedding1[245,:10])
    # print(all(text_embedding1[246,:10] == text_embedding1[245,:10]))

    # visualize_cosine_similarity(text_embedding1, 
    #                                title=f"text_embedding1",
    #                                save_path="/storage/U-ViT/tools/text_embedding1.png")

    # 使用函数分析text_embedding1中的零值
    # embedding1_stats = analyze_zero_values(text_embedding1, "text_embedding1")
    
    # 计算余弦相似度
    # similarity = calculate_cosine_similarity(text_embedding1[:145], text_embedding1[:145])
    # print(f"余弦相似度形状: {similarity.shape}")
    # print(f"余弦相似度: {similarity}")
    
    
    # 如果是矩阵，显示平均相似度
    # if similarity.size > 1:
    #     print(f"平均余弦相似度: {np.mean(similarity)}")
    #     print(f"最大余弦相似度: {np.max(similarity)}")
    #     print(f"最小余弦相似度: {np.min(similarity)}")
        
    #     # 可视化余弦相似度矩阵
    #     print("\n正在生成余弦相似度可视化图...")
    #     visualize_cosine_similarity(similarity, 
    #                                title=f"text_embedding1自身的余弦相似度矩阵 ({similarity.shape[0]}x{similarity.shape[1]})",
    #                                save_path="/storage/U-ViT/tools/text_embedding1自身的余弦相似度矩阵0-145.png")