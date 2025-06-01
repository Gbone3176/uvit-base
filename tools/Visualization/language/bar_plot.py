import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm

def save_grouped_barchart(data_dict, output_path, figsize=(12, 6), dpi=300, title=None):
    """
    保存多模型性能对比的分组柱状图（支持中文显示）
    
    参数:
        data_dict (dict): 输入数据字典
        output_path (str): 输出文件路径
        figsize (tuple): 图表尺寸，默认(12,6)
        dpi (int): 输出分辨率，默认300
        title (str): 图表标题（中文）
    """
    # 加载中文字体
    font_path = '/storage/U-ViT/tools/Visualization/language/fonts/SimHei/SimHei.ttf'
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 数据预处理
    categories = list(next(iter(data_dict.values())).keys())
    models = list(data_dict.keys())
    num_models = len(models)
    num_categories = len(categories)
    
    # 创建蓝色渐变颜色
    blues = LinearSegmentedColormap.from_list('blues', ['#E6F2FF', '#003366'])
    color_palette = [blues(i/(num_models-1)) for i in range(num_models)]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算柱状图位置
    bar_width = 0.8 / num_models
    index = np.arange(num_categories)
    # 使用tab10色板生成对比色
    # colors = plt.cm.Set2.colors  # Pastel1最多支持9种颜色
    colors = [
    '#003375',  
    '#005c99',  
    '#0081bd',  
    '#00a5e0',  
    '#f7cadb',  
    '#e77e95',  
    '#b83d5e'   
    ]

    # 绘制柱状图
    for i, (model_name, scores) in enumerate(data_dict.items()):
        values = [scores[cat] for cat in categories]
        ax.bar(index + i * bar_width, values, bar_width, 
               label=model_name, color=colors[i],
               edgecolor='none', linewidth=0.5)
    
    # 设置中文标签
    ax.set_xlabel('词语类别', fontproperties=font_prop, fontsize=18)
    ax.set_ylabel('注意力权重', fontproperties=font_prop, fontsize=18)
    # ax.set_title(title or '不同模型在各类别上的表现对比', 
    #             fontproperties=font_prop, fontsize=20, pad=20)
    ax.set_xticks(index + bar_width * (num_models-1)/2)
    ax.set_xticklabels(categories, fontproperties=font_prop, fontsize=16)
    ax.set_ylim(0, 1)
    
    # 设置图例（模型名称保持英文原样）
    ax.legend(bbox_to_anchor=(0.74, 1), loc='upper left', prop=font_prop, fontsize=20, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

data = {
        # "BioLinkbert-base":{"疾病名词":0.2005, "程度描述": 0.0864, "位置描述": 0.1946, "无关词汇":0.5085},
        "BioLinkbert":{"疾病名词":0.5614, "程度描述": 0.5662, "位置描述": 0.4954, "无关词汇":0.3831},
        "PubMedBERT":{"疾病名词":0.4907, "程度描述": 0.7684, "位置描述": 0.4071, "无关词汇":0.4920},
        "SapBERT":{"疾病名词":0.3045, "程度描述": 0.4063, "位置描述": 0.2675, "无关词汇":0.6339},
        "RadBERT":{"疾病名词":0.3747, "程度描述": 0.5270, "位置描述": 0.4417, "无关词汇":0.3569},
        "BioMedCLIP":{"疾病名词":0.0790, "程度描述": 0.0954, "位置描述": 0.1274, "无关词汇":0.6564},
        "PubMedCLIP":{"疾病名词":0.1507, "程度描述": 0.1812, "位置描述": 0.0838, "无关词汇":0.4522},
        "CLIP":{"疾病名词":0.2420, "程度描述": 0.2588, "位置描述": 0.3368, "无关词汇":0.4186},
        }

fig = save_grouped_barchart(data, output_path="/storage/U-ViT/tools/Visualization/language/Vis/模型对比.png", title="ChestXray14医学文本理解性能对比")
plt.show()
