#!/usr/bin/env python3
"""
测试文本词汇对图像区域影响的可视化脚本

使用方法:
python test_word_influence.py --config configs/chestXray14_uvit_small_t2i.py --prompt "chest X-ray showing pneumonia in the left lung"
"""

import sys
sys.path.append('.')
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# 导入项目模块
from tools.Visualization.Vision.uvit_attention_vis import UViTAttentionMonitor
from tools.Visualization.Vision.enhanced_attention_vis import EnhancedAttentionVisualizer
from libs.uvit import UViT
from libs.clip import FrozenCLIPEmbedder
from libs.autoencoder import get_model
from libs.clip.simple_tokenizer import SimpleTokenizer
from utils import set_logger, sample_from_model_classifier_free_guidance
import ml_collections

def load_model_and_config(config_path, nnet_path=None):
    """
    加载模型和配置
    """
    # 加载配置
    config_module = config_path.replace('/', '.').replace('.py', '')
    config = __import__(config_module, fromlist=['get_config']).get_config()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载文本编码器
    clip = FrozenCLIPEmbedder(version=config.clip_img_size, device=device)
    
    # 加载自动编码器
    autoencoder = get_model(**config.autoencoder)
    autoencoder.to(device)
    autoencoder.eval()
    
    # 加载U-ViT模型
    nnet = UViT(**config.nnet)
    nnet.to(device)
    nnet.eval()
    
    # 如果提供了模型路径，加载预训练权重
    if nnet_path and os.path.exists(nnet_path):
        checkpoint = torch.load(nnet_path, map_location=device)
        if 'nnet' in checkpoint:
            nnet.load_state_dict(checkpoint['nnet'])
        else:
            nnet.load_state_dict(checkpoint)
        print(f"已加载模型权重: {nnet_path}")
    
    return nnet, autoencoder, clip, config, device

def analyze_word_influence(prompt, nnet, autoencoder, clip, config, device, output_dir):
    """
    分析文本中词汇对生成图像的影响
    """
    print(f"分析提示词: {prompt}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建注意力监控器
    attention_monitor = UViTAttentionMonitor(nnet, model_type='uvit')
    enhanced_visualizer = EnhancedAttentionVisualizer()
    
    # 注册钩子
    attention_monitor.register_hooks()
    
    try:
        # 编码文本
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.encode(prompt)
        words = [tokenizer.decode([token]) for token in tokens]
        
        print(f"检测到的词汇: {words}")
        
        # 准备输入
        prompts = [prompt]
        contexts = clip.encode(prompts)
        
        # 生成图像
        print("正在生成图像...")
        with torch.no_grad():
            # 创建随机噪声
            z = torch.randn(1, *config.z_shape, device=device)
            
            # 采样生成
            samples = sample_from_model_classifier_free_guidance(
                nnet, z, contexts, 
                n_samples=1,
                cfg_scale=config.get('cfg_scale', 7.5),
                sample_steps=config.get('sample_steps', 50)
            )
            
            # 解码图像
            samples = autoencoder.decode(samples)
            
        print("图像生成完成")
        
        # 保存生成的图像
        generated_image = samples[0]
        image_path = os.path.join(output_dir, "generated_image.png")
        
        # 转换为PIL图像并保存
        img_tensor = (generated_image + 1) / 2  # 从[-1,1]转换到[0,1]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_pil = Image.fromarray((img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        img_pil.save(image_path)
        print(f"生成的图像已保存到: {image_path}")
        
        # 获取注意力权重
        attention_weights = {}
        if hasattr(attention_monitor, 'attention_stores') and attention_monitor.attention_stores:
            for layer_name, store in attention_monitor.attention_stores.items():
                if hasattr(store, 'attention_weights') and store.attention_weights:
                    attention_weights[layer_name] = store.attention_weights[-1]
        
        if attention_weights:
            print("正在创建增强的可视化...")
            
            # 创建词汇注意力热图
            heatmap_path = os.path.join(output_dir, "word_attention_heatmap.png")
            enhanced_visualizer.visualize_word_attention_heatmap(
                attention_weights=attention_weights,
                words=words,
                generated_image=generated_image,
                save_path=heatmap_path
            )
            print(f"词汇注意力热图已保存到: {heatmap_path}")
            
            # 创建注意力分析报告
            report_path = os.path.join(output_dir, "attention_analysis_report.html")
            enhanced_visualizer.create_attention_analysis_report(
                attention_weights=attention_weights,
                words=words,
                prompt=prompt,
                save_path=report_path
            )
            print(f"注意力分析报告已保存到: {report_path}")
            
            # 创建渐进式注意力可视化
            progressive_path = os.path.join(output_dir, "progressive_attention.gif")
            enhanced_visualizer.create_progressive_attention_visualization(
                attention_weights=attention_weights,
                words=words,
                generated_image=generated_image,
                save_path=progressive_path
            )
            print(f"渐进式注意力可视化已保存到: {progressive_path}")
            
        else:
            print("警告: 未能获取到注意力权重")
            
        # 使用原始的注意力可视化
        attention_monitor.visualize_all(
            save_dir=os.path.join(output_dir, "basic_attention"),
            word_indices=list(range(len(words))),
            prompt=prompt
        )
        print(f"基础注意力可视化已保存到: {os.path.join(output_dir, 'basic_attention')}")
        
    finally:
        # 移除钩子
        attention_monitor.remove_hooks()
        print("注意力监控钩子已移除")

def main():
    parser = argparse.ArgumentParser(description='测试文本词汇对图像区域的影响')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--model', type=str, help='预训练模型路径（可选）')
    parser.add_argument('--prompt', type=str, required=True, help='输入的文本提示')
    parser.add_argument('--output', type=str, default='./word_influence_results', help='输出目录')
    
    args = parser.parse_args()
    
    print("=== 文本词汇影响分析工具 ===")
    print(f"配置文件: {args.config}")
    print(f"模型路径: {args.model or '使用随机初始化权重'}")
    print(f"文本提示: {args.prompt}")
    print(f"输出目录: {args.output}")
    print()
    
    try:
        # 加载模型和配置
        print("正在加载模型...")
        nnet, autoencoder, clip, config, device = load_model_and_config(args.config, args.model)
        print(f"模型已加载到设备: {device}")
        
        # 分析词汇影响
        analyze_word_influence(
            prompt=args.prompt,
            nnet=nnet,
            autoencoder=autoencoder,
            clip=clip,
            config=config,
            device=device,
            output_dir=args.output
        )
        
        print("\n=== 分析完成 ===")
        print(f"所有结果已保存到: {args.output}")
        print("\n生成的文件:")
        print("- generated_image.png: 生成的图像")
        print("- word_attention_heatmap.png: 词汇注意力热图")
        print("- attention_analysis_report.html: 详细分析报告")
        print("- progressive_attention.gif: 渐进式注意力可视化")
        print("- basic_attention/: 基础注意力可视化结果")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())