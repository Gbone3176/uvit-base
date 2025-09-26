import ml_collections
import torch
from torch import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import utils
from datasets import get_dataset
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import einops
import libs.autoencoder
import libs.clip
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from typing import List, Tuple, Dict, Optional
import re
from collections import defaultdict


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


class DiffusionTargetFunction:
    """扩散模型专用的target函数，用于计算特定词汇对生成图像的贡献度"""
    
    def __init__(self, tokenizer=None, keyword=None, token_positions=None):
        self.tokenizer = tokenizer
        self.keyword = keyword
        self.token_positions = token_positions or []
        
    def __call__(self, model_output, context=None, timestep=None):
        """计算目标分数，关注特定词汇的贡献"""
        if context is not None and len(self.token_positions) > 0:
            # 基于文本context和token位置计算目标分数
            batch_size = model_output.shape[0]
            
            # 计算与关键词相关的context特征的影响
            if len(context.shape) == 3:  # [batch, seq_len, dim]
                # 提取关键词对应位置的context特征
                keyword_context = torch.zeros_like(context)
                for pos in self.token_positions:
                    if pos < context.shape[1]:
                        keyword_context[:, pos, :] = context[:, pos, :]
                
                # 计算关键词context与模型输出的相关性
                context_norm = F.normalize(keyword_context.mean(dim=1), dim=-1)  # [batch, dim]
                output_flat = model_output.view(batch_size, -1)  # [batch, spatial*channels]
                output_norm = F.normalize(output_flat, dim=-1)
                
                # 计算相似度作为目标分数
                if context_norm.shape[-1] != output_norm.shape[-1]:
                    # 如果维度不匹配，使用简化的计算方式
                    target_score = model_output.mean()
                else:
                    similarity = torch.sum(context_norm * output_norm, dim=-1)
                    target_score = similarity.mean()
            else:
                target_score = model_output.mean()
        else:
            # 默认使用输出的均值作为目标分数
            target_score = model_output.mean()
            
        return target_score


class AttentionRecorder:
    """记录模型attention层的激活值"""
    
    def __init__(self):
        self.activations = {}
        self.gradients = {}
        self.hooks = []
    
    def register_hooks(self, model, target_layers=None):
        """注册hook来记录激活值和梯度"""
        if target_layers is None:
            # 默认记录所有attention层
            target_layers = []
            for name, module in model.named_modules():
                if 'attn' in name.lower() or 'attention' in name.lower():
                    target_layers.append(name)
        
        for name, module in model.named_modules():
            if name in target_layers:
                # 记录前向激活
                hook_forward = module.register_forward_hook(
                    lambda module, input, output, name=name: self._save_activation(name, output)
                )
                # 记录反向梯度
                hook_backward = module.register_full_backward_hook(
                    lambda module, grad_input, grad_output, name=name: self._save_gradient(name, grad_output)
                )
                self.hooks.extend([hook_forward, hook_backward])
    
    def _save_activation(self, name: str, activation):
        """保存激活值"""
        if isinstance(activation, tuple):
            activation = activation[0]
        self.activations[name] = activation.detach()
    
    def _save_gradient(self, name: str, gradient):
        """保存梯度"""
        if isinstance(gradient, tuple):
            gradient = gradient[0]
        if gradient is not None:
            self.gradients[name] = gradient.detach()
    
    def clear_hooks(self):
        """清除所有hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}


class UViTGradCAM:
    """针对U-ViT模型的Grad-CAM实现"""
    
    def __init__(self, model, target_layers=None, tokenizer=None):
        self.model = model
        self.target_layers = target_layers or self._get_default_target_layers()
        self.recorder = AttentionRecorder()
        self.tokenizer = tokenizer
        
    def _get_default_target_layers(self):
        """自动识别U-ViT模型中的关键attention层"""
        target_layers = []
        for name, module in self.model.named_modules():
            # 查找attention相关的层
            if any(keyword in name.lower() for keyword in ['attn', 'attention', 'cross_attn']):
                # 优先选择深层的attention层
                if any(keyword in name for keyword in ['mid', 'up', 'down']):
                    target_layers.append(name)
        
        # 如果没有找到特定层，使用通用的attention层
        if not target_layers:
            for name, module in self.model.named_modules():
                if 'attn' in name.lower():
                    target_layers.append(name)
                    
        return target_layers[:3]  # 限制层数避免过多计算
        
    def reshape_transform_uvit(self, tensor, height=16, width=16):
        """将U-ViT的输出重塑为空间维度"""
        if len(tensor.shape) == 3:  # [batch, seq_len, dim]
            batch_size, seq_len, dim = tensor.shape
            # 假设去除class token后的序列长度对应空间维度
            spatial_tokens = seq_len - 1  # 减去class token
            if spatial_tokens == height * width:
                # 重塑为空间维度
                result = tensor[:, 1:, :].reshape(batch_size, height, width, dim)
                result = result.permute(0, 3, 1, 2)  # [batch, dim, height, width]
                return result
        return tensor
    
    def generate_cam(self, input_tensor, context=None, timestep=None, keyword=None, token_positions=None):
        """生成CAM热力图"""
        # 注册hooks
        self.recorder.register_hooks(self.model, self.target_layers)
        
        try:
            # 前向传播
            self.model.eval()
            
            # 根据输入参数调用模型
            if context is not None and timestep is not None:
                # 扩散模型的调用方式
                output = self.model(input_tensor, timestep, context=context)
            else:
                # 简化调用
                output = self.model(input_tensor)
            
            # 创建target函数
            target_fn = DiffusionTargetFunction(
                tokenizer=self.tokenizer,
                keyword=keyword,
                token_positions=token_positions
            )
            
            # 计算目标分数
            target_score = target_fn(output, context=context, timestep=timestep)
            
            # 反向传播
            self.model.zero_grad()
            target_score.backward(retain_graph=True)
            
            # 生成CAM
            cams = {}
            for layer_name in self.target_layers:
                if layer_name in self.recorder.activations and layer_name in self.recorder.gradients:
                    activation = self.recorder.activations[layer_name]
                    gradient = self.recorder.gradients[layer_name]
                    
                    # 处理不同形状的激活和梯度
                    if len(gradient.shape) == 4:  # [batch, channels, height, width]
                        weights = torch.mean(gradient, dim=(2, 3), keepdim=True)
                        cam = torch.sum(weights * activation, dim=1, keepdim=True)
                    elif len(gradient.shape) == 3:  # [batch, seq_len, dim]
                        weights = torch.mean(gradient, dim=1, keepdim=True)
                        cam = torch.sum(weights * activation, dim=2, keepdim=True)
                        # 重塑为空间维度
                        cam = self.reshape_transform_uvit(cam.unsqueeze(-1))
                    else:
                        # 其他情况的处理
                        weights = torch.mean(gradient, dim=-1, keepdim=True)
                        cam = torch.sum(weights * activation, dim=-1, keepdim=True)
                    
                    # 应用ReLU和归一化
                    cam = F.relu(cam)
                    cam = cam - cam.min()
                    cam = cam / (cam.max() + 1e-8)
                    
                    cams[layer_name] = cam
            
            return cams
            
        finally:
            self.recorder.clear_hooks()


class WordLevelCAMAnalyzer:
    """词汇级别的CAM分析器"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        
    def extract_keyword_positions(self, text: str, keyword: str) -> List[int]:
        """提取关键词在文本中的token位置"""
        # 简化的token位置提取
        words = text.lower().split()
        keyword_lower = keyword.lower()
        
        positions = []
        for i, word in enumerate(words):
            # 移除标点符号进行匹配
            clean_word = re.sub(r'[^\w]', '', word)
            if keyword_lower in clean_word or clean_word in keyword_lower:
                positions.append(i + 1)  # +1 因为通常有CLS token
                
        return positions
    
    def analyze_word_influence(self, model, input_tensor, context, timestep, text: str, 
                             target_words: List[str]) -> Dict[str, np.ndarray]:
        """分析多个词汇对图像生成的影响"""
        grad_cam = UViTGradCAM(model, tokenizer=self.tokenizer)
        word_influences = {}
        
        for word in target_words:
            # 获取词汇的token位置
            token_positions = self.extract_keyword_positions(text, word)
            
            if token_positions:
                # 生成该词汇的CAM
                cam_results = grad_cam.generate_cam(
                    input_tensor=input_tensor,
                    context=context,
                    timestep=timestep,
                    keyword=word,
                    token_positions=token_positions
                )
                
                # 合并多个层的CAM结果
                if cam_results:
                    combined_cam = self._combine_cams(cam_results)
                    word_influences[word] = combined_cam
                    
        return word_influences
    
    def _combine_cams(self, cam_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        """合并多个层的CAM结果"""
        if not cam_dict:
            return None
            
        # 将所有CAM结果平均
        cam_tensors = []
        for cam in cam_dict.values():
            if isinstance(cam, torch.Tensor):
                cam_np = cam.squeeze().cpu().numpy()
                if len(cam_np.shape) == 2:  # 确保是2D
                    cam_tensors.append(cam_np)
                    
        if cam_tensors:
            combined = np.mean(cam_tensors, axis=0)
            return combined
        return None


class CAMVisualizer:
    """CAM可视化工具"""
    
    def __init__(self):
        self.colormap = cv2.COLORMAP_JET
    
    def overlay_cam_on_image(self, image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """将CAM热力图叠加到原始图像上"""
        # 确保图像和CAM的尺寸匹配
        if image.shape[:2] != cam.shape[:2]:
            cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # 将CAM转换为彩色热力图
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam), self.colormap)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加图像
        overlayed = alpha * cam_colored + (1 - alpha) * image
        return overlayed.astype(np.uint8)
    
    def create_visualization(self, original_image: np.ndarray, cam_dict: Dict[str, np.ndarray], 
                           keyword: str, save_path: str = None) -> None:
        """创建完整的可视化结果"""
        num_cams = len(cam_dict)
        fig, axes = plt.subplots(1, num_cams + 1, figsize=(5 * (num_cams + 1), 5))
        
        if num_cams == 0:
            axes = [axes]
        
        # 显示原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 显示每个层的CAM结果
        for idx, (layer_name, cam) in enumerate(cam_dict.items()):
            if isinstance(cam, torch.Tensor):
                cam = cam.squeeze().cpu().numpy()
            
            overlayed = self.overlay_cam_on_image(original_image, cam)
            axes[idx + 1].imshow(overlayed)
            axes[idx + 1].set_title(f'CAM - {layer_name}\nKeyword: {keyword}')
            axes[idx + 1].axis('off')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Attention Intensity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CAM可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def create_multi_word_visualization(self, original_image: np.ndarray, 
                                      word_influences: Dict[str, np.ndarray], 
                                      save_path: str = None) -> None:
        """创建多词汇影响的可视化结果"""
        num_words = len(word_influences)
        if num_words == 0:
            print("没有词汇影响数据可供可视化")
            return
            
        # 创建子图布局
        cols = min(4, num_words + 1)  # 最多4列
        rows = (num_words + 1 + cols - 1) // cols  # 向上取整
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        
        # 确保axes是2D数组
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        # 显示原始图像
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 显示每个词汇的影响
        plot_idx = 1
        for word, cam in word_influences.items():
            if cam is not None:
                row = plot_idx // cols
                col = plot_idx % cols
                
                # 确保CAM是2D数组
                if len(cam.shape) > 2:
                    cam = cam.squeeze()
                    
                overlayed = self.overlay_cam_on_image(original_image, cam)
                axes[row, col].imshow(overlayed)
                axes[row, col].set_title(f'Word: "{word}"', fontsize=12, fontweight='bold')
                axes[row, col].axis('off')
                
                plot_idx += 1
        
        # 隐藏多余的子图
        for i in range(plot_idx, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
        cbar.set_label('Word Influence Intensity', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多词汇CAM可视化结果已保存到: {save_path}")
        
        plt.show()


def evaluate_with_cam(config, keyword=None):
    """带CAM可视化的评估函数"""
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config.seed)
    logging.info(f'Using device: {device}')

    config = ml_collections.FrozenConfigDict(config)
    utils.set_logger(log_level='info')

    dataset = get_dataset(**config.dataset)

    # 处理单个文本输入
    if config.input_text:
        prompt = config.input_text
        logging.info(f'使用提供的文本: {prompt}')
    elif config.input_file:
        with open(config.input_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        logging.info(f'从文件读取文本: {prompt}')
    else:
        raise ValueError("必须提供 input_text 或 input_file 参数")

    print(f"处理文本: {prompt}")
    
    # 初始化CLIP编码器
    clip = libs.clip.BertEmbedder4Vis(version='michiyasunaga/BioLinkBERT-base')
    clip.eval()
    clip.to(device)
            
    # 编码单个文本
    context, token_mask = clip.encode(prompt, keyword)
    
    # 加载神经网络模型
    nnet = utils.get_nnet(**config.nnet)
    nnet.to(device)
    logging.info(f'从 {config.nnet_path} 加载模型')
    nnet.load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if config.sample.scale == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    # 加载自动编码器
    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    _betas = stable_diffusion_beta_schedule()
    N = len(_betas)

    logging.info(config.sample)
    logging.info(f'N={N}')

    # 确保输出目录存在
    os.makedirs(config.sample.path, exist_ok=True)

    # 生成单个图像
    logging.info("开始生成图像...")
    
    # 创建随机噪声（批次大小为1）
    z_init = torch.randn(1, *config.z_shape, device=device)
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

    def model_fn(x, t_continuous):
        t = t_continuous * N
        return cfg_nnet(x, t, context=context)

    # 使用DPM求解器进行采样
    dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
    z = dpm_solver.sample(z_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)
    
    # 解码生成的潜在表示
    samples = dataset.unpreprocess(decode(z))
    
    # 保存生成的图像
    sample = samples[0]
    output_filename = config.get('output_filename', 'generated_image.png')
    output_path = os.path.join(config.output_path, output_filename)
    save_image(sample, output_path)
    
    logging.info(f"图像已保存到: {output_path}")
    print(f"生成完成！图像保存在: {output_path}")
    
    # CAM可视化（如果提供了关键词）
    if keyword:
        print(f"\n开始生成关键词 '{keyword}' 的CAM可视化...")
        
        # 转换图像为numpy格式
        original_image = sample.permute(1, 2, 0).cpu().numpy()
        original_image = (original_image * 255).astype(np.uint8)
        
        # 初始化词汇级别分析器
        word_analyzer = WordLevelCAMAnalyzer(tokenizer=None)
        
        # 分析单个关键词或多个关键词
        if ',' in keyword:
            # 多个关键词
            target_words = [word.strip() for word in keyword.split(',')]
        else:
            # 单个关键词
            target_words = [keyword]
        
        print(f"分析词汇: {target_words}")
        
        # 生成CAM（使用最后一步的潜在表示和对应的时间步）
        with torch.enable_grad():
            z.requires_grad_(True)
            
            # 创建时间步张量（使用接近结束的时间步）
            timestep = torch.tensor([1.0], device=device)  # 接近生成结束
            
            # 分析词汇影响
            word_influences = word_analyzer.analyze_word_influence(
                model=nnet,
                input_tensor=z,
                context=context,
                timestep=timestep,
                text=prompt,
                target_words=target_words
            )
        
        if word_influences:
            # 创建可视化
            visualizer = CAMVisualizer()
            
            if len(target_words) == 1:
                # 单词汇可视化
                word = target_words[0]
                if word in word_influences and word_influences[word] is not None:
                    cam_dict = {f'word_{word}': word_influences[word]}
                    cam_output_path = os.path.join(config.output_path, f'cam_{word}_{output_filename}')
                    visualizer.create_visualization(original_image, cam_dict, word, cam_output_path)
                else:
                    print(f"未能为词汇 '{word}' 生成CAM结果")
            else:
                # 多词汇可视化
                cam_output_path = os.path.join(config.output_path, f'cam_multi_words_{output_filename}')
                visualizer.create_multi_word_visualization(original_image, word_influences, cam_output_path)
        else:
            print("未能生成CAM结果")


from absl import flags
from absl import app
from ml_collections import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "训练配置文件", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "要评估的神经网络模型路径")
flags.DEFINE_string("output_path", None, "输出图像的路径")
flags.DEFINE_string("input_text", None, "输入的文本提示（直接指定文本）")
flags.DEFINE_string("input_file", None, "输入文本文件的路径（从文件读取文本）")
flags.DEFINE_string("output_filename", "generated_image.png", "输出图像的文件名")
flags.DEFINE_string("keyword", None, "要可视化的关键词")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.input_text = FLAGS.input_text
    config.input_file = FLAGS.input_file
    config.output_filename = FLAGS.output_filename
    
    # 验证输入参数
    if not FLAGS.input_text and not FLAGS.input_file:
        raise ValueError("必须提供 --input_text 或 --input_file 参数之一")
    
    if FLAGS.input_text and FLAGS.input_file:
        logging.warning("同时提供了 input_text 和 input_file，将优先使用 input_text")
    
    evaluate_with_cam(config, keyword=FLAGS.keyword)


if __name__ == "__main__":
    app.run(main)