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


# class KeywordExtractor:
#     """提取文本中关键词的位置信息"""
    
#     def __init__(self, tokenizer=None):
#         self.tokenizer = tokenizer
    
#     def extract_keyword_positions(self, text: str, keyword: str) -> List[Tuple[int, int]]:
#         """提取关键词在文本中的位置"""
#         positions = []
#         text_lower = text.lower()
#         keyword_lower = keyword.lower()
        
#         start = 0
#         while True:
#             pos = text_lower.find(keyword_lower, start)
#             if pos == -1:
#                 break
#             positions.append((pos, pos + len(keyword)))
#             start = pos + 1
        
#         return positions
    
#     def get_token_positions(self, text: str, keyword: str) -> List[int]:
#         """获取关键词对应的token位置（简化版本）"""
#         words = text.lower().split()
#         keyword_lower = keyword.lower()
        
#         token_positions = []
#         for i, word in enumerate(words):
#             if keyword_lower in word:
#                 token_positions.append(i)
        
#         return token_positions


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
                hook_backward = module.register_backward_hook(
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
    
    def __init__(self, model, target_layers=None):
        self.model = model
        self.target_layers = target_layers or ['mid_block']
        self.recorder = AttentionRecorder()
        
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
    
    def generate_cam(self, input_tensor, target_keyword_positions=None):
        """生成CAM热力图"""
        # 注册hooks
        self.recorder.register_hooks(self.model, self.target_layers)
        
        try:
            # 前向传播
            self.model.eval()
            output = self.model(input_tensor)
            
            # 计算目标分数（这里简化为输出的均值）
            if target_keyword_positions:
                # 基于关键词位置计算目标分数
                target_score = output.mean()
            else:
                target_score = output.mean()
            
            # 反向传播
            self.model.zero_grad()
            target_score.backward(retain_graph=True)
            
            # 生成CAM
            cams = {}
            for layer_name in self.target_layers:
                if layer_name in self.recorder.activations and layer_name in self.recorder.gradients:
                    activation = self.recorder.activations[layer_name]
                    gradient = self.recorder.gradients[layer_name]
                    
                    # 计算权重
                    weights = torch.mean(gradient, dim=(2, 3), keepdim=True) if len(gradient.shape) == 4 else torch.mean(gradient, dim=1, keepdim=True)
                    
                    # 生成CAM
                    cam = torch.sum(weights * activation, dim=1, keepdim=True)
                    cam = F.relu(cam)
                    
                    # 归一化
                    cam = cam - cam.min()
                    cam = cam / (cam.max() + 1e-8)
                    
                    cams[layer_name] = cam
            
            return cams
            
        finally:
            self.recorder.clear_hooks()


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
        
        # 初始化CAM
        grad_cam = UViTGradCAM(nnet, target_layers=['mid_block'])
        
        # 生成CAM（使用最后一步的潜在表示）
        with torch.enable_grad():
            z.requires_grad_(True)
            cam_results = grad_cam.generate_cam(z, token_positions if 'token_positions' in locals() else None)
        
        if cam_results:
            # 转换图像为numpy格式
            original_image = sample.permute(1, 2, 0).cpu().numpy()
            original_image = (original_image * 255).astype(np.uint8)
            
            # 创建可视化
            visualizer = CAMVisualizer()
            cam_output_path = os.path.join(config.output_path, f'cam_{keyword}_{output_filename}')
            visualizer.create_visualization(original_image, cam_results, keyword, cam_output_path)
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