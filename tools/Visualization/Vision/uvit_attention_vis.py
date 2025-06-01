import einops
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any
from torchvision.utils import save_image
import os
import math
import types
from einops import rearrange
from PIL import Image
import torch.nn.functional as F

class UViTAttnStore:
    """为U-ViT模型存储注意力权重的类"""
    
    def __init__(self, layer_name=None):
        self.layer_name = layer_name
        self.attns = []
        self.is_self_attn = "self" in layer_name if layer_name else True
        # 假设U-ViT默认有8个头
        self.num_heads = 8  
    
    def __call__(self, x):
        """
        替代U-ViT模型中的注意力计算，同时存储注意力权重
        x: 输入张量, 可能是 [B, L, D] 或 [B, H, L, D]
        """
        # 检查输入维度并相应处理
        if len(x.shape) == 3:
            # 标准3D输入 [B, L, D]
            B, L, C = x.shape
            
            # 计算qkv投影，使用之前设置的属性
            qkv = self.qkv(x)  # 调用原始模块的qkv函数
            
            # 根据U-ViT模型的实现方式处理注意力计算
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # flash attention模式
                qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
                q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
                
                # 计算注意力权重
                scale = math.sqrt(q.shape[-1])
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                
                # 存储注意力权重
                self.attns.append(attn.detach().clone())
                
                # 继续原始计算
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                x = einops.rearrange(x, 'B H L D -> B L (H D)')
            else:
                # 标准注意力计算
                qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
                q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
                
                # 计算注意力权重
                scale = math.sqrt(q.shape[-1])
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                
                # 存储注意力权重
                self.attns.append(attn.detach().clone())
                
                # 继续原始计算
                x = (attn @ v).transpose(1, 2).reshape(B, L, C)
            
            # 使用原始的投影层完成计算
            x = self.to_out(x)
            
        elif len(x.shape) == 4:
            # 4D输入 [B, H, L, D]
            B, H, L, D = x.shape
            
            # 检查并适应头的数量
            if H != self.num_heads:
                print(f"警告: 输入的头数 {H} 与预期的 {self.num_heads} 不匹配")
                self.num_heads = H
            
            # 计算qkv投影，使用之前设置的属性
            # 首先将输入调整为标准格式以便处理
            x_reshaped = einops.rearrange(x, 'B H L D -> B L (H D)')
            qkv = self.qkv(x_reshaped)
            
            # 处理注意力计算
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                # flash attention模式
                qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
                q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
                
                # 计算注意力权重
                scale = math.sqrt(q.shape[-1])
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                
                # 存储注意力权重
                self.attns.append(attn.detach().clone())
                
                # 继续原始计算
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                x = einops.rearrange(x, 'B H L D -> B L (H D)')
            else:
                # 标准注意力计算
                qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
                q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
                
                # 计算注意力权重
                scale = math.sqrt(q.shape[-1])
                attn = (q @ k.transpose(-2, -1)) * scale
                attn = attn.softmax(dim=-1)
                
                # 存储注意力权重
                self.attns.append(attn.detach().clone())
                
                # 继续原始计算
                x = (attn @ v).transpose(1, 2).reshape(B, L, H*D)
            
            # 使用原始的投影层完成计算
            x = self.to_out(x)
            
            # 将输出重新格式化为输入格式
            x = einops.rearrange(x, 'B L (H D) -> B H L D', H=H)
        else:
            raise ValueError(f"意外的输入形状: {x.shape}，期望形状为 [B, L, D] 或 [B, H, L, D]")
        
        return x
    
    def reset(self):
        """重置存储的注意力权重"""
        self.attns = []
    
    def visualize_attention(self, save_path="uvit_attn_maps.png", text_token_num=256, img_hw=16, word_indices=None):
        """
        可视化UViT自注意力，支持文本token与图像token的关注关系
        Args:
            save_path: 保存路径
            text_token_num: 文本token数量（如77）
            img_hw: 图像patch的高宽（如32x32）
        """
        import matplotlib.pyplot as plt

        if not self.attns:
            print("没有可视化的注意力图")
            return

        # 取最后一层的注意力
        attn = self.attns[-1]  # [B, H, N, N]
        attn = attn.mean(dim=1)  # 多头平均 [B, N, N]
        attn = attn[0]           # 只取第一个batch [N, N]

        save_image(attn, save_path.replace(".png", ".jpg"), normalize=True)

        extra = 1 + text_token_num
        # 假设token顺序为 [1 + 文本token + 图像token]
        if word_indices is not None and isinstance(word_indices, int):
            attn_vec = attn[extra + word_indices, extra:extra + img_hw * img_hw]  # (256,)
        else:
            # 默认可视化第0个patch
            attn_vec = attn[extra, extra:extra + img_hw * img_hw]  # (256,)

        # reshape为16x16
        attn_map = attn_vec.reshape(img_hw, img_hw)  # (16, 16)
        # 插值到256x256
        attn_map_up = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
        # 用save_image保存为灰度图
        save_image(attn_map_up, save_path, normalize=True)
        print(f"注意力图已保存到: {save_path}")




class UViTAttentionMonitor:
    """监控U-ViT模型注意力的类"""
    
    def __init__(self, model):
        self.model = model
        self.attn_stores = {}
        self.original_methods = {}
        
        # 检查模型类型并设置属性
        if hasattr(model, 'in_blocks') and hasattr(model, 'out_blocks'):
            self.model_type = 'uvit'
            print(f"检测到U-ViT模型: in_blocks: {len(model.in_blocks)}, out_blocks: {len(model.out_blocks)}")
        elif hasattr(model, 'blocks'):
            self.model_type = 'standard'
            print(f"检测到标准Transformer模型: blocks: {len(model.blocks)}")
        else:
            print("警告: 未知的模型结构，可能无法正确注册钩子")
            print(f"模型属性: {dir(model)}")
            self.model_type = 'unknown'
        
    def register_hooks(self, block_indices=None, attn_types=None):
        """
        为模型的注意力块注册钩子
        
        Args:
            block_indices: 要监控的块索引，例如 [0, 4, 8] 或 ["in_0", "mid", "out_0"]
            attn_types: 注意力类型，可以是 'self'、'cross' 或 ['self', 'cross']
        """
        if attn_types is None:
            attn_types = ['self']
            
        # 确保block_indices和attn_types是列表
        if block_indices is None:
            # 使用默认值
            pass
        elif isinstance(block_indices, (list, tuple)):
            # 如果是列表或元组，转换为列表
            block_indices = list(block_indices)
        else:
            # 如果是单个值，放入列表
            block_indices = [block_indices]
            
        if isinstance(attn_types, (list, tuple)):
            attn_types = list(attn_types)
        else:
            attn_types = [attn_types]
            
        # 重置存储器
        self.remove_hooks()
        
        # 根据模型类型处理不同的注册方式
        if self.model_type == 'uvit':
            # 访问模型的块
            if hasattr(self.model, 'in_blocks'):
                in_blocks = self.model.in_blocks
            else:
                print("警告: 模型没有in_blocks属性")
                in_blocks = []
                
            if hasattr(self.model, 'out_blocks'):
                out_blocks = self.model.out_blocks
            else:
                print("警告: 模型没有out_blocks属性")
                out_blocks = []
                
            # 如果未指定块索引，则选择一些默认块
            if block_indices is None:
                # 构建默认索引列表
                block_indices = []
                
                # 添加第一个和最后一个in_block
                if in_blocks:
                    block_indices.append("in_0")
                    if len(in_blocks) > 1:
                        block_indices.append(f"in_{len(in_blocks)-1}")
                
                # 添加mid_block
                if hasattr(self.model, 'mid_block'):
                    block_indices.append("mid")
                    
                # 添加第一个和最后一个out_block
                if out_blocks:
                    block_indices.append("out_0")
                    if len(out_blocks) > 1:
                        block_indices.append(f"out_{len(out_blocks)-1}")
            
            print(f"将要注册的块: {block_indices}")
            
            # 注册钩子到指定的块
            for block_idx in block_indices:
                # 处理不同类型的块标识
                if isinstance(block_idx, str):
                    if block_idx.startswith("in_"):
                        # in_blocks
                        try:
                            idx = int(block_idx.split("_")[1])
                            if in_blocks and 0 <= idx < len(in_blocks):
                                block = in_blocks[idx]
                                self._register_attention_hook(block, 'attn', f"in_block{idx}_self")
                            else:
                                print(f"警告: in_blocks索引 {idx} 超出范围，模型只有 {len(in_blocks) if in_blocks else 0} 个in_blocks")
                        except (IndexError, ValueError) as e:
                            print(f"警告: 无法解析 in_blocks 索引 '{block_idx}': {e}")
                    elif block_idx == "mid":
                        # mid_block
                        if hasattr(self.model, 'mid_block'):
                            block = self.model.mid_block
                            self._register_attention_hook(block, 'attn', f"mid_block_self")
                        else:
                            print("警告: 模型没有 mid_block 属性")
                    elif block_idx.startswith("out_"):
                        # out_blocks
                        try:
                            idx = int(block_idx.split("_")[1])
                            if out_blocks and 0 <= idx < len(out_blocks):
                                block = out_blocks[idx]
                                self._register_attention_hook(block, 'attn', f"out_block{idx}_self")
                            else:
                                print(f"警告: out_blocks索引 {idx} 超出范围，模型只有 {len(out_blocks) if out_blocks else 0} 个out_blocks")
                        except (IndexError, ValueError) as e:
                            print(f"警告: 无法解析 out_blocks 索引 '{block_idx}': {e}")
                    else:
                        print(f"警告: 未知的块标识 '{block_idx}'")
                else:
                    # 如果提供的是数字索引，尝试转换为整数并使用
                    try:
                        numeric_idx = int(block_idx)
                        if in_blocks and 0 <= numeric_idx < len(in_blocks):
                            block = in_blocks[numeric_idx]
                            self._register_attention_hook(block, 'attn', f"in_block{numeric_idx}_self")
                        else:
                            print(f"警告: 块索引 {numeric_idx} 超出范围，模型只有 {len(in_blocks) if in_blocks else 0} 个in_blocks")
                    except (TypeError, ValueError) as e:
                        print(f"警告: 无法将 '{block_idx}' 转换为有效的块索引: {e}")
        elif self.model_type == 'standard':
            # 标准Transformer结构
            blocks = self.model.blocks
            
            if block_indices is None:
                # 默认选择几个块
                block_indices = [0, len(blocks) // 2, len(blocks) - 1]
            
            # 注册钩子
            for block_idx in block_indices:
                if block_idx >= len(blocks):
                    print(f"警告: 块索引 {block_idx} 超出范围，模型只有 {len(blocks)} 个块")
                    continue
                    
                block = blocks[block_idx]
                
                # 根据注意力类型注册钩子
                for attn_type in attn_types:
                    if attn_type == 'self' and hasattr(block, 'attn'):
                        self._register_attention_hook(block, 'attn', f"block{block_idx}_self")
                    elif attn_type == 'cross' and hasattr(block, 'cross_attn'):
                        self._register_attention_hook(block, 'cross_attn', f"block{block_idx}_cross")
        
        return self.attn_stores
        
    
    def _register_attention_hook(self, module, attn_name, store_name):
        """为特定注意力模块注册钩子"""
        if not hasattr(module, attn_name):
            print(f"警告: 模块没有 {attn_name} 属性")
            return
            
        attn_module = getattr(module, attn_name)
        
        # 创建注意力存储器
        attn_store = UViTAttnStore(store_name)
        self.attn_stores[store_name] = attn_store
        
        # 保存原始方法
        orig_forward = attn_module.forward
        self.original_methods[store_name] = orig_forward
        
        # 设置必要的属性
        if hasattr(attn_module, 'qkv'):
            # 如果attention模块使用组合的qkv计算
            setattr(attn_store, 'qkv', attn_module.qkv)
        else:
            # 单独的q,k,v计算
            for attr_name in ['to_q', 'to_k', 'to_v']:
                if hasattr(attn_module, attr_name):
                    setattr(attn_store, attr_name, getattr(attn_module, attr_name))
        
        # 设置输出投影
        if hasattr(attn_module, 'to_out'):
            setattr(attn_store, 'to_out', attn_module.to_out)
        elif hasattr(attn_module, 'proj'):
            setattr(attn_store, 'to_out', attn_module.proj)
        
        # 从原始注意力模块设置num_heads
        if hasattr(attn_module, 'num_heads'):
            attn_store.num_heads = attn_module.num_heads
        
        # 创建新的forward函数，适应U-ViT的注意力实现
        def new_forward(self, x):
            B, L, C = x.shape
            
            # 调用注意力存储器的__call__方法
            # 注意: 在U-ViT中，注意力层只接收一个参数x
            return attn_store(x)
        
        # 替换forward函数
        attn_module.forward = types.MethodType(new_forward, attn_module)
    
    def _extract_qkv(self, module, x, attr_name):
        """从QKV联合计算中提取单独的q、k、v"""
        # 根据uvit_t2i.py中的Attention类实现
        qkv = module.qkv(x)
        B, L, C = x.shape
        
        # 根据注意力模式选择处理方式
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # flash attention模式
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=module.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            
            if attr_name == 'to_q':
                return q
            elif attr_name == 'to_k':
                return k
            elif attr_name == 'to_v':
                return v
            else:
                return module.proj(torch.einops.rearrange(
                    q @ v, 'B H L D -> B L (H D)'
                ))
        else:
            # math模式
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=module.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            
            if attr_name == 'to_q':
                return q
            elif attr_name == 'to_k':
                return k
            elif attr_name == 'to_v':
                return v
            else:
                return module.proj(torch.einops.rearrange(
                    q @ v, 'B H L D -> B L (H D)'
                ))
    
    def remove_hooks(self):
        """移除所有钩子，恢复原始forward函数"""
        for name, orig_method in self.original_methods.items():
            # 解析名称以找到正确的模块
            if name.startswith("in_block"):
                block_idx = int(name.split("block")[1].split("_")[0])
                block = self.model.in_blocks[block_idx]
                module_name = 'attn'
            elif name.startswith("mid_block"):
                block = self.model.mid_block
                module_name = 'attn'
            elif name.startswith("out_block"):
                block_idx = int(name.split("block")[1].split("_")[0])
                block = self.model.out_blocks[block_idx]
                module_name = 'attn'
            elif name.startswith("block"):
                block_idx = int(name.split("block")[1].split("_")[0])
                block = self.model.blocks[block_idx]
                module_name = 'attn' if 'self' in name else 'cross_attn'
            else:
                print(f"警告: 未知的模块名称 {name}")
                continue
            
            if hasattr(block, module_name):
                module = getattr(block, module_name)
                module.forward = orig_method
        
        self.original_methods = {}
        self.attn_stores = {}
    
    def reset_stores(self):
        """重置所有注意力存储器"""
        for store in self.attn_stores.values():
            store.reset()
    
    def visualize_all(self, save_dir="attention_maps", word_indices=None):
        """可视化所有注意力图"""
        os.makedirs(save_dir, exist_ok=True)
        
        for name, store in self.attn_stores.items():
            save_path = os.path.join(save_dir, f"{name}.png")
            store.visualize_attention(save_path, word_indices= word_indices)