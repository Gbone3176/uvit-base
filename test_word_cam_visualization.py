#!/usr/bin/env python3
"""
测试词汇级别的CAM可视化功能

使用示例:
python test_word_cam_visualization.py \
    --config configs/chestXray14_uvit_small_t2i_clip.py \
    --nnet_path /path/to/model.pth \
    --output_path ./outputs \
    --input_text "chest x-ray showing pneumonia in the left lung" \
    --keyword "pneumonia,lung"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sample_t2i_discrete_gradCam_enhanced import evaluate_with_cam
from absl import flags
from absl import app
from ml_collections import config_flags
import ml_collections

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "训练配置文件", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "要评估的神经网络模型路径")
flags.DEFINE_string("output_path", "./outputs", "输出图像的路径")
flags.DEFINE_string("input_text", "chest x-ray showing pneumonia in the left lung", "输入的文本提示")
flags.DEFINE_string("keyword", "pneumonia,lung", "要可视化的关键词（用逗号分隔多个词汇）")
flags.DEFINE_string("output_filename", "test_cam_visualization.png", "输出图像的文件名")

def test_single_word():
    """测试单个词汇的可视化"""
    print("=== 测试单个词汇可视化 ===")
    
    # 创建测试配置
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.benchmark = False
    
    # 数据集配置
    config.dataset = ml_collections.ConfigDict()
    config.dataset.name = 'chestXray14'
    config.dataset.path = '/path/to/dataset'  # 这里需要根据实际情况修改
    config.dataset.cfg = True
    config.dataset.p_uncond = 0.1
    
    # 网络配置
    config.nnet = ml_collections.ConfigDict()
    config.nnet.name = 'uvit_t2i'
    config.nnet.img_size = 256
    config.nnet.in_chans = 4
    config.nnet.patch_size = 8
    config.nnet.embed_dim = 512
    config.nnet.depth = 12
    config.nnet.num_heads = 8
    config.nnet.mlp_ratio = 4
    config.nnet.qkv_bias = False
    config.nnet.mlp_time_embed = False
    config.nnet.num_classes = -1
    config.nnet.use_checkpoint = False
    config.nnet.conv = True
    config.nnet.skip = True
    
    # 自动编码器配置
    config.autoencoder = ml_collections.ConfigDict()
    config.autoencoder.pretrained_path = '/path/to/autoencoder.pth'
    
    # 采样配置
    config.sample = ml_collections.ConfigDict()
    config.sample.sample_steps = 50
    config.sample.scale = 7.5
    config.sample.path = './outputs'
    
    # 潜在空间形状
    config.z_shape = (4, 32, 32)
    
    # 设置测试参数
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.input_text = "chest x-ray showing pneumonia"
    config.output_filename = "single_word_test.png"
    
    try:
        evaluate_with_cam(config, keyword="pneumonia")
        print("✓ 单个词汇可视化测试成功")
        return True
    except Exception as e:
        print(f"✗ 单个词汇可视化测试失败: {e}")
        return False

def test_multiple_words():
    """测试多个词汇的可视化"""
    print("\n=== 测试多个词汇可视化 ===")
    
    # 使用与单词测试相同的配置
    config = ml_collections.ConfigDict()
    config.seed = 42
    config.benchmark = False
    
    # 基本配置（简化版本）
    config.dataset = ml_collections.ConfigDict()
    config.dataset.name = 'chestXray14'
    config.dataset.cfg = True
    config.dataset.p_uncond = 0.1
    
    config.nnet = ml_collections.ConfigDict()
    config.nnet.name = 'uvit_t2i'
    
    config.autoencoder = ml_collections.ConfigDict()
    
    config.sample = ml_collections.ConfigDict()
    config.sample.sample_steps = 50
    config.sample.scale = 7.5
    config.sample.path = './outputs'
    
    config.z_shape = (4, 32, 32)
    
    # 设置测试参数
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.input_text = "chest x-ray showing pneumonia in the left lung with pleural effusion"
    config.output_filename = "multi_word_test.png"
    
    try:
        evaluate_with_cam(config, keyword="pneumonia,lung,pleural")
        print("✓ 多个词汇可视化测试成功")
        return True
    except Exception as e:
        print(f"✗ 多个词汇可视化测试失败: {e}")
        return False

def test_cam_components():
    """测试CAM组件的基本功能"""
    print("\n=== 测试CAM组件 ===")
    
    try:
        # 测试导入
        from sample_t2i_discrete_gradCam_enhanced import (
            DiffusionTargetFunction,
            UViTGradCAM,
            WordLevelCAMAnalyzer,
            CAMVisualizer
        )
        print("✓ 所有组件导入成功")
        
        # 测试基本初始化
        target_fn = DiffusionTargetFunction(keyword="test")
        print("✓ DiffusionTargetFunction 初始化成功")
        
        analyzer = WordLevelCAMAnalyzer()
        print("✓ WordLevelCAMAnalyzer 初始化成功")
        
        visualizer = CAMVisualizer()
        print("✓ CAMVisualizer 初始化成功")
        
        # 测试关键词位置提取
        positions = analyzer.extract_keyword_positions(
            "chest x-ray showing pneumonia in lung", 
            "pneumonia"
        )
        print(f"✓ 关键词位置提取成功: {positions}")
        
        return True
        
    except Exception as e:
        print(f"✗ CAM组件测试失败: {e}")
        return False

def main(argv):
    """主测试函数"""
    print("开始测试词汇级别的CAM可视化功能...")
    print(f"配置文件: {FLAGS.config}")
    print(f"模型路径: {FLAGS.nnet_path}")
    print(f"输出路径: {FLAGS.output_path}")
    print(f"测试文本: {FLAGS.input_text}")
    print(f"关键词: {FLAGS.keyword}")
    
    # 确保输出目录存在
    os.makedirs(FLAGS.output_path, exist_ok=True)
    
    # 运行测试
    results = []
    
    # 1. 测试组件
    results.append(test_cam_components())
    
    # 2. 如果提供了模型路径，运行完整测试
    if FLAGS.nnet_path and os.path.exists(FLAGS.nnet_path):
        # 使用实际配置运行测试
        config = FLAGS.config
        config.nnet_path = FLAGS.nnet_path
        config.output_path = FLAGS.output_path
        config.input_text = FLAGS.input_text
        config.output_filename = FLAGS.output_filename
        
        try:
            evaluate_with_cam(config, keyword=FLAGS.keyword)
            print("✓ 完整可视化流程测试成功")
            results.append(True)
        except Exception as e:
            print(f"✗ 完整可视化流程测试失败: {e}")
            results.append(False)
    else:
        print("⚠ 未提供有效的模型路径，跳过完整流程测试")
    
    # 总结结果
    print("\n=== 测试结果总结 ===")
    passed = sum(results)
    total = len(results)
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！词汇级别CAM可视化功能已就绪")
    else:
        print("⚠ 部分测试失败，请检查错误信息")

if __name__ == "__main__":
    app.run(main)