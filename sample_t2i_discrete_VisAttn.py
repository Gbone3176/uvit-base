import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
from datasets import get_dataset
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image
from tools.Visualization.Vision.uvit_attention_vis import UViTAttentionMonitor, UViTAttnStore
import numpy as np
import os
os.CUDA_VISIBLE_DEVICES = '0,1'


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info')
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)

    with open(config.input_path, 'r') as f:
        prompts = f.read().strip().split('\n')


    clip = libs.clip.BioMedClipEmbedder()
    clip.eval()
    clip.to(device)

    tokens = clip.token(prompts)
    print(tokens)
    contexts = clip.encode(prompts)
    

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if config.sample.scale == 0:
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
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
    logging.info(f'mixed_precision={config.mixed_precision}')
    logging.info(f'N={N}')

    z_init = torch.randn(contexts.size(0), *config.z_shape, device=device)
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

    def model_fn(x, t_continuous):
        t = t_continuous * N
        return cfg_nnet(x, t, context=contexts)

    # 创建注意力监控器
    attn_monitor = None
    if config.get('visualize_attention', False):
        # 确保获取到的模型是正确的
        model = accelerator.unwrap_model(nnet)
        print(f"模型类型: {type(model)}")
        
        attn_monitor = UViTAttentionMonitor(model)
        
        # 注册钩子 - 为U-ViT指定块标识
        # 确保block_indices是列表
        block_indices = list(config.get('attn_vis_blocks', ["in_0", "mid", "out_0"]))
        attn_types = list(config.get('attn_vis_types', ['self']))
        
        print(f"注册块: {block_indices}, 类型: {attn_types}")
        attn_stores = attn_monitor.register_hooks(block_indices, attn_types)
        print(f"已注册 {len(attn_stores)} 个注意力存储器")

        # 创建用于保存注意力图的目录
        attn_dir = os.path.join(config.output_path, "attention_maps")
        os.makedirs(attn_dir, exist_ok=True)
        
        # 提取要关注的单词索引
        word_indices = []
        blind_words = ["there","the", "and", "with", "from", "for", "this", "that", "a", "some", "an", "is", "are", "to", "of", "on", "in", "at", "as", "by", "it", "its", "which", "who", "what", "where", "when"]
        label = ["atelectasis", "cardiomegaly", "effusion", "infiltration", "mass", "nodule", "pneumonia", "pneumothorax", "consolidation", "edema", "emphysema", "fibrosis", "pleural_thickening", "hernia"]
        for item in tokens:
            found = False
            for i, token in enumerate(item):
                if len(token) > 3 and token in label and token not in blind_words:
                    word_indices.append(i)
                    found = True
                    break
            if not found and item:
                word_indices.append(0)  # 如果没找到合适的词，使用第一个词
        
        # 确保词索引列表至少与批次大小一样长
        if len(word_indices) < len(tokens):
            word_indices.extend([0] * (len(tokens) - len(word_indices)))

        # 只关注[CLS]
        word_indices = [0] * len(tokens)
        
        # 记录每个prompt关注的词
        with open(os.path.join(attn_dir, "word_indices.txt"), "w") as f:
            for i, (token, idx) in enumerate(zip(tokens, word_indices)):
                prompt = " ".join(token)
                word = token[idx] if idx < len(token) else ""
                f.write(f"Prompt {i}: '{prompt}' -> Word {idx}: '{word}'\n")
        
    if config.get('visualize_attention', False) and config.get('visualize_steps', False):
        # 自定义采样以记录每个时间步的注意力
        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)

        def custom_sample():
            # 重置注意力存储器
            attn_monitor.reset_stores()
            
            # 初始化
            x = z_init
            steps = config.sample.sample_steps
            eps = 1. / N
            T = 1.0
            
            # 创建时间步目录
            timestep_dir = os.path.join(config.output_path, "timesteps")
            os.makedirs(timestep_dir, exist_ok=True)
            
            # 获取时间步
            time_steps = dpm_solver.get_time_steps(skip_type="time_uniform", t_T=T, t_0=eps, N=steps, device=device)
            
            # 手动执行DPM-Solver的单步循环
            for i in range(len(time_steps)-1):
                # 确定当前步骤的阶数(通常第一步是1阶，其余是2阶)
                order = 1 if i == 0 else 2
                
                # 获取当前时间步
                s = time_steps[i].expand(x.shape[0])
                t = time_steps[i+1].expand(x.shape[0])
                
                # 执行单步
                x = dpm_solver.dpm_solver_update(x, s, t, order)
                
                # 保存中间结果
                if accelerator.is_main_process and i % config.get('save_step_interval', 10) == 0:
                    # 解码当前状态
                    step_samples = dataset.unpreprocess(decode(x))
                    
                    # 保存图像
                    step_idx = int(s.item() * N)
                    for j, (sample, prompt) in enumerate(zip(step_samples, prompts)):
                        save_image(sample, os.path.join(timestep_dir, f"step_{step_idx}_{j}_{prompt[:20]}.png"))
                    
                    # 可视化当前注意力
                    step_attn_dir = os.path.join(timestep_dir, f"attn_step_{step_idx}")
                    os.makedirs(step_attn_dir, exist_ok=True)
                    attn_monitor.visualize_all(step_attn_dir, word_indices)
                    
                    # 重置注意力存储器，准备下一步
                    attn_monitor.reset_stores()
            
            return x
        
        # 使用自定义采样
        z = custom_sample()
    else:
        # 使用原始采样

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        # z = dpm_solver.sample(z_init, steps=1, eps=1. / N, T=1.)
        z = dpm_solver.sample(z_init, steps=config.sample.sample_steps, eps=1. / N, T=1.)

    samples = dataset.unpreprocess(decode(z))

    # 可视化注意力（如果启用）
    if attn_monitor is not None and accelerator.is_main_process:

        # 可视化所有注意力图
        attn_monitor.visualize_all(attn_dir, word_indices)
        
        # 移除钩子，避免内存泄漏
        attn_monitor.remove_hooks()

    os.makedirs(config.output_path, exist_ok=True)
    for sample, prompt in zip(samples, prompts):
        save_image(sample, os.path.join(config.output_path, f"Vis_attention.png"))


from absl import flags
from absl import app
from ml_collections import config_flags



FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output images.")
flags.DEFINE_string("input_path", "/storage/U-ViT/tools/Visualization/Vision/prompts.txt", "The path to input texts.")
# Visualization options
flags.DEFINE_boolean("visualize_attention", False, "是否可视化注意力图")
flags.DEFINE_list("attn_vis_blocks", "in_0,mid,out_0", "要可视化的块标识，例如 in_0,mid,out_0")
flags.DEFINE_list("attn_vis_types", "self,cross", "要可视化的注意力类型（self或cross）")
flags.DEFINE_boolean("visualize_steps", False, "是否可视化每个时间步的注意力")
flags.DEFINE_integer("save_step_interval", 10, "保存中间结果的步骤间隔")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.input_path = FLAGS.input_path

    # 添加注意力可视化配置
    config.visualize_attention = FLAGS.visualize_attention
    if FLAGS.visualize_attention:
        config.attn_vis_blocks = FLAGS.attn_vis_blocks
        config.attn_vis_types = FLAGS.attn_vis_types
        config.visualize_steps = FLAGS.visualize_steps
        config.save_step_interval = FLAGS.save_step_interval

    evaluate(config)


if __name__ == "__main__":
    app.run(main)