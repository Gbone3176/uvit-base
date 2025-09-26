import ml_collections
import torch
from torch import multiprocessing as mp
import torch.nn.functional as F
import accelerate
import utils
from utils import _overlay_cam_rgb
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
import os
from contextlib import contextmanager


torch.manual_seed(42)

#Attn hook & Grad hook
activations = {}
gradients = {}

def forward_hook(module, input, output):
    # 某些注意力层可能返回结构体，这里统一拿张量
    activations['value'] = output if torch.is_tensor(output) else output[0]

def backward_hook(module, grad_input, grad_output):
    # grad_output[0] 是 dS/d(output)
    gradients['value'] = grad_output[0]


def split_tokens(A, x, num_clip_token=256, patch_size=2, extras_extra=0):
    """
    A: [B, L, D] 或 [B, D, L]
    num_clip_token: 你的 config.nnet.num_clip_token (默认 256)
    x: 当前步的 latent [B, 4, H, W]（用来推回 H_p,W_p）
    extras_extra: 若你额外加了 class token，这里填 class token 数；默认 0
    返回: dict 包含 time/text/img 三段，以及 img 段的 H_p,W_p
    """
    # 统一成 [B, L, D]
    if A.dim() == 3 and A.shape[-1] != A.shape[-2]:  # 粗略判断是否 [B,D,L]
        A = A.transpose(1, 2).contiguous()           # [B,L,D]

    N_img, H_p, W_p = infer_image_tokens(x, patch_size=patch_size)
    extras = 1 + num_clip_token + extras_extra

    time_tok = A[:, 0:1, :]                    # [B, 1, D]
    text_tok = A[:, 1:1+num_clip_token, :]     # [B, num_clip_token, D]
    img_tok  = A[:, extras:extras+N_img, :]    # [B, N_img, D]

    return dict(time=time_tok, text=text_tok, img=img_tok, H_p=H_p, W_p=W_p)

def tokens_to_2d(A_tok, H_p, W_p):
    """
    A_tok: [B, N_img, D] → [B, D, H_p, W_p]
    """
    B, N_img, D = A_tok.shape
    A_2d = A_tok.transpose(1, 2).contiguous().view(B, D, H_p, W_p)
    return A_2d


def gradcam_on_image_tokens(A, dA, num_clip_token, x, patch_size=2, extras_extra=0):
    """
    从整层激活/梯度中抽取“图像 tokens”做 Grad-CAM。
    A, dA: 通常来自 hook，形状 [B, L, D] 或 [B, D, L]
    返回: CAM 的低分辨率图 [B,1,H_p,W_p]
    """
    parts_A  = split_tokens(A,  x, num_clip_token=num_clip_token, patch_size=patch_size, extras_extra=extras_extra)
    parts_dA = split_tokens(dA, x, num_clip_token=num_clip_token, patch_size=patch_size, extras_extra=extras_extra)

    A_img  = tokens_to_2d(parts_A['img'],  parts_A['H_p'], parts_A['W_p'])   # [B,D,H_p,W_p]
    dA_img = tokens_to_2d(parts_dA['img'], parts_A['H_p'], parts_A['W_p'])

    alpha   = dA_img.mean(dim=(2, 3))                         # [B, D]
    weights = alpha.unsqueeze(-1).unsqueeze(-1)               # [B, D, 1, 1]
    cam     = (weights * A_img).sum(dim=1, keepdim=True).relu()  # [B,1,H_p,W_p]
    return cam, parts_A['H_p'], parts_A['W_p']

def upsample_cam(cam, H_img, W_img):
    cam = F.interpolate(cam, size=(H_img, W_img), mode='bilinear', align_corners=False)
    cam = cam - cam.amin(dim=(1,2,3), keepdim=True)
    cam = cam / (cam.amax(dim=(1,2,3), keepdim=True) + 1e-8)
    return cam

@contextmanager
def register_attention_hooks(model, forward_hook_fn, backward_hook_fn):
    """上下文管理器,自动管理attention hook的注册和释放"""
    hook_handles = []
    
    try:
        # 注册前向和后向钩子到输入块
        for i, block in enumerate(model.in_blocks):
            print(f"注册hook到in_blocks[{i}].attn")
            forward_handle = block.attn.register_forward_hook(forward_hook_fn)
            backward_handle = block.attn.register_full_backward_hook(backward_hook_fn)
            hook_handles.extend([forward_handle, backward_handle])

        # 注册到中间块的attention
        print("注册hook到mid_block.attn")
        forward_handle = model.mid_block.attn.register_forward_hook(forward_hook_fn)
        backward_handle = model.mid_block.attn.register_full_backward_hook(backward_hook_fn)
        hook_handles.extend([forward_handle, backward_handle])

        # 注册到输出块的attention
        for i, block in enumerate(model.out_blocks):
            print(f"注册hook到out_blocks[{i}].attn")
            forward_handle = block.attn.register_forward_hook(forward_hook_fn)
            backward_handle = block.attn.register_full_backward_hook(backward_hook_fn)
            hook_handles.extend([forward_handle, backward_handle])
        
        print(f"总共注册了 {len(hook_handles)} 个hook")
        yield hook_handles
        
    finally:
        # 自动释放所有hook
        print(f"释放 {len(hook_handles)} 个hook...")
        for handle in hook_handles:
            handle.remove()
        print("所有hook已释放")

@torch.no_grad()
def _prep_empty_context(dataset, B, device):
    empty = torch.tensor(dataset.empty_context, device=device)
    return einops.repeat(empty, 'L D -> B L D', B=B)

def compute_cam_once(nnet, x_t, t_vec, context, empty_context, cfg_scale,
                     num_clip_token, patch_size,
                     activations, gradients):
    """
    在当前 (x_t, t_vec) 上执行一次 CFG 前向与反向，使用 hooks 里的 A/dA 计算 CAM（只看图像 tokens）。
    返回 cam_low: [B,1,H_p,W_p], A_shape, dA_shape 便于调试。
    """
    # === 允许梯度 ===
    with torch.enable_grad():
        x_t = x_t.detach().requires_grad_(True)
        nnet.zero_grad(set_to_none=True)

        # 1) CFG 前向（保持与你采样时的一致）
        pred_cond = nnet(x_t, t_vec, context=context)  # [B,4,H,W]
        if cfg_scale != 0:
            pred_uncond = nnet(x_t, t_vec, context=empty_context)
            pred = pred_cond + cfg_scale * (pred_cond - pred_uncond)
        else:
            pred = pred_cond

        # 2) 标量目标并反传，触发 backward_hook
        S = pred.sum()
        S.backward(retain_graph=True)

    # 3) 从 hooks 拿激活与梯度
    A  = activations['value']   # 预期 [B,L,D] 或 [B,D,L]
    dA = gradients['value']
    # 统一成 [B,L,D]
    # if A.dim()==3 and A.shape[-1]!=A.shape[-2]:
    #     A  = A.transpose(1,2).contiguous()
    #     dA = dA.transpose(1,2).contiguous()

    # 4) 切出图像 tokens → 还原2D → Grad-CAM
    from math import isfinite
    N_img, H_p, W_p = infer_image_tokens(x_t, patch_size=patch_size)
    extras = 1 + num_clip_token
    A_img  = A[:, extras:extras+N_img, :]   # [B,N_img,D]
    dA_img = dA[:, extras:extras+N_img, :]

    A_2d  = A_img.transpose(1,2).contiguous().view(x_t.size(0), -1, H_p, W_p)   # [B,D,H_p,W_p]
    dA_2d = dA_img.transpose(1,2).contiguous().view(x_t.size(0), -1, H_p, W_p)  # [B,D,H_p,W_p]

    alpha   = dA_2d.mean(dim=(2,3))                      # [B,D]
    weights = alpha.unsqueeze(-1).unsqueeze(-1)          # [B,D,1,1]
    cam_low = (weights * A_2d).sum(dim=1, keepdim=True).relu()  # [B,1,H_p,W_p]

    return cam_low, tuple(A.shape), tuple(dA.shape)

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def evaluate(config):
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

    # 处理单个文本输入
    if config.input_text:
        # 直接使用提供的文本
        prompt = config.input_text
        logging.info(f'使用提供的文本: {prompt}')
    elif config.input_file:
        # 从文件读取单行文本
        with open(config.input_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        logging.info(f'从文件读取文本: {prompt}')
    else:
        raise ValueError("必须提供 input_text 或 input_file 参数")

    print(f"处理文本: {prompt}")

    # 初始化CLIP编码器
    clip = libs.clip.BertEmbedder(version='michiyasunaga/BioLinkBERT-base', mask=True)
    clip.eval()
    clip.to(device)

            
    # 编码单个文本
    context, attn_mask = clip.encode(prompt)  # 传入列表，返回批次维度为1的张量
    # context = context * attn_mask.unsqueeze(-1).to(context.device) # mask

    # 加载神经网络模型
    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'从 {config.nnet_path} 加载模型')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    def cfg_nnet(x, timesteps, context):
        _cond = nnet(x, timesteps, context=context)
        if config.sample.scale == 0:
            print("config.sample.scale == 0, 不使用CFG")
            return _cond
        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))
        _uncond = nnet(x, timesteps, context=_empty_context)
        return _cond + config.sample.scale * (_cond - _uncond)

    denoise_block = accelerator.unwrap_model(nnet)

    target_layer = denoise_block.out_blocks[5].attn
    h_f = target_layer.register_forward_hook(forward_hook)
    h_b = target_layer.register_full_backward_hook(backward_hook)

    def cam_forward(x, t_vec):
        _cond = denoise_block(x, t_vec, context=context)
        if config.sample.scale == 0:
            # print("config.sample.scale == 0, 不使用CFG")
            return _cond
        _empty = torch.tensor(dataset.empty_context, device=device)
        _empty = einops.repeat(_empty, 'L D -> B L D', B=x.size(0))
        _uncond = denoise_block(x, t_vec, context=_empty)
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

    # === 单步 CAM 验证 ===
    B, C, H, W = 1, *config.z_shape   # (4,32,32)、

    H_img = config.sample.resolution; W_img = config.sample.resolution
    # cam = upsample_cam(cam_low, H_img, W_img)   # [1,1,256,256]



    logging.info(config.sample)
    logging.info(f'mixed_precision={config.mixed_precision}')
    logging.info(f'N={N}')

    # 确保输出目录存在
    os.makedirs(config.output_path, exist_ok=True)

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
    z = dpm_solver.cam_sample(
        z_init, steps=config.sample.sample_steps, eps=1./N, T=1.,
        method='multistep',    # 或 'fast' / 'singlestep'，与当前采样一致即可
        order=1,               # 保持外部步就是内部步，但是采样更慢
        cam_enable=True,
        cam_forward=cam_forward,
        cam_num_clip_token=config.nnet.num_clip_token,
        cam_patch_size=config.nnet.patch_size,
        cam_hooks={'activations': activations, 'gradients': gradients},
        cam_steps=None, 
        # cam_steps=[config.sample.get('vis_step', 10)],  # 用 cam_steps 列表替代 cam_vis_step
        cam_stride=5,
        cam_limit=None,
    )

    # 解码生成的潜在表示
    with torch.cuda.amp.autocast():
        samples = dataset.unpreprocess(decode(z)).clamp(0,1)  # [1,3,H_lat*scale, W_lat*scale]
    
    # 没有任何 CAM 帧直接跳过
    if not getattr(dpm_solver, 'cams', None):
        print("cams为空, 可能采样的内部步与外部步骤不匹配, 检查DPM_Solver的order, cam_enable, cam_steps, cam_stride参数")
    else:
        out_dir = config.output_path
        os.makedirs(out_dir, exist_ok=True)

        # 推断放大倍率（例如 latent 32→图 256 则为8）
        H_lat, W_lat = z.shape[-2:]
        vae_scale = int(round(config.resolution / H_lat))
        H_img = H_lat * vae_scale
        W_img = W_lat * vae_scale

        # 最终图尺寸对齐
        final_img = samples[0:1].to(device)
        if final_img.shape[-2:] != (H_img, W_img):
            final_img = F.interpolate(final_img, size=(H_img, W_img), mode='bilinear', align_corners=False)

        # 遍历所有帧
        for slot in dpm_solver.cams:
            step_id = slot["step"]
            cam_low = slot["cam_low"].to(device)           # [1,1,H_p,W_p]
            cam = upsample_cam(cam_low, H_img, W_img).to(device)

            # 该步的图（用该步 latent 解码）
            x_latent = slot["latent"].to(device)
            with torch.cuda.amp.autocast():
                step_img = dataset.unpreprocess(decode(x_latent)).clamp(0,1)
            if step_img.shape[-2:] != (H_img, W_img):
                step_img = F.interpolate(step_img, size=(H_img, W_img), mode='bilinear', align_corners=False)

            # 叠图
            step_overlay, step_heatmap_rgb   = _overlay_cam_rgb(step_img,  cam, alpha=0.4, use_colormap=True)
            final_overlay, final_heatmap_rgb = _overlay_cam_rgb(final_img, cam, alpha=0.4, use_colormap=True)

            # 保存
            save_image(step_overlay,  os.path.join(out_dir, f'cam_step{step_id:03d}_on_step.png'))
            save_image(final_overlay, os.path.join(out_dir, f'cam_step{step_id:03d}_on_final.png'))
            save_image(step_heatmap_rgb,  os.path.join(out_dir, f'cam_step{step_id:03d}_heatmap.png'))

        print(f"CAM 已保存 {len(dpm_solver.cams)} 帧，多步叠图输出目录：{out_dir}")

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
    
    evaluate(config)


if __name__ == "__main__":
    app.run(main)