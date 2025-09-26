import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
import csv


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit':
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    elif name == 'uvit_pn':
        from libs.uvit_pro import UViTpn
        return UViTpn(**kwargs)
    elif name == 'uvit_t2i_label_pron':
        from libs.uvit_t2i_label_pron import UViT
        return UViT(**kwargs)
    elif name == 'unet_t2i_label_pron':
        from libs.unet_t2i_label_pron import UNetModel
        return UNetModel(**kwargs)
    elif name.startswith('DiT'):
        # Support for specific DiT model configurations
        from libs.dit_t2i_label_pron import DiT_models
        if name in DiT_models:
            return DiT_models[name](**kwargs)
        else:
            raise NotImplementedError(f"DiT model '{name}' not found. Available models: {list(DiT_models.keys())}")
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1

def sample2dir_labels(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    labels_path = os.path.join(path, "labels.csv")  # 标签保存路径

    # 初始化 CSV 文件并写入表头
    if accelerator.is_main_process:
        with open(labels_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])  # CSV 文件头

    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        # 调用 sample_fn 获取图像和标签
        samples, labels = sample_fn(mini_batch_size)  # 假设 sample_fn 返回 (images, labels)
        
        # 反预处理图像
        if unpreprocess_fn is not None:
            samples = unpreprocess_fn(samples)
        
        # 使用 accelerator.gather 收集所有进程的数据
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        labels = accelerator.gather(labels.contiguous())[:_batch_size]

        # 主进程保存图像和标签
        if accelerator.is_main_process:
            for sample, label in zip(samples, labels):
                # 保存图像
                save_image(sample, os.path.join(path, f"{idx}.png"))
                
                # 保存标签到 CSV 文件
                with open(labels_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([f"{idx}.png", label.item()])  # 假设 label 是单个值
                
                idx += 1

def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def infer_image_tokens(x, patch_size=2):
    """
    x: 输入的latent embedding, 提取N_img 和w和h方向的 patch 数量
    """
    H, W = x.shape[-2], x.shape[-1]
    H_p, W_p = H // patch_size, W // patch_size
    N_img = H_p * W_p
    return N_img, H_p, W_p

def compute_cam_once_from_callable(cam_forward,               # callable: (x_t, t_vec) -> pred tensor
                                   x_t, t_vec,                # 当前步 latent & 时间索引
                                   num_clip_token, patch_size,
                                   activations, gradients):   # 全局 hook 容器
    """
    在 (x_t, t_vec) 上执行一次前向与反向，触发 hooks，产出“图像 tokens”的低分辨率 CAM。
    返回 cam_low: [B,1,H_p,W_p]
    """
    # 允许梯度
    with torch.enable_grad():
        x_t = x_t.detach().requires_grad_(True)
        # 1) 前向（由外部把 CFG 等细节封装在 cam_forward 里）
        pred = cam_forward(x_t, t_vec)    # 形状通常是 [B,4,H,W]
        # 2) 标量目标并反传
        S = pred.sum()
        S.backward(retain_graph=True)

    # 3) 从 hooks 读取 A/dA，并统一到 [B,L,D]
    A  = activations['value']
    dA = gradients['value']
    # if A.dim()==3 and A.shape[-1]!=A.shape[-2]:
    #     A  = A.transpose(1,2).contiguous()
    #     dA = dA.transpose(1,2).contiguous()

    # 4) 只取图像 tokens → 还原 2D 网格 → Grad-CAM
    N_img, H_p, W_p = infer_image_tokens(x_t, patch_size=patch_size)
    extras = 1 + num_clip_token                           # time(1) + text(num_clip_token)
    A_img  = A[:, extras:extras+N_img, :]                 # [B,N_img,D]
    dA_img = dA[:, extras:extras+N_img, :]

    A_2d  = A_img.transpose(1,2).contiguous().view(x_t.size(0), -1, H_p, W_p)   # [B,D,H_p,W_p]
    dA_2d = dA_img.transpose(1,2).contiguous().view(x_t.size(0), -1, H_p, W_p)  # [B,D,H_p,W_p]

    alpha   = dA_2d.mean(dim=(2,3))                      # [B,D]
    weights = alpha.unsqueeze(-1).unsqueeze(-1)          # [B,D,1,1]
    cam_low = (weights * A_2d).sum(dim=1, keepdim=True).relu()  # [B,1,H_p,W_p]
    return cam_low

def _overlay_cam_rgb(base_img_01, cam01, alpha=0.4, use_colormap=True):
    """
    base_img_01: [B,3,H,W], 0~1
    cam01:       [B,1,H,W], 0~1
    返回: overlay [B,3,H,W], heatmap_rgb [B,3,H,W]
    """
    B, _, H, W = base_img_01.shape
    cam_np = (cam01[0,0].detach().cpu().numpy()*255).astype(np.uint8)

    heatmap_rgb = None
    if use_colormap:
        try:
            import cv2
            hm = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0  # BGR->RGB
            heatmap_rgb = torch.from_numpy(hm).permute(2,0,1).unsqueeze(0).float().to(base_img_01.device)  # [1,3,H,W]
        except Exception:
            use_colormap = False

    if not use_colormap:
        heatmap_rgb = cam01.repeat(1,3,1,1)  # 退化为灰度

    overlay = (1 - alpha) * base_img_01 + alpha * heatmap_rgb
    overlay = overlay.clamp(0,1)
    return overlay, heatmap_rgb

    # ====== 多步 CAM 采集工具 ======
def _should_capture(step_counter, cam_steps=None, cam_stride=None, cam_limit=None, already_captured=0):
    """
    决定当前步是否需要做 CAM：
    - cam_steps: list[int]，明确指哪些外部步
    - cam_stride: int，每隔多少步采一次（从1开始计数）
    - cam_limit: 最多采集多少帧（None=不限制）
    二者都提供时，满足其一即可；优先检查 cam_steps 命中。
    """
    if cam_limit is not None and already_captured >= cam_limit:
        return False
    if cam_steps is not None:
        if step_counter in cam_steps:
            return True
    if cam_stride is not None and cam_stride > 0:
        if (step_counter % cam_stride) == 0:
            return True
    return False

def _append_cam_slot(obj, step_counter, cam_low, x_req, t_vec):
    """
    把当前帧存入对象 obj（通常是 dpm_solver 实例）。
    """
    if not hasattr(obj, "cams"):
        obj.cams = []
    obj.cams.append({
        "step": int(step_counter),
        "cam_low": cam_low.detach().cpu(),     # [B,1,H_p,W_p]
        "latent": x_req.detach().cpu(),        # [B,4,H_lat,W_lat]
        "t": t_vec.detach().cpu(),             # [B]
    })