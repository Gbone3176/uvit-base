import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
import torch.nn.functional as F
from collections import Counter
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = '5,6'

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product: 将一个标量或标量数组与一个张量进行广播相乘
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):  # sample from q(xn|x0), where n is uniform
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = torch.randn_like(x0)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def LSimple(x0, nnet, schedule, **kwargs):
    n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
    eps_pred, recon_loss = nnet(xn, n, **kwargs)
    diffusion_loss = mos(eps - eps_pred)
    total_loss = diffusion_loss + recon_loss
    return total_loss, diffusion_loss, recon_loss


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    # 配置DistributedDataParallel参数以处理未使用的参数
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)
    # logging.info(config)
    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    # logging.info(True)
    accelerator.wait_for_everyone()
    # logging.info(True)
    if accelerator.is_main_process:
        # wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
        #            name=config.hparams, job_type='train', mode='offline')
        # Initialize TensorBoard writer
        tensorboard_dir = os.path.join(config.workdir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        tb_writer = SummaryWriter(tensorboard_dir)
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
        tb_writer = None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=True)

    ### set resampler
    targets = train_dataset.targets # 获取所有标签并计算每个类别的样本数量
    class_counts = Counter(targets)
    print("class_counters:", class_counts)
    num_samples = len(targets) # 计算训练集样本总数，以此计算每个样本的权重，使所有类别均衡
    print("num_samples:", num_samples)
    weights = num_samples / np.array([class_counts[t] for t in targets])
    weights = torch.DoubleTensor(weights)
    weights_counter = Counter(np.array(weights))
    print("weights_counter:", weights_counter)
    
    # 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(weights, num_samples)
    # print("num_sampler:", list(sampler)) # for Debug

    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, sampler=sampler, drop_last=True,
                                      num_workers=16, pin_memory=True, persistent_workers=True)

    ###

    test_dataset = dataset.get_split(split='test', labeled=True)  # for sampling
    test_dataset_loader = DataLoader(test_dataset, batch_size=config.sample.mini_batch_size, shuffle=True, drop_last=True,
                                     num_workers=16, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    logging.info(True)
    nnet, nnet_ema, optimizer, train_dataset_loader, test_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader, test_dataset_loader)
    lr_scheduler = train_state.lr_scheduler

    # 这里加载nnet的主要预训练权重    
    # weight = torch.load(config.pretrain.nnet_path, map_location='cpu')
    # pos_embed = weight['pos_embed']
    # pos_start = pos_embed[:, 0, :]
    # pos_middle = pos_embed[:, 1:78, :]
    # pos_end = pos_embed[:, 78:, :]
    # pos_middle = F.interpolate(pos_middle.permute(2, 0, 1), (257), mode='linear').permute(1, 2, 0)
    # pos_embed = torch.cat((pos_start.unsqueeze(0), pos_middle, pos_end), dim=1)
    # weight['pos_embed'] = pos_embed
    # accelerator.unwrap_model(nnet).load_state_dict(weight, strict=False)

    # # 加载包含 label_emb 的第二个权重文件
    # label_weight_path = config.pretrain.label_emb_weight  
    # try:
    #     label_weight = torch.load(label_weight_path, map_location='cpu')
    #     if 'label_emb.weight' in label_weight:
    #         # 取前16个类别的嵌入向量（根据你的num_classes调整）
    #         weight['label_emb.weight'] = label_weight['label_emb.weight'][:config.nnet.num_classes, :]
    #         logging.info(f"Successfully loaded label_emb.weight from {label_weight_path}")
    #         logging.info(f"label_emb.weight shape: {weight['label_emb.weight'].shape}")
    #     else:
    #         logging.warning(f"label_emb.weight not found in {label_weight_path}")
    # except FileNotFoundError:
    #     logging.warning(f"Label weight file not found: {label_weight_path}")
    #     logging.info("Will initialize label_emb randomly")
    # except Exception as e:
    #     logging.error(f"Error loading label weights: {e}")

    # # 加载合并后的权重
    # accelerator.unwrap_model(nnet).load_state_dict(weight, strict=False)

    # logging.info(True)

    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(**config.autoencoder)
    autoencoder.to(device)

    @ torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    def get_context_generator():
        while True:
            for data in test_dataset_loader:
                ### input both text and label
                _, _context, _label = data
                yield _context, _label

    context_generator = get_context_generator()

    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(_betas)
    logging.info(f'use {_schedule}')

    def cfg_nnet(x, timesteps, context, label):
        ### input text and label condition
        _cond = nnet_ema(x, timesteps, context=context, y=label)

        _empty_context = torch.tensor(dataset.empty_context, device=device)
        _empty_context = einops.repeat(_empty_context, 'L D -> B L D', B=x.size(0))

        _empty_class = torch.tensor(dataset.K, device=device).unsqueeze(dim=0)
        _empty_class = einops.repeat(_empty_class, 'C -> B C', B=x.size(0)).squeeze(dim=1)
        _uncond = nnet_ema(x, timesteps, context=_empty_context, y=_empty_class)

        return _cond + config.sample.scale * (_cond - _uncond)

    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        _z = autoencoder.sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0])
        total_loss, diffusion_loss, recon_loss = LSimple(_z, nnet, _schedule, context=_batch[1], y=_batch[2], x_clean=_z)  # currently only support the extracted feature version
        
        # Gather losses for logging
        _metrics['loss'] = accelerator.gather(total_loss.detach()).mean()
        _metrics['diffusion_loss'] = accelerator.gather(diffusion_loss.detach()).mean()
        _metrics['recon_loss'] = accelerator.gather(recon_loss.detach()).mean()
        
        accelerator.backward(total_loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        
        # Log to TensorBoard
        if accelerator.is_main_process and tb_writer is not None:
            tb_writer.add_scalar('Train/Total_Loss', _metrics['loss'].item(), train_state.step)
            tb_writer.add_scalar('Train/Diffusion_Loss', _metrics['diffusion_loss'].item(), train_state.step)
            tb_writer.add_scalar('Train/Reconstruction_Loss', _metrics['recon_loss'].item(), train_state.step)
            tb_writer.add_scalar('Train/Learning_Rate', train_state.optimizer.param_groups[0]['lr'], train_state.step)
        
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            return cfg_nnet(x, t, **kwargs)

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)
        return decode(_z)

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm=dpm_solver, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _context, _label = next(context_generator)
            assert _context.size(0) == _n_samples
            return dpm_solver_sample(_n_samples, sample_steps, context=_context, label =_label)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                # wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
                
                # Log FID to TensorBoard
                if tb_writer is not None:
                    tb_writer.add_scalar(f'Eval/FID_{n_samples}', _fid, train_state.step)
                    
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            # wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            contexts = torch.tensor(dataset.contexts, device=device)[: 2 * 5]
            labels = torch.tensor(dataset.labels, device=device)[: 2 * 5]
            samples = dpm_solver_sample(_n_samples=2 * 5, _sample_steps=50, context=contexts, label=labels)
            samples_grid = make_grid(dataset.unpreprocess(samples), 5)
            # 保存图像到文件系统
            save_image(samples_grid, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            # 将图像添加到TensorBoard
            if accelerator.is_main_process and tb_writer is not None:
                tb_writer.add_image('Generated_Samples', samples_grid, train_state.step)
            # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            fid = eval_step(n_samples=10000, sample_steps=50)  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)
    
    # Close TensorBoard writer
    if accelerator.is_main_process and tb_writer is not None:
        tb_writer.close()



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)

    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)