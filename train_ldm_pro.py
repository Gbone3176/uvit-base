from collections import Counter

import sde_pro as sde
import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
from accelerate.utils import DistributedDataParallelKwargs
# os.environ["ACCELERATE_MASTER_PORT"] = "8888"
# import pdb

def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True # 根据配置启用 cudnn.benchmark 和禁用 cudnn.deterministic 以优化 GPU 性能。
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn') # 设置多进程启动方法为 spawn
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[kwargs]) # 初始化 Accelerator，并获取设备信息
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}') # 记录进程使用的设备信息

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config) # 获取config文件内容

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None # 在子进程中禁用 print
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset) # 获取数据集
    assert os.path.exists(dataset.fid_stat) # 检查计算FID的文件是否存在

    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond') #得到训练集
    # print("lenth of train_dataset:", len(train_dataset)) #debug 用

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
                                      num_workers=12, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device) # 初始化nnet，nnet_ema，optimizer和lr_scheduler
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    lr_scheduler = train_state.lr_scheduler

    ### load pretrained model weights
    # pdb.set_trace()
    weight = torch.load(config.nnet_path, map_location='cpu') # 这里只加载nnet的预训练权重
    logging.info(weight.keys())
    
    weight['label_emb.weight'] = weight['label_emb.weight'][:16, :] # 取决于数据集，嵌入层只保留前K+1个类别的嵌入向量，包括一个与p_cond与有关的zero_tocken
    accelerator.unwrap_model(nnet).load_state_dict(weight, strict=False)
    ###

    train_state.resume(config.ckpt_root) # 加载最新的一个cpkt文件夹下的所有权重文件，恢复网络的训练

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path) # 加载自回归编码器
    autoencoder.to(device)

    @ torch.cuda.amp.autocast()  # 用于自动混合精度 (Automatic Mixed Precision, AMP) 训练
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

    # set the score_model to train
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())


    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        if config.train.mode == 'uncond':
            _z = autoencoder.sample(_batch) if 'feature' in config.dataset.name else encode(_batch) # 无条件，没有标签
            loss = sde.LSimple(score_model, _z, pred=config.pred, x_clean=_z)
        elif config.train.mode == 'cond':
            _z = autoencoder.sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0]) # 有条件，batch[1]一般是标签
            loss = sde.LSimple(score_model, _z, pred=config.pred, y=_batch[1], x_clean=_z)
        else:
            raise NotImplementedError(config.train.mode)
        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999)) # 设置nnet_ema的平滑系数
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics) #把lr，loss等训练参数以dict的形式打包

    def eval_step(n_samples, sample_steps, algorithm):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
            else:
                raise NotImplementedError

            if algorithm == 'euler_maruyama_sde':
                _z = sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _z_init, sample_steps, **kwargs)
            elif algorithm == 'euler_maruyama_ode':
                _z = sde.euler_maruyama(sde.ODE(score_model_ema), _z_init, sample_steps, **kwargs)
            elif algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear')
                model_fn = model_wrapper(
                    score_model_ema.noise_pred,
                    noise_schedule,
                    time_input_type='0',
                    model_kwargs=kwargs
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                _z = dpm_solver.sample(
                    _z_init,
                    steps=sample_steps,
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
            else:
                raise NotImplementedError
            return decode(_z)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess) # 得到生成的图像

            _fid = 0.0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, dtype=torch.float32, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))  # 对每一个batch的数据进行to(device)的操作
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            z_init = torch.randn(15 * 5, *config.z_shape, device=device) # 初始化噪声，不理解为什么不能换
            if config.train.mode == 'uncond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50)
            elif config.train.mode == 'cond':
                y = einops.repeat(torch.arange(15, device=device) % dataset.K, 'nrow -> (nrow ncol)', ncol=5)
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50, y=y)
            else:
                raise NotImplementedError
            samples = decode(z)
            samples = make_grid(dataset.unpreprocess(samples), 5)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            fid = eval_step(n_samples=10000, sample_steps=50, algorithm='dpm_solver')  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt')) # 最后再重复测试一次fid最低的step的FID50K

    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps, algorithm=config.sample.algorithm)



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
    config.nnet_path = '/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-ViT/weights/imagenet256_uvit_large.pth'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)