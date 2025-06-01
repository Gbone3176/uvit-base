import torch
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from torchvision.utils import make_grid, save_image
import utils
import sde
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from absl import logging
import libs.autoencoder
import einops
import tempfile
import numpy as np
from collections import Counter
from accelerate import Accelerator
from datasets import get_dataset
from tools.fid_score import calculate_fid_given_paths
import wandb

from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


def load_checkpoint_and_generate_samples(config, checkpoint_dir, class_sample_counts, sample_steps=50,
                                         algorithm='dpm_solver'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accelerator = Accelerator()
    device = accelerator.device
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    # Initialize model, optimizer, etc.
    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, lr_scheduler = (
        train_state.nnet,
        train_state.nnet_ema,
        train_state.optimizer,
        train_state.lr_scheduler
    )

    # set the score_model to train
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())

    # Load checkpoint files
    checkpoint_files = {f: os.path.join(checkpoint_dir, f) for f in
                        ['nnet.pth', 'nnet_ema.pth', 'optimizer.pth', 'lr_scheduler.pth']}

    # Load the network state dict
    if os.path.exists(checkpoint_files['nnet.pth']):
        nnet.load_state_dict(torch.load(checkpoint_files['nnet.pth'], map_location='cpu'))
    if os.path.exists(checkpoint_files['nnet_ema.pth']):
        nnet_ema.load_state_dict(torch.load(checkpoint_files['nnet_ema.pth'], map_location='cpu'))
    if os.path.exists(checkpoint_files['optimizer.pth']):
        optimizer.load_state_dict(torch.load(checkpoint_files['optimizer.pth'], map_location='cpu'))
    if os.path.exists(checkpoint_files['lr_scheduler.pth']):
        lr_scheduler.load_state_dict(torch.load(checkpoint_files['lr_scheduler.pth'], map_location='cpu'))

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def sample_fn_per_class(_class, _n_samples):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        kwargs = dict(y=torch.full((_n_samples,), _class, device=device, dtype=torch.long))

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

    path = config.sample_dir or tempfile.mkdtemp()
    os.makedirs(path, exist_ok=True)

    filename = os.path.basename(checkpoint_dir)
    basename_without_ext = os.path.splitext(filename)[0]
    step_number = int(basename_without_ext)

    for _class, _n_samples in class_sample_counts.items():
        samples = sample_fn_per_class(_class, _n_samples)
        class_path = os.path.join(path, f'class_{_class}')
        os.makedirs(class_path, exist_ok=True)
        for i, sample in enumerate(samples):
            save_image(sample, os.path.join(class_path, f'{step_number}_{i}.png'))



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
    # config.nnet_path = '/storage/ScientificPrograms/Conditional_Diffusion/U-ViT-main/weights/imagenet256_uvit_large.pth'
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples_Vis')
    # Specify the number of samples per class
    class_sample_counts = {0:10, 1:10, 2:10, 3:10, 4:10, 5:10, 6:10}  # Adjust class indices and number of samples as needed

    checkpoints = [os.path.join(config.ckpt_root, f'{step}.ckpt') for step in
                   [50000, 100000, 150000, 200000, 250000, 300000]]  # Add more steps as needed
    for ckpt in checkpoints:
        load_checkpoint_and_generate_samples(config, ckpt, class_sample_counts)

if __name__ == "__main__":
    app.run(main)
