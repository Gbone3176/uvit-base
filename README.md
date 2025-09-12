This codebase implements the transformer-based backbone 📌*U-ViT*📌 for diffusion models, as introduced in the [paper](https://arxiv.org/abs/2209.12152).
U-ViT treats all inputs as tokens and employs long skip connections. *The long skip connections grealy promote the performance and the convergence speed*.

<img src="skip_im.png" alt="drawing" width="400"/>

💡This codebase contains:
* An implementation of [U-ViT](libs/uvit.py) with optimized attention computation
* Pretrained U-ViT models on common image generation benchmarks (CIFAR10, CelebA 64x64, ImageNet 64x64, ImageNet 256x256, ImageNet 512x512)
* Efficient training scripts for [pixel-space diffusion models](train.py), [latent space diffusion models](train_ldm_discrete.py) and [text-to-image diffusion models](train_t2i_discrete.py)
* Efficient evaluation scripts for [pixel-space diffusion models](eval.py) and [latent space diffusion models](eval_ldm_discrete.py) and [text-to-image diffusion models](eval_t2i_discrete.py)
* A Colab notebook demo for sampling from U-ViT on ImageNet (FID=2.29) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baofff/U-ViT/blob/main/UViT_ImageNet_demo.ipynb)


<img src="sample.png" alt="drawing" width="800"/>


💡This codebase supports useful techniques for efficient training and sampling of diffusion models:
* Mixed precision training with the [huggingface accelerate](https://github.com/huggingface/accelerate) library (🥰automatically turned on)
* Efficient attention computation with the [facebook xformers](https://github.com/facebookresearch/xformers) library (needs additional installation)
* Gradient checkpointing trick, which reduces ~65% memory (🥰automatically turned on)
* With these techniques, we are able to train our largest U-ViT-H on ImageNet at high resolutions such as 256x256 and 512x512 using a large batch size of 1024 with *only 2 A100*❗


Training speed and memory of U-ViT-H/2 on ImageNet 256x256 using a batch size of 128 with a A100:

| mixed precision training | xformers | gradient checkpointing |  training speed   |    memory     |
|:------------------------:|:--------:|:----------------------:|:-----------------:|:-------------:|
|            ❌             |    ❌     |           ❌            |         -         | out of memory |
|            ✔             |    ❌     |           ❌            | 0.97 steps/second |   78852 MB    |
|            ✔             |    ✔     |           ❌            | 1.14 steps/second |   54324 MB    |
|            ✔             |    ✔     |           ✔            | 0.87 steps/second |   18858 MB    |



## Dependency

```sh
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116  # install torch-1.13.1
pip install accelerate==0.12.0 absl-py ml_collections einops wandb ftfy==6.1.1 transformers==4.23.1

# xformers is optional, but it would greatly speed up the attention computation.
pip install -U xformers
pip install -U --pre triton
```

* This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+. (Perhaps other versions also work, but I haven't tested it.)
* We highly suggest install [xformers](https://github.com/facebookresearch/xformers), which would greatly speed up the attention computation for *both training and inference*.



## Pretrained Models


|                                                         Model                                                          |  FID  | training iterations | batch size |
|:----------------------------------------------------------------------------------------------------------------------:|:-----:|:-------------------:|:----------:|
|      [CIFAR10 (U-ViT-S/2)](https://drive.google.com/file/d/1yoYyuzR_hQYWU0mkTj659tMTnoCWCMv-/view?usp=share_link)      | 3.11  |        500K         |    128     |
|   [CelebA 64x64 (U-ViT-S/4)](https://drive.google.com/file/d/13YpbRtlqF1HDBNLNRlKxLTbKbKeLE06C/view?usp=share_link)    | 2.87  |        500K         |    128     |
|  [ImageNet 64x64 (U-ViT-M/4)](https://drive.google.com/file/d/1igVgRY7-A0ZV3XqdNcMGOnIGOxKr9azv/view?usp=share_link)   | 5.85  |        300K         |    1024    |
|  [ImageNet 64x64 (U-ViT-L/4)](https://drive.google.com/file/d/19rmun-T7RwkNC1feEPWinIo-1JynpW7J/view?usp=share_link)   | 4.26  |        300K         |    1024    |
| [ImageNet 256x256 (U-ViT-L/2)](https://drive.google.com/file/d/1w7T1hiwKODgkYyMH9Nc9JNUThbxFZgs3/view?usp=share_link)  | 3.40  |        300K         |    1024    |
| [ImageNet 256x256 (U-ViT-H/2)](https://drive.google.com/file/d/13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u/view?usp=share_link)  | 2.29  |        500K         |    1024    |
| [ImageNet 512x512 (U-ViT-L/4)](https://drive.google.com/file/d/1mkj4aN2utHMBTWQX9l1nYue9vleL7ZSB/view?usp=share_link)  | 4.67  |        500K         |    1024    |
| [ImageNet 512x512 (U-ViT-H/4)](https://drive.google.com/file/d/1uegr2o7cuKXtf2akWGAN2Vnlrtw5YKQq/view?usp=share_link)  | 4.05  |        500K         |    1024    |
|      [MS-COCO (U-ViT-S/2)](https://drive.google.com/file/d/15JsZWRz2byYNU6K093et5e5Xqd4uwA8S/view?usp=share_link)      | 5.95  |         1M          |    256     |
|   [MS-COCO (U-ViT-S/2, Deep)](https://drive.google.com/file/d/1gHRy8sn039Wy-iFL21wH8TiheHK8Ky71/view?usp=share_link)   | 5.48  |         1M          |    256     |



## Preparation Before Training and Evaluation

#### Autoencoder
Download `stable-diffusion` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains image autoencoders converted from [Stable Diffusion](https://github.com/CompVis/stable-diffusion)). 
Put the downloaded directory as `assets/stable-diffusion` in this codebase.
The autoencoders are used in latent diffusion models.

#### Data
* ImageNet 64x64: Put the standard ImageNet dataset (which contains the `train` and `val` directory) to `assets/datasets/ImageNet`.
* ImageNet 256x256 and ImageNet 512x512: Extract ImageNet features according to `scripts/extract_imagenet_feature.py`.
* MS-COCO: Download COCO 2014 [training](http://images.cocodataset.org/zips/train2014.zip), [validation](http://images.cocodataset.org/zips/val2014.zip) data and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip). Then extract their features according to `scripts/extract_mscoco_feature.py` `scripts/extract_test_prompt_feature.py` `scripts/extract_empty_feature.py`.

the corresponding txt is stored at `/storage/U-ViT/scripts/Vis_caps`

```bash
#Biomedclip
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/storage/U-ViT python scripts/ChestXray14-BioMedClip/extract_test_prompt_feature.py

# pubmedclip
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/storage/U-ViT python scripts/ChestXray14-pubmedclip/extract_CXR14_feature.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/storage/U-ViT python scripts/ChestXray14-pubmedclip/extract_empty_feature.py

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/storage/U-ViT python scripts/ChestXray14-pubmedclip/extract_test_prompt_feature.py

# 以下是BERT系列的model_name
# michiyasunaga/BioLinkBERT-base (256)
# michiyasunaga/BioLinkBERT-large (256 hidden_size 1024, 似乎不太可行)
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
# StanfordAIMI/RadBERT

#bert-series
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/storage/U-ViT python scripts/VAP/extract_feature.py --split train --version michiyasunaga/BioLinkBERT-base

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/storage/U-ViT python scripts/VAP/extract_empty_feature.py --version michiyasunaga/BioLinkBERT-base

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/storage/U-ViT python scripts/VAP/extract_test_prompt_feature.py --version michiyasunaga/BioLinkBERT-base
```

#### Reference statistics for FID
Download `fid_stats` directory from this [link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) (which contains reference statistics for FID).
Put the downloaded directory as `assets/fid_stats` in this codebase.
In addition to evaluation, these reference statistics are used to monitor FID during the training process.

## Training
We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library to help train with distributed data parallel and mixed precision. The following is the training command:
```sh
# the training setting
num_processes=2  # the number of gpus you have, e.g., 2
train_script=train.py  # the train script, one of <train.py|train_ldm.py|train_ldm_discrete.py|train_t2i_discrete.py>
                       # train.py: training on pixel space
                       # train_ldm.py: training on latent space with continuous timesteps
                       # train_ldm_discrete.py: training on latent space with discrete timesteps
                       # train_t2i_discrete.py: text-to-image training on latent space
config=configs/cifar10_uvit_small.py  # the training configuration
                                      # you can change other hyperparameters by modifying the configuration file

# launch training
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 $train_script --config=$config
```


We provide all commands to reproduce U-ViT training in the paper:
```sh
# CIFAR10 (U-ViT-S/2)
CUDA_VISIBLE_DEVICES=3,7 accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision fp16 train_ldm_discrete.py --config=configs/cifar10_uvit_small.py

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --config=configs/celeba64_uvit_small.py 

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_mid.py

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train.py --config=configs/imagenet64_uvit_large.py

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet256_uvit_large.py

# ImageNet 256x256 (U-ViT-H/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet256_uvit_huge.py

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm.py --config=configs/imagenet512_uvit_large.py

# ImageNet 512x512 (U-ViT-H/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 train_ldm_discrete.py --config=configs/imagenet512_uvit_huge.py

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16

# ChestXray14 (U-ViT-S/2) T2I @Biomedclip
#原始版本
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision fp16 train_t2i_discrete.py --config=configs/chestXray14_uvit_small_t2i.py

#query版本,不work
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision fp16 train_t2i_discrete.py --config=configs/chestXray14_uvit_small_t2i_query.py

# ChestXray14 256x256 (U-ViT-S/2 @pubmedclip)
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision fp16 train_t2i_discrete.py --config=configs/chestXray14_uvit_small_t2i_pubmedclip.py

# Biomedclip
# michiyasunaga/BioLinkBERT-base 
# michiyasunaga/BioLinkBERT-large (hidden_size 1024, 似乎不太可行)
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
# StanfordAIMI/RadBERT

# ChestXray14 256x256 (U-ViT-S/2 @bert-series)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --mixed_precision fp16  train_t2i_discrete.py --config=configs/chestXray14_uvit_small_t2i_bert.py  --config.model_name_or_path=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext

# #RL版本没做
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 train_t2i_discrete.py --config=configs/chestXray14_uvit_small_t2i_RL.py


# ChestXray14 256x256 (U-ViT-H/2 @Biomedclip) 标签生图
CUDA_VISIBLE_DEVICES=0,6 accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_ldm_discrete.py --config=configs/chestXray14_256_ldm_uvit.py 

# ChestXray14 256x256 (U-ViT-H/2 @Biomedclip & Prototype)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_ldm_pro.py --config=configs/chestXray14_256_ldm_uvit_pro.py


# ChestXray14 256x256 (U-ViT-H/2 @Biomedclip & Prototype)
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_ldm_pro.py --config=configs/chestXray14_256_ldm_uvit_pro.py

#VAP Xray14 256x256 (U-ViT-S/2 deep @BioLinkBERT & Prototype)
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_ISIC_small_t2i.py
```

## experiments of denoising block
```bash
# U-ViT-small-deep
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_small_t2i_deep.py

CUDA_VISIBLE_DEVICES=6 accelerate launch --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_small_t2i_deep.py

# UNet

# DiT

# UDiT

```


## Fetch the statistics of datasets

```bash
cd tools
# ChestXray14
python cal_fid_score.py /storage/dataset/ChestXray14/images /storage/U-ViT/assets/fid_stats/fid_stats_ChestXray14.npz --batch-size 256 --dims 2048 --device cuda:1 --num-workers 4
```

## Evaluation (Compute FID)

We use the [huggingface accelerate](https://github.com/huggingface/accelerate) library for efficient inference with mixed precision and multiple gpus. The following is the evaluation command:
```sh
# the evaluation setting
num_processes=2  # the number of gpus you have, e.g., 2
eval_script=eval.py  # the evaluation script, one of <eval.py|eval_ldm.py|eval_ldm_discrete.py|eval_t2i_discrete.py>
                     # eval.py: for models trained with train.py (i.e., pixel space models)
                     # eval_ldm.py: for models trained with train_ldm.py (i.e., latent space models with continuous timesteps)
                     # eval_ldm_discrete.py: for models trained with train_ldm_discrete.py (i.e., latent space models with discrete timesteps)
                     # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
config=configs/cifar10_uvit_small.py  # the training configuration

# launch evaluation
accelerate launch --multi_gpu --num_processes $num_processes --mixed_precision fp16 eval_script --config=$config
```
The generated images are stored in a temperary directory, and will be deleted after evaluation. If you want to keep these images, set `--config.sample.path=/save/dir`.


We provide all commands to reproduce FID results in the paper:
```sh
# CIFAR10 (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/cifar10_uvit_small.py --nnet_path=cifar10_uvit_small.pth

# CelebA 64x64 (U-ViT-S/4)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval.py --config=configs/celeba64_uvit_small.py --nnet_path=celeba64_uvit_small.pth

# ImageNet 64x64 (U-ViT-M/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_mid.py --nnet_path=imagenet64_uvit_mid.pth

# ImageNet 64x64 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval.py --config=configs/imagenet64_uvit_large.py --nnet_path=imagenet64_uvit_large.pth

# ImageNet 256x256 (U-ViT-L/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet256_uvit_large.py --nnet_path=imagenet256_uvit_large.pth

# ImageNet 256x256 (U-ViT-H/2)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm_discrete.py --config=configs/imagenet256_uvit_huge.py --nnet_path=imagenet256_uvit_huge.pth

# ImageNet 512x512 (U-ViT-L/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm.py --config=configs/imagenet512_uvit_large.py --nnet_path=imagenet512_uvit_large.pth

# ImageNet 512x512 (U-ViT-H/4)
accelerate launch --multi_gpu --num_processes 8 --mixed_precision fp16 eval_ldm_discrete.py --config=configs/imagenet512_uvit_huge.py --nnet_path=imagenet512_uvit_huge.pth

# MS-COCO (U-ViT-S/2)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --nnet_path=mscoco_uvit_small.pth

# MS-COCO (U-ViT-S/2, Deep)
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 eval_t2i_discrete.py --config=configs/mscoco_uvit_small.py --config.nnet.depth=16 --nnet_path=mscoco_uvit_small_deep.pth

# chestXray14 256x256 (U-ViT-H/2 Prototype)
accelerate launch --num_processes 2 --mixed_precision fp16 eval_ldm_pro.py --config=configs/chestXray14_256_ldm_uvit_pro.py --nnet_path=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-ViT/workdir/chestXray14_256_ldm_uvit_pro-seed42/default/ckpts/65000.ckpt/nnet.pth

# ChestXray14  (U-ViT-S/2) @biomedclip
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 --main_process_port 29501 eval_t2i_discrete.py --config=configs/chestXray14_uvit_small_t2i.py  --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth

# ChestXray14  (U-ViT-S/2) @pubmedclip
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 --main_process_port 29501 eval_t2i_discrete.py --config=chestXray14_uvit_small_t2i_pubmedclip.py --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth

# ChestXray14  (U-ViT-S/2) @bert-series
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --mixed_precision fp16 --main_process_port 29501 eval_t2i_discrete.py --config=configs/chestXray14_uvit_small_t2i.py --config.model_name_or_path=Biomedclip --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth
```
## Evaliation (Compute Clip-Score)
```bash
# 采样1000张图像

# biomedclip
CUDA_VISIBLE_DEVICES=0,1 python sample_t2i_discrete.py \
    --config=configs/chestXray14_uvit_small_t2i.py \
    --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth \
    --output_path=/storage/U-ViT/assets/ClipScore/BiomedClip/imgs \
    --input_path=/storage/U-ViT/assets/ClipScore/BiomedClip/run_vis

# Biomedclip
# michiyasunaga/BioLinkBERT-base
# michiyasunaga/BioLinkBERT-large
# microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# cambridgeltl/SapBERT-from-PubMedBERT-fulltext
# StanfordAIMI/RadBERT

#/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth
#/storage/U-ViT/workdir/chestXray14_uvit_small_t2i_bert/BioLinkBERT-base/ckpts/70000.ckpt/nnet.pth
#/storage/U-ViT/workdir/chestXray14_uvit_small_t2i_bert/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/ckpts/75000.ckpt/nnet.pth
#/storage/U-ViT/workdir/chestXray14_uvit_small_t2i_bert/SapBERT-from-PubMedBERT-fulltext/65000.ckpt/nnet.pth
#/storage/U-ViT/workdir/RadBERT/nnet.pth

#bert-series
CUDA_VISIBLE_DEVICES=0 python sample_t2i_discrete.py \
    --config=configs/chestXray14_uvit_small_t2i_bert.py \
    --config.model_name_or_path=cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
    --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i_bert/SapBERT-from-PubMedBERT-fulltext/65000.ckpt/nnet.pth \
    --output_path=/storage/U-ViT/assets/ClipScore/SapBERT-from-PubMedBERT-fulltext/imgs \
    --input_path=/storage/U-ViT/assets/ClipScore/SapBERT-from-PubMedBERT-fulltext/run_vis

CUDA_VISIBLE_DEVICES=0 python sample_t2i_discrete.py \
    --config=configs/chestXray14_uvit_small_t2i_clip.py \
    --nnet_path=/storage/U-ViT/workdir/CLIP/nnet.pth \
    --output_path=/storage/U-ViT/assets/ClipScore/clip-CS/imgs \
    --input_path=/storage/U-ViT/assets/ClipScore/clip-CS/run_vis
```

```bash 
# sample txt

```


```bash
conda activate bionlp

cd /storage/U-ViT/tools/clip-score/src/clip_score

CUDA_VISIBLE_DEVICES=1 python clip_score.py \
/storage/U-ViT/assets/ClipScore/Vis-pics \
/storage/U-ViT/assets/ClipScore/Xray-ORI \
--real_flag=img --fake_flag=txt \
--biomed
```
