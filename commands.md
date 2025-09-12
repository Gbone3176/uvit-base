## experiments of denoising block
```bash
# U-ViT-small-deep
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_small_t2i_deep.py

CUDA_VISIBLE_DEVICES=6 accelerate launch --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_small_t2i_deep.py

# UNet
CUDA_VISIBLE_DEVICES=3 accelerate launch --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_unet_t2i.py
# DiT

# UDiT

```