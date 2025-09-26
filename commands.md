## experiments of denoising block
```bash
# U-ViT-small-deep
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --multi_gpu --num_processes 2 --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_small_t2i_deep.py

CUDA_VISIBLE_DEVICES=6 accelerate launch --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_small_t2i_deep.py

# UNet
CUDA_VISIBLE_DEVICES=6 accelerate launch --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_unet_t2i.py

# DiT
CUDA_VISIBLE_DEVICES=5,6 accelerate launch --mixed_precision fp16 train_t2i_discrete_VAP.py --config=configs/VAP_Xray14_dit_b_4_t2i.py
# UDiT

```

## Sample with Grad-Cam
```bash
python sample_t2i_discrete_gradCam.py \
 --config configs/sample/chestXray14_uvit_small_t2i_bert.py \
 --nnet_path /storage/U-ViT/workdir/chestXray14_uvit_small_t2i_bert/BioLinkBERT-base/ckpts/65000.ckpt/nnet.pth \
 --output_path ./outputs \
 --input_text "There is severe edema in the left lower lung." \
 --output_filename "right_lower_edema_scale@0.png"
```
