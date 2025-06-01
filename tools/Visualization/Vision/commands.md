# 基本
```bash
CUDA_VISIBLE_DEVICES=1 python sample_t2i_discrete_Vis.py \
    --config=configs/chestXray14_uvit_small_t2i.py \
    --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth \
    --output_path=/storage/U-ViT/tools/Visualization/Vision/outputs \
    --input_path=/storage/U-ViT/tools/Visualization/Vision/prompts.txt \
    --visualize_attention=True
```

# 带有每步可视化
```bash
CUDA_VISIBLE_DEVICES=1 python sample_t2i_discrete_Vis.py \
    --config=configs/chestXray14_uvit_small_t2i.py \
    --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth \
    --output_path=/storage/U-ViT/tools/Visualization/Vision/outputs \
    --input_path=/storage/U-ViT/tools/Visualization/Vision/prompts.txt \
    --visualize_attention=True \
    --visualize_steps=True \
    --save_step_interval=2
```

# 自定义注意力块和类型
```bash
CUDA_VISIBLE_DEVICES=0,2 accelerate launch --num_processes 2 --mixed_precision fp16 sample_t2i_discrete_Vis.py \
    --config=configs/chestXray14_uvit_small_t2i.py \
    --nnet_path=/storage/U-ViT/workdir/chestXray14_uvit_small_t2i/default/ckpts/55000.ckpt/nnet.pth \
    --output_path=/storage/U-ViT/tools/Visualization/Vision/outputs \
    --input_path=/storage/U-ViT/tools/Visualization/Vision/prompts.txt \
    --visualize_attention=True \
    --attn_vis_blocks=in_5,mid,out_5 \
    --visualize_steps=True \
    --save_step_interval=25 \
    --attn_vis_types=self
```