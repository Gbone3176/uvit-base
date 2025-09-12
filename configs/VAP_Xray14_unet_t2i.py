import ml_collections

###在这个版本中我们在原始UVIT文生图框架的基础上引入了label embedding
###将label embedding转为label prototype
def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.18215
    )

    config.train = d(
        n_steps=100000,
        batch_size=2,
        log_interval=10,
        eval_interval=2,
        save_interval=2,
    )
    # config.train = d(
    #     n_steps=100000,
    #     batch_size=2,
    #     log_interval=10,
    #     eval_interval=2000,
    #     save_interval=2000,
    # )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5
    )

    config.nnet = d(
        name='unet_t2i_label_pron',
        image_size=32,
        in_channels=4,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions="16,8",
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        num_classes=16,
        use_checkpoint=False,
        num_heads=8,
        num_head_channels=40,
        use_scale_shift_norm=False,
        clip_dim=768,
        num_clip_token=256,
    )

    config.dataset = d(
        name='xray14-img-text-label-features',
        path='/storage/U-ViT/assets/datasets/VAP/ChestXray14-256_features-BioLinkBERT-base',
        cfg=True,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=50,
        cfg=True,
        scale=1.0, 
        path=''
    )
    
    return config
