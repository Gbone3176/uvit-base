import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 42
    config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth'
    )

    config.train = d(
        n_steps=80000,
        batch_size=512,
        mode='cond',
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,

        # n_steps=2,
        # batch_size=2,
        # mode='cond',
        # log_interval=1,
        # eval_interval=1,
        # save_interval=1,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=5000
    )

    config.nnet = d(
        name='uvit_pn',
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=1024,
        depth=20,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=16,
        use_checkpoint=False
    )

    config.dataset = d(
        name='ChestXray14_256_ldm_features',
        path='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-ViT/assets/ChestXray14_256_ldm_features',
        cfg=True,
        p_uncond=0.15
    )

    config.sample = d(
        sample_steps=50,
        n_samples=50000,  # train 最后一轮 测试FID10K
        mini_batch_size=50,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path='/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/U-ViT/sample/ChestXray-ldm-pro-pcound15-50k'
    )

    return config
