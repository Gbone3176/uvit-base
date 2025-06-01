import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1206
    config.z_shape = (4, 32, 32)

    # michiyasunaga/BioLinkBERT-base
    # michiyasunaga/BioLinkBERT-large 256 hidden_size 1024, 似乎不太可行
    # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
    # cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    # StanfordAIMI/RadBERT

    config.model_name_or_path = "openai/clip-vit-large-patch14"
    config.resolution = 256


    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl.pth',
        scale_factor=0.23010
    )

    config.train = d(
        n_steps=75000,
        batch_size=256,
        log_interval=500,
        eval_interval=5000,
        save_interval=5000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.9, 0.9),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=0
    )

    config.nnet = d(
        name='uvit_t2i',
        img_size=32,
        in_chans=4,
        patch_size=2,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        clip_dim=768,
        num_clip_token=77
    )

    config.dataset = d(
        name='ChestXray14_features',
        # path='/storage/U-ViT/assets/datasets/ChestXray14-256_features-BiomedCLIP', # debug
        path=f'/storage/U-ViT/assets/datasets/ChestXray14-{config.resolution}_features-clip',
        cfg=True,
        p_uncond=0.1
    )
    
    # train
    config.sample = d(
        sample_steps=50,
        n_samples=10000,
        mini_batch_size=50,
        cfg=True,
        scale=1.,
        path=f'/storage/U-ViT/sample/ChestXray-t2i-{config.resolution}_features-{config.model_name_or_path.split("/")[-1]}',
    )

    # # eval
    # config.sample = d(
    #     sample_steps=50,
    #     n_samples=1000,
    #     mini_batch_size=50,
    #     cfg=True,
    #     scale=1.,
    #     path=f'/storage/U-ViT/sample/ChestXray-t2i-{config.resolution}_features-{config.model_name_or_path.split("/")[-1]}/eval',
    # )

    return config
