

Model_cfg = dict(
        model_cfg_name="CaMP_Config",
        model_cfg_args=dict(
            
            pretrained_weights = './model_weights/eva02_L_pt_m38m_p14to16.pth',
            tuning_mode = 'PEFT',
            
            backbone_cfg=dict(
                img_size=384,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4*2/3,
                qkv_bias=True,
                norm_cfg=dict(
                    norm_type="LayerNorm",
                    layer_args=dict(
                        eps=1e-6
                    )
                ),
                subln=True,
                xattn=True,
                naiveswiglu=True,
                rope=True,
                pt_hw_seq_len=24,
                intp_freq=True,
                ft_cfg=[
                    dict(
                        type="backbone_ft",
                        bottleneck=64,
                        adapter_scalar=2.0,
                        learnable_scalar=True,
                        act_cfg=dict(
                            act_type="ReLU",
                            layer_args=dict(
                                inplace=True
                            )
                        ),
                        adapter_layernorm_option=True,
                        dropout_layer=dict(
                            drop_type="Dropout",
                            drop_prob=0.0,
                            inplace=True
                        )
                    ),
                    dict(
                        type="neck_ft",
                        out_channels=256,
                        scale_factors=[4.0, 2.0, 1.0, 0.5],
                        norm_cfg=dict(
                            norm_type="BatchNorm2d"
                        )
                    )
                ]
            ),
            decode_head_cfg=dict(
                in_channels=[256, 256, 256, 256],
                channels=256,
                num_classes=2,
                out_channels=1,
                norm_cfg=dict(
                    norm_type="BatchNorm2d",
                    requires_grad=True,
                    layer_args=dict(
                        eps=1e-5,
                        momentum=0.1,
                        affine=True,
                        track_running_stats=True
                    )
                ),
                in_index=[0, 1, 2, 3],
                align_corners=False,
                ft_cfg=dict(
                    prompt_in_channels=2048,
                    tm_channels=256
                )
            ),
            threshold=None,
            loss_decode=[
                dict(
                    loss_type="BCEWithLogitsLoss",
                    reduction="mean",
                    loss_weight=1.0,
                    loss_name="mask_loss_bce"
                ),
                
                dict(
                    loss_type="DiceLoss",
                    reduction="mean",
                    loss_weight=0.5,
                    loss_name="mask_loss_dice"
                )
            ]
        )
    )
