

_base_ = [
    # '../_base_/datasets/FGSeg_MM_cfg.py',
    '../_base_/datasets/FGSeg_cfg.py',
    '../_base_/schedules/temp_scheduler.py', 
]


runtime = dict(
    logger_name = 'default',
    
)


Dataset_cfg = dict(
    dataset_cfg_args=dict(
        # data_root=['./datasets/CAMO', './datasets/COD10KCAM'], 
        data_root='./datasets/CAMO', 
        # data_root='./datasets/CHAMELEON', 
        # data_root='./datasets/COD10KCAM', 
        # data_root='./datasets/KvasirSEG',
        # class_suffix=None,
        # seg_map_suffix = '.jpg', 
        truncate_ratio=None,
        transform_cfg_args=dict(
            
            train_pipeline=[
                dict(
                    transform_type='ToTensor', 
                ), 
                dict(
                    transform_type='RandomResize',
                    scale=(384, 384),
                    ratio_range=(1, 1.333),
                    keep_ratio=False,
                    resize_mask=True
                ), 
                dict(
                    transform_type='RandomCrop',
                    size=384
                ), 
                dict(
                    transform_type='RandomHorizontalFlip',
                    prob=0.5
                ), 
                dict(
                    transform_type='Normalize',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ], 
            validate_pipeline=[
                dict(
                    transform_type='ToTensor',
                ), 
                dict(
                    transform_type='Resize',
                    scale=(512, 384),
                    keep_ratio=True,
                    resize_mask=True
                ), 
                # dict(
                #     transform_type='Resize',
                #     scale=(384, 384),
                #     keep_ratio=False,
                #     resize_mask=True
                # ), 
                dict(
                    transform_type='CenterCrop',
                    size=384
                ), 
                dict(
                    transform_type='Normalize',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
            
            
        ),
    )
)


epochs = 40
Scheduler_cfg = dict(
    scheduler_cfg_args=dict(
        device='cuda',
        seed=3407,
        
        num_workers=0,
        
        optimizer_name='AdamW',
        optimizer_args=dict(
                    lr=0.00015,
                    weight_decay=1e-4,
                ),
        
        lr_scheduler_name= 'CosineAnnealingLR_Warmup',
        lr_scheduler_args= dict(
                    epochs=epochs,     
                    warmup_epochs=10,
                    warmup_factor=1e-3, 
                    end_factor=1e-6
                ),
        
        batch_size=10,
        epochs=epochs,
        eval_interval=2,
        amp = True,
   
        metrics_cfg=[
            dict(metric_type='MAE', 
                 resize_logits=True, 
                 name='MAE'),
            
            # dict(metric_type='Fmeasure', 
            #      resize_logits=True, 
            #      name='adp_F', 
            #      metric_args=dict(
            #         beta_sq=0.3, 
            #         mode='adaptive')),
            
            dict(metric_type='WeightedFmeasure', 
                 resize_logits=True, 
                 name='Weighted_F', 
                 metric_args=dict(
                     beta_sq=0.3)),
            
            dict(metric_type='Smeasure', 
                 resize_logits=True, 
                 name='S_measure', 
                 metric_args=dict(
                     alpha=0.5)),
            
            dict(metric_type='Emeasure', 
                 resize_logits=True, 
                 name='E_measure', 
                 metric_args=dict(
                     mode='mean')), 
            
            
            # dict(metric_type='Fmeasure', 
            #      resize_logits=True, 
            #      name='max_F', 
            #      metric_args=dict(
            #         beta_sq=0.3, 
            #         mode='max')),
            
            # dict(metric_type='Emeasure', 
            #      resize_logits=True, 
            #      name='max_Emeasure', 
            #      metric_args=dict(
            #          mode='max')),
        ]
    )
)

Model_cfg = dict(
        model_cfg_name="CAMF_Config",
        model_cfg_args=dict(
            
            pretrained_weights = '/root/WeLi/model_weights/eva02_L_pt_m38m_p14to16.pth',
            finetune_weights = '/root/WeLi/work_dir/camf_duts.pth',
            # finetune_weights = '/root/WeLi/work_dir/camf_cod.pth',
            
 
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
                        adapter_layernorm_option=None,
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
                    prompt_in_len = 32,
                    tm_channels = 256,
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
                ), 
                
                # dict(
                #     loss_type="UALLoss",
                #     reduction="mean",
                #     loss_weight=1.0,
                #     loss_name='maskreg_loss_ual'
                # ),
                
                dict(
                    loss_type='CosSimilarityLoss',
                    reduction="mean",
                    loss_weight=0.5,
                    loss_name="alg_loss_cos"
                )
            ]
        )
    )


































