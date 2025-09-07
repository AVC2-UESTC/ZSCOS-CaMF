


Dataset_cfg = dict(
    dataset_cfg_name='Cityscapes_SemSeg_Dataset_Config',
    dataset_cfg_args=dict(
        data_root=None,
        truncate_ratio=None,
        
        
        transform_cfg_name='Semantic_Segmentation_Transform_Config',
        transform_cfg_args=dict(
            
            train_pipeline=[
                dict(
                    transform_type='ToTensor', 
                    num_classes = 151,
                ), 
                dict(
                    transform_type='RandomResize',
                    scale=(518, 518),
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
                    num_classes = 151,
                ), 
                dict(
                    transform_type='Resize',
                    scale=(682, 518),
                    keep_ratio=True,
                    resize_mask=True
                ), 
                dict(
                    transform_type='CenterCrop',
                    size=518,
                ), 
                dict(
                    transform_type='Normalize',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
            
            
        )
    )
)




















