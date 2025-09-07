

Dataset_cfg = dict(
    dataset_cfg_name='ContourPrediction_Dataset_Config',
    dataset_cfg_args=dict(
        data_root=None,
        truncate_ratio=None,
        
        
        transform_cfg_name='ContourPrediction_Transform_Config',
        transform_cfg_args=dict(
            
            train_pipeline=[
                dict(
                    transform_type='ToTensor', 
                ), 
                dict(
                    transform_type='Resize',
                    scale=(384, 384),
                    keep_ratio=False,
                ), 
                
                dict(
                    transform_type='RandomHorizontalFlip',
                    flip_prob=0.5
                ), 
                dict(
                    transform_type='Normalize_Fourier',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225), 
                    M=257,
                )
            ], 
            validate_pipeline=[
                dict(
                    transform_type='ToTensor',
                ), 
                dict(
                    transform_type='Resize',
                    scale=(384, 384),
                    keep_ratio=False,
                ), 
                
                dict(
                    transform_type='Normalize_Fourier',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225), 
                    M=257,
                )
            ]
            
            
        )
    )
)

