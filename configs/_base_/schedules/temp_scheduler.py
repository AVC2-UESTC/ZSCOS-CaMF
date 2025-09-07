



Scheduler_cfg = dict(
    scheduler_cfg_name='Scheduler_Config',
    scheduler_cfg_args=dict(
        device='cuda',
        seed=3407,
        
        num_workers='auto',
        optimizer_name='AdamW',
        optimizer_args=dict(
                    lr=0.0001,
                    weight_decay=1e-4,
                ),
        
        lr_scheduler_name= 'CosineAnnealingLR',
        lr_scheduler_args= dict(
                    T_max=100,     
                ),
        
        batch_size=10,
        epochs=50,
        eval_interval=2,
        amp = False, 
        metrics_cfg=None
    )
)




