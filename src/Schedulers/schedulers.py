import os
from typing import Dict
import torch

from typing import Union, List


from ..Models.builder import build_metric
# =============================================================================
# ============================ Train ==========================================

class Scheduler_Config():
    '''
    
    '''

    def __init__(self, 
                 device, 
                 seed=None,
                 num_workers: Union[str, int]=0,
                 
                 optimizer_name: str = 'AdamW',
                 optimizer_args: Dict = dict(
                    lr=0.001,
                    weight_decay=1e-4,
                ),
                 
                 lr_scheduler_name: str = 'CosineAnnealingLR',
                 lr_scheduler_args: Dict = dict(
                    T_max=100,     
                ),
                 
                 batch_size: int = 4, 

                 epochs: int = 100, 
                #  warmup_epochs:int = 1,
                 eval_interval: int = 10, 

                 print_freq: int = 50, 
                 resume = None, # file
                 start_epoch: int = 0, 
                 amp = False, 
                 metrics_cfg = None) -> None:
        #create self properties 
        assert metrics_cfg is not None, "metrics_cfg is not defined"
        self.seed = seed
        self.device = device
        
        if num_workers == 'auto':
            self.num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
        else:
            self.num_workers = num_workers
            
        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        
        self.lr_scheduler_name = lr_scheduler_name
        self.lr_scheduler_args = lr_scheduler_args
        
        self.batch_size = batch_size
        self.epochs = epochs
        # self.warmup_epochs = warmup_epochs
        self.eval_interval = eval_interval
        self.print_freq = print_freq
        self.resume = resume
        # self.fine_tune_w = fine_tune_w
        self.start_epoch = start_epoch
        self.amp = amp
        if isinstance(metrics_cfg, dict):
            self.metrics_cfg = [metrics_cfg]
        else:
            self.metrics_cfg = metrics_cfg
        
    
    @property
    def get_metric_dict(self):
        metric_dict = dict()
        for metric_cfg in self.metrics_cfg:
            metric_dict[metric_cfg['name']] = build_metric(**metric_cfg)
        
        return metric_dict
        # {metric1_name: metric1_inst, ...}
        
    
    
    def create_optimizer(self, params: List):
        
        # supported_optimizer_dict = {
        #     'SGD': torch.optim.SGD,
        #     'Adam': torch.optim.Adam,
        #     'AdamW': torch.optim.AdamW,
        #     'RMSprop': torch.optim.RMSprop,
        #     'Adamax': torch.optim.Adamax,
        #     'ASGD': torch.optim.ASGD,
        #     'LBFGS': torch.optim.LBFGS,
        #     'Rprop': torch.optim.Rprop,
        #     'SparseAdam': torch.optim.SparseAdam,
        #     'Adadelta': torch.optim.Adadelta,
        #     'Adagrad': torch.optim.Adagrad,
        # }
        
        supported_optimizer_dict = {name: value for name, value in torch.optim.__dict__.items() if isinstance(value, type) and issubclass(value, torch.optim.Optimizer)}
        
        optimizer_class = supported_optimizer_dict[self.optimizer_name]
        
        optimizer_inst = optimizer_class(params, **self.optimizer_args)
        
        
        return optimizer_inst


    # def create_lr_scheduler(self, optimizer: torch.optim.Optimizer):
        
    #     supported_lr_schedulers = {name: value for name, value in torch.optim.lr_scheduler.__dict__.items() if isinstance(value, type) and issubclass(value, torch.optim.lr_scheduler.LRScheduler)}
        
    #     if self.lr_scheduler_name=='LambdaLR':
    #         assert 'lr_lambda' in self.lr_scheduler_args.keys(), 'lr_lambda must be provided for LambdaLR'
            
    #         lr_scheduler_class = supported_lr_schedulers[self.lr_scheduler_name]
            
    #         local_lr_scheduler_funcs = 
    #         self.lr_scheduler_args.pop('lr_lambda')
            
    #         lr_scheduler_inst = lr_scheduler_class(optimizer, lr_lambda= , **self.lr_scheduler_args)
        
        
    #     lr_scheduler_class = supported_lr_schedulers[self.lr_scheduler_name]
    #     lr_scheduler_inst = lr_scheduler_class(optimizer, **self.lr_scheduler_args)
        
    #     return lr_scheduler_inst
        
        
# ======================= Config function ===========================

def scheduler_ft_0(
    device = 'cuda', 
    batch_size=20, 
    weight_dacay=1e-4, 
    epochs=20,
    warmup_epochs=3,# cannot be 0 
    eval_interval=1, 
    lr = 0.0015, 
    print_freq=50, 
    **kwargs
           
):
    
    args = Scheduler_Config(
        device=device,
        batch_size=batch_size,
        weight_decay=weight_dacay,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        eval_interval=eval_interval,
        lr=lr,
        print_freq=print_freq,
        **kwargs
    )

    return args
        

# ============================ End ==================================================
# ===================================================================================


# ===================================================================================
# ====================== inference ==================================================

class Scheduler_inference_Config():
    '''
    
    '''

    def __init__(self, 
                 device, 
                 seed=None,
                 pretrained = None, #pretrain weigthts
                 fine_tune_w = None, # fine tuning weights 
                 visualization = None,
                 image_path = None, 
                 target_path = None,
                 prompts_path = None,
                 save_path = None,
        ) -> None:
        #create self properties 
        self.seed = seed
        self.device = device
        self.pretrained = pretrained
        self.fine_tune_w = fine_tune_w
        
        self.visualization = visualization
        
        self.image_path = image_path
        self.target_path = target_path
        self.prompts_path = prompts_path
        self.save_path = save_path
        

def scheduler_inference_0_cfg(
    device = 'cuda',
    seed = None,
    pretrained = None,
    fine_tune_w = None, 
    visualization = None,
    image_path = None,
    target_path = None,
    prompts_path = None,
    save_path = None,
):
    args = Scheduler_inference_Config(
        device=device,
        seed=seed,
        pretrained=pretrained,
        fine_tune_w=fine_tune_w,
        visualization=visualization,
        image_path=image_path,
        target_path=target_path,
        prompts_path=prompts_path,
        save_path=save_path,
    )
    
    return args

# ============================ End ==================================================
# ===================================================================================



#========================debug=============================

def debug():
    # pass args to the schedule config above
    schedule_cfg = Scheduler_Config(
        data_path = None,
        device = 'cuda',
        batch_size = 16,
        weight_dacay = 0.0005,
        epochs = 100,
        eval_interval = 10,
        lr = 0.001,
        print_freq = 50,
        resume = None,
        
        start_epoch = 0,
        amp = None
    )
    print(schedule_cfg)
    
    
if __name__ == "__main__":
    debug()









