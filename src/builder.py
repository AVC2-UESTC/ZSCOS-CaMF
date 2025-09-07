
from typing import Dict

import torch





from . import Datasets
# from torchvision.datasets import ImageFolder
def build_dataset(dataset_cfg_name: str, 
                  dataset_cfg_args: dict):
    
    supported_datasets = [item for item in dir(Datasets) if not (item.startswith("__") and item.endswith("__"))]
    # supported_datasets.append(ImageFolder)
    assert dataset_cfg_name in supported_datasets, f'Unsupported dataset: {dataset_cfg_name}, supported datasets are: {supported_datasets}. Please add your own dataset class in "src/Datasets/__init__.py".'
    
    assert dataset_cfg_args is not None, 'dataset_cfg_args cannot be None.'
    
    
    # dataset_cfg_name = 'Datasets.' + dataset_cfg_name
    # dataset_cfg_class = eval(dataset_cfg_name)
    dataset_cfg_class = getattr(Datasets, dataset_cfg_name)
    
    dataset_cfg_instance = dataset_cfg_class(**dataset_cfg_args)
    
    return dataset_cfg_instance
    
    

from . import Schedulers
def build_scheduler(scheduler_cfg_name: str, scheduler_cfg_args: dict):
    
    supported_schedulers = [item for item in dir(Schedulers) if not (item.startswith("__") and item.endswith("__"))]
    
    assert scheduler_cfg_name in supported_schedulers, f'Unsupported scheduler: {scheduler_cfg_name}, supported schedulers are: {supported_schedulers}. Please add your own scheduler class in "src/schedulers.py".'
    
    assert scheduler_cfg_args is not None, 'scheduler_cfg_args cannot be None.'
    
    scheduler_cfg_class = getattr(Schedulers, scheduler_cfg_name)
    
    scheduler_cfg_instance = scheduler_cfg_class(**scheduler_cfg_args)
    
    return scheduler_cfg_instance
    
    
from .Schedulers import lr_schedulers
def create_lr_scheduler(optimizer: torch.optim.Optimizer, 
                        lr_scheduler_name: str, 
                        #
                        
                        args:Dict
                        
                        ) -> torch.optim.lr_scheduler.LRScheduler:
    torch_lr_schedulers = {name: value for name, value in torch.optim.lr_scheduler.__dict__.items() if isinstance(value, type) and issubclass(value, torch.optim.lr_scheduler.LRScheduler)}
    
    local_lr_schedulers = {name: value for name, value in lr_schedulers.__dict__.items() if isinstance(value, type)}
    
    if lr_scheduler_name in local_lr_schedulers.keys():
        lr_scheduler_class_inst = local_lr_schedulers[lr_scheduler_name](optimizer, args=args)
        lr_scheduler_inst = lr_scheduler_class_inst.get_lr_scheduler()
        
    elif lr_scheduler_name in torch_lr_schedulers.keys():
        lr_scheduler_class = torch_lr_schedulers[lr_scheduler_name]
        lr_scheduler_inst = lr_scheduler_class(optimizer, **args)
    else:
        raise NotImplementedError(f'Unsupported lr_scheduler: {lr_scheduler_name}')
    
    return lr_scheduler_inst
    
    
    
    
    
from . import Models
def build_model(model_cfg_name: str, model_cfg_args: dict):
    supported_models = [item for item in dir(Models) if item.endswith("_Config")]
    
    assert model_cfg_name in supported_models, f'Unsupported model: {model_cfg_name}, supported models are: {supported_models}. Please add your own model class in "src/Models/[specific model folder, i.e. segmentors]".'
    
    assert model_cfg_args is not None, 'model_cfg_args cannot be None.'
    
    model_cfg_class = getattr(Models, model_cfg_name)
    
    model_cfg_instance = model_cfg_class(**model_cfg_args)
    
    return model_cfg_instance



























