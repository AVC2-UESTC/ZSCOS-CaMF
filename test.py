import os
import warnings
import time
import datetime

from typing import Union, List, Dict
from collections import OrderedDict

import torch
from torch.utils import data

from train_utils import get_parameter_groups
from train_utils.train_and_eval import train_one_epoch, evaluate


# try:
#     import wandb
# except ImportError:
#     warnings.warn("wandb not installed")
    # print("wandb not installed")
    


# from src.Models.builder import build_metric
from mmengine.config import Config as MMConfig

from src.builder import build_dataset, build_model, build_scheduler

def main(scheduler_cfg, dataset_cfg, model_cfg, runtime: Dict):
    # print(scheduler_cfg)
    # print(dataset_cfg)
    # print(model_cfg)
    
    # logger_name = runtime.get('logger_name')
    logger_name = 'default'
    logger_args = None
    
    
    if scheduler_cfg.seed:
        torch.manual_seed(scheduler_cfg.seed)
        
    device = torch.device(scheduler_cfg.device if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for testing')
    
    batch_size = 1

    results_file = "test_results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    #load dataset cfg
    # instantiate dataset
    # train_dataset = dataset_cfg.dataset_train
    val_dataset = dataset_cfg.dataset_val
    
    num_workers = scheduler_cfg.num_workers
    
    co_fn_val = getattr(val_dataset, 'collate_fn', None)

   

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=False,
                                      collate_fn=co_fn_val
                                      )
    
    # ===================== model define =================================
   
    model, save_weights_keys = model_cfg.model
    # print(model)
    
    if model_cfg.pretrained_weights is not None:
        print('Weights loaded')
    else:
        warnings.warn('No pretrained weights are loaded.')
            
    
    
    
    
    model.to(device)
    
    metric_dict = scheduler_cfg.get_metric_dict
    
    
    if logger_name == 'default':
        print('Using default logger.')
        logger = None
    else:
        raise ValueError(f"Unsupported logger: {logger}")
        
    
    
    # ======================= Begin training =============================
    
    start_time = time.time()
    
    metric_info_dict = evaluate(model, val_data_loader, 
                                device=device, 
                                epoch=0, 
                                metrics=metric_dict, 
                                logger_name=logger_name, 
                                logger=logger)
        
        
        # save results
        # write into txt
    with open(results_file, "a") as f:
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        write_info = f"[epoch: {0}] Val_Metrics: {metric_info_dict} \n"
                        
        f.write(write_info)
        
        
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Test time usage: {}".format(total_time_str))
            


if __name__ == "__main__":
    
    
    config = './configs/CAMF/config_CAMF_fgseg.py' # CAMF    
    
    
    assert os.path.exists(config), f"No such file: {config}"

    config = MMConfig.fromfile(config)
    
    # args = cfg.cfg_segformer_sod() # load config file
    Scheduler_cfg = config.get("Scheduler_cfg")
    Dataset_cfg = config.get("Dataset_cfg")
    Model_cfg = config.get("Model_cfg")
    runtime = config.get("runtime")
    # dict
    
    scheduler_cfg_inst = build_scheduler(scheduler_cfg_name=Scheduler_cfg['scheduler_cfg_name'], 
                                         scheduler_cfg_args=Scheduler_cfg['scheduler_cfg_args'])
    dataset_cfg_inst = build_dataset(dataset_cfg_name=Dataset_cfg['dataset_cfg_name'],
                                     dataset_cfg_args=Dataset_cfg['dataset_cfg_args'])
    model_cfg_inst = build_model(model_cfg_name=Model_cfg['model_cfg_name'],
                                 model_cfg_args=Model_cfg['model_cfg_args'])
    
    
    if not os.path.exists("./work_dir"):
        os.mkdir("./work_dir")
        os.mkdir("./work_dir/save_models")
        os.mkdir("./work_dir/save_finetunes")
    
    main(scheduler_cfg_inst, dataset_cfg_inst, model_cfg_inst, runtime)