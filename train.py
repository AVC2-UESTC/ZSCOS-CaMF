import os
import warnings
import time
import datetime

from typing import Union, List, Dict
from collections import OrderedDict

import torch
from torch.utils import data

from train_utils import get_parameter_groups
from train_utils.train_and_eval import train_one_epoch, evaluate, tensor_dict_to_device


try:
    import wandb
except ImportError:
    warnings.warn("wandb not installed")
    


# from src.Models.builder import build_metric
from mmengine.config import Config as MMConfig

from src.builder import build_dataset, build_model, build_scheduler, create_lr_scheduler

def main(scheduler_cfg, dataset_cfg, model_cfg, runtime: Dict):
    # print(scheduler_cfg)
    # print(dataset_cfg)
    # print(model_cfg)
    
    logger_name = runtime.get('logger_name')
    logger_args = runtime.get('logger_args')
    
    
    if scheduler_cfg.seed:
        torch.manual_seed(scheduler_cfg.seed)
        
    device = torch.device(scheduler_cfg.device if torch.cuda.is_available() else "cpu")
    print(f'Using {device} for training')
    
    batch_size = scheduler_cfg.batch_size

    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    #load dataset cfg
    # instantiate dataset
    train_dataset = dataset_cfg.dataset_train
    val_dataset = dataset_cfg.dataset_val
    
    num_workers = scheduler_cfg.num_workers
    
    co_fn_train = getattr(train_dataset, 'collate_fn', None)
    co_fn_val = getattr(val_dataset, 'collate_fn', None)

    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=False,
                                        collate_fn=co_fn_train
                                        )

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
        if model_cfg.tuning_mode == 'PEFT':
            save_weights_flag = 'Partial'
        elif model_cfg.tuning_mode == 'Full':
            save_weights_flag = 'All'
    else:
        save_weights_flag = 'All'
            
    
    # get skip weight decay list
    no_wd_method = getattr(model, 'no_weight_decay', None)
    if no_wd_method is not None:
        skip_weight_decay_list = no_wd_method()
    else:
        skip_weight_decay_list = []
    

    # params_group = get_params_groups(model, weight_decay=scheduler_args.weight_decay)
    wd = scheduler_cfg.optimizer_args.weight_decay
    params_group = get_parameter_groups(model, weight_decay=wd, skip_list=skip_weight_decay_list)
    optimizer = scheduler_cfg.create_optimizer(params_group)
    
    lr_scheduler_args = scheduler_cfg.lr_scheduler_args
    lr_scheduler_args['dataset_len'] = len(train_data_loader)
    lr_scheduler = create_lr_scheduler(optimizer, scheduler_cfg.lr_scheduler_name, args=lr_scheduler_args)

    scaler = torch.cuda.amp.GradScaler() if scheduler_cfg.amp else None
    
    
    # ======================= Resume ==============================
    
    if scheduler_cfg.resume:
        print(f'Resume from {scheduler_cfg.resume}')
        checkpoint = torch.load(scheduler_cfg.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        scheduler_cfg.start_epoch = checkpoint['epoch'] + 1
        if scheduler_cfg.amp:
            scaler.load_state_dict(checkpoint["scaler"])
    
    
    model.to(device)
    
    metric_dict = scheduler_cfg.get_metric_dict
    
    
    # ===================== Runtime setting =========================
    # dataset_name_list = dataset_cfg.data_root 
    # if isinstance(dataset_name_list, list):
    #     dataset_name = ''
    #     for root in dataset_name:
    #         name = os.path.basename(root)
    #         dataset_name = dataset_name + '+' + name
    # else:
    #     dataset_name = os.path.basename(dataset_name_list)
    
    if logger_name == 'wandb':
        print('Using wandb.')
        os.environ['WANDB_API_KEY'] = logger_args['api_key']
        wandb.login()
        
        logger = wandb.init(
            project=logger_args['project'],
            config={
                'learning_rate': scheduler_cfg.optimizer_args.lr,
                'epochs': scheduler_cfg.epochs,
                # 'dataset': dataset_name,
                'model': model_cfg.__class__.__name__,
            }
        )
        
        logger.define_metric("train_metric/epoch")
        logger.define_metric("train_metric/*", step_metric="train_metric/epoch")
        
        logger.define_metric("val_metric/epoch")
        logger.define_metric("val_metric/*", step_metric="val_metric/epoch")
        
        wandb.watch(model)
        
        vis_args = dict(vis_pred=logger_args.get('vis_pred'), 
                        num_batch=logger_args.get('vis_num_batch', None),)
        
    elif logger_name == 'default':
        print('Using default logger.')
        logger = None
    else:
        raise ValueError(f"Unsupported logger: {logger}")
        
    
    
    # ======================= Begin training =============================
    
    start_time = time.time()
    
    for epoch in range(scheduler_cfg.start_epoch, scheduler_cfg.epochs):
        
        # visualization setting
        if logger_name == 'wandb':
            if logger_args.get('vis_pred') == 'in&pred':
                (inputs_0, targets_0) = next(iter(train_data_loader))
                model.eval()
                with torch.no_grad():
                    inputs_0 = tensor_dict_to_device(inputs_0, device)
                    model_out = model.predict(inputs_0, return_logits=False)
                
                tags = list(inputs_0.keys()) + list(model_out.keys())
                vis_table = wandb.Table(columns=tags)
                
                vis_args['table'] = vis_table
                
            elif logger_args.get('vis_pred') == 'All':
                (inputs_0, targets_0) = next(iter(train_data_loader))
                model.eval()
                with torch.no_grad():
                    inputs_0 = tensor_dict_to_device(inputs_0, device)
                    model_out = model.predict(inputs_0, return_logits=False)
                
                tags = list(inputs_0.keys()) + list(targets_0.keys()) + list(model_out.keys())
                vis_table = wandb.Table(columns=tags)
                
                vis_args['table'] = vis_table
            else:
                vis_args = None
                
        else:
            vis_args = None
        
        
        
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch, 
                                        metrics=metric_dict,
                                        logger_name=logger_name,
                                        logger=logger,
                                        lr_scheduler=lr_scheduler, print_freq=scheduler_cfg.print_freq, scaler=scaler)

        
        save_weights_dict = model.state_dict()
        for name, p in model.named_parameters():
            if name not in save_weights_keys:
                save_weights_dict.pop(name)
                
        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": scheduler_cfg}
        if scheduler_cfg.amp:
            save_file["scaler"] = scaler.state_dict()
            
        # save_ft_file = model.state_dict() 
        # save_weights_file = model.state_dict()

        # validate
        
        if epoch % scheduler_cfg.eval_interval == 0 or epoch == scheduler_cfg.epochs - 1:
            # 每间隔eval_interval个epoch验证一次，减少验证频率节省训练时间
            metric_info_dict = evaluate(model, val_data_loader, 
                                        device=device, 
                                        epoch=epoch, 
                                        metrics=metric_dict, 
                                        logger_name=logger_name, 
                                        logger=logger, 
                                        vis_args=vis_args,
                                        
                                        )
            
            
            # save results
            # write into txt
            with open(results_file, "a") as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"Val_Metrics: {metric_info_dict} \n"
                f.write(write_info)
            
        # only save latest 10 epoch weights
        if os.path.exists(f"work_dir/save_models/model_{epoch-10}.pth"):
            os.remove(f"work_dir/save_models/model_{epoch-10}.pth")
        if os.path.exists(f'work_dir/save_models/model_Resume_{epoch-10}.pth'):
            os.remove(f'work_dir/save_models/model_Resume_{epoch-10}.pth')
        if os.path.exists(f"work_dir/save_finetunes/ft_{epoch-10}.pth"):
            os.remove(f"work_dir/save_finetunes/ft_{epoch-10}.pth")
            
        if save_weights_flag == 'Partial':
            torch.save(save_weights_dict, f"work_dir/save_finetunes/ft_{epoch}.pth")
            torch.save(save_file, f"work_dir/save_models/model_Resume_{epoch}.pth")
        elif save_weights_flag == 'All':
            torch.save(save_file, f"work_dir/save_models/model_{epoch}.pth")
                
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
            


if __name__ == "__main__":
    
    
    
    config = './configs/CaMP/config_CaMP_adapter_fgseg.py' # CaMP
    
   
    
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