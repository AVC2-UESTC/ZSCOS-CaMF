import math
import json
import warnings

from typing import Dict

import numpy as np

import torch
from torch.nn import functional as F
import train_utils.distributed_utils as utils


try:
    import wandb
except ImportError:
    warnings.warn("wandb not installed")






def tensor_dict_to_device(tensor_dict: Dict[str, torch.Tensor], device: str):
    
    if isinstance(tensor_dict, dict):
        
        for key, value in tensor_dict.items():
            if value is not None:
                # check whether list
                if isinstance(value, list):
                    # tensor_dict[key] = [i.to(device) for i in value]
                    tensor_dict[key] = [i.to(device) if isinstance(i, torch.Tensor) else i for i in value]
                elif isinstance(value, str):
                    tensor_dict[key] = value
                else:
                    tensor_dict[key] = value.to(device)
            else:
                tensor_dict[key] = None
                
    else:
        tensor_dict = tensor_dict.to(device)
    
    return tensor_dict


def tensor_dict_to_ndarray_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:

    ndarray_dict = {}

    for key, value in tensor_dict.items():
        if value is not None:
            # check whether list
            if isinstance(value, list):
                ndarray_dict[key] = [i.cpu().numpy() if isinstance(i, torch.Tensor) else i for i in value]
            else:
                ndarray_dict[key] = value.cpu().numpy()
        else:
            ndarray_dict[key] = None

    return ndarray_dict


def log_vis_predictions(inputs: Dict, targets: Dict, prediction: Dict, vis_table, mode: str):
    log_inputs = tensor_dict_to_ndarray_dict(inputs) # dict(image=..., ...)
    log_targets = tensor_dict_to_ndarray_dict(targets)
    log_prediction = tensor_dict_to_ndarray_dict(prediction)
    
    if mode == 'All':
        tag_values = list(log_inputs.values()) + list(log_targets.values()) + list(log_prediction.values())
    elif mode == 'in&pred':
        tag_values = list(log_inputs.values()) + list(log_prediction.values())
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
    for i_tag_value in zip(*tag_values):
        # wb_image_list = [wandb.Image(i.transpose((1, 2, 0))) if len(i.shape) == 3 else i for i in i_tag_value]
        wb_list = []
        for i in i_tag_value:
            if isinstance(i, np.ndarray):
                if len(i.shape) == 3:
                    wb_list.append(wandb.Image(i.transpose((1, 2, 0))))
                else:
                    wb_list.append(i)
            elif isinstance(i, str):
                wb_list.append(i)
            else:
                wb_list.append(i)
        
        
        vis_table.add_data(*wb_list)
    




def evaluate_old(model, data_loader, device, metrics: dict):
    model.eval()
    # mae_metric = utils.MeanAbsoluteError()
    # f1_metric = utils.F1Score()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for inputs, target in metric_logger.log_every(data_loader, 500, header):
            
            
            # to device 
            inputs = tensor_dict_to_device(inputs, device)
            target = tensor_dict_to_device(target, device)
            
            output = model.logits(inputs)
            output = torch.sigmoid(output)
            
            mask_label = target['seg_maps']
            
            # post norm
            # ma = torch.max(output)
            # mi = torch.min(output)
            # output = (output - mi) / (ma - mi)

            # mae_metric.update(output, targets)
            # f1_metric.update(output, targets)
            
            for metric in metrics.values():
                metric.update(output, mask_label)

        

    return metrics


def evaluate(model: torch.nn.Module, 
             data_loader, 
             device: str, 
             epoch: int, 
             metrics: dict, 
             logger_name: str,
             logger,
             vis_args: Dict=None,
             ):
    model.eval()
    
    if logger_name == 'default':
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for inputs, targets in metric_logger.log_every(data_loader, 500, header):
                
                # to device 
                inputs = tensor_dict_to_device(inputs, device)
                targets = tensor_dict_to_device(targets, device)
                
                # output = model.logits(inputs)
                # output = torch.sigmoid(output)
                _, preds = model.predict(inputs, return_logits=True)
                
                
                for metric in metrics.values():
                    metric.update(preds, targets)
                    
        metric_info_dict = dict()
        for metric_name, metric in metrics.items():
            metric_info_dict[metric_name] = metric.compute()
        print(f'[epoch: {epoch}] Validation Metrics: {metric_info_dict}')

    
    elif logger_name == 'wandb':
        
        if vis_args is not None:
            vis_table = vis_args['table']
            vis_num_batch = vis_args['num_batch']
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                
                # to device 
                inputs = tensor_dict_to_device(inputs, device)
                targets = tensor_dict_to_device(targets, device)
                
                prediction, preds = model.predict(inputs, return_logits=True)
                
                if vis_args is not None and batch_idx < vis_num_batch:
                    log_vis_predictions(inputs=inputs, targets=targets, prediction=prediction, vis_table=vis_table, mode=vis_args['vis_pred'])
                
                for metric in metrics.values():
                    metric.update(preds, targets)
                    
        metric_info_dict = dict()
        for metric_name, metric in metrics.items():
            metric_info_dict[metric_name] = metric.compute()
        print(f'[epoch: {epoch}] Validation Metrics: {metric_info_dict}')
        
     
        
        logger_metric_info_dict = dict()
        for metric_name, metric in metric_info_dict.items():
            logger_metric_info_dict['val_metric/'+metric_name] = metric
        logger_metric_info_dict['val_metric/epoch'] = epoch
        
        # logger.log(dict(val=logger_metric_info_dict))
        logger.log(logger_metric_info_dict)
        
        if vis_args is not None:
            logger.log(dict(visualization_prediction=vis_table))

    return metric_info_dict



def train_one_epoch_old(model, optimizer, data_loader, device, epoch, lr_scheduler, metrics: dict, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
  

    for inputs, target in metric_logger.log_every(data_loader, print_freq, header):
        # image, prompt = inputs['image'].to(device), inputs['prompt'].to(device)
        # target = target.to(device)
        # inputs = {'image': image, 'prompt': prompt}
        
        # to device 
        inputs = tensor_dict_to_device(inputs, device)
        target = tensor_dict_to_device(target, device)
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # output = model(image)
            # loss = criterion(output, target)
            losses, logits = model.loss(inputs, target, return_logits=True)
            mask_label = target['seg_maps']
            # acc_seg = losses.pop('acc_seg')
            loss = sum(losses.values())
            
            # mae_metric.update(logits, target)
            # f1_metric.update(logits, target)
            
            
            for metric in metrics.values():
                metric.update(logits, mask_label)  
            

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)
        
    # print(acc_seg) # temp statement
    # mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
    # print(f"train_MAE: {mae_info:.3f} train_maxF1: {f1_info:.3f}")
    
    
    
    metric_info_dict = dict()
    for metric_name, metric in metrics.items():
        metric_info_dict[metric_name] = metric.compute()
    print(f'Training Metrics: {metric_info_dict}')

    return metric_logger.meters["loss"].global_avg, lr



def train_one_epoch(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    data_loader, 
                    device: str, 
                    epoch: int, 
                    lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 
                    metrics: Dict, 
                    logger_name: str,
                    logger,
                    print_freq: int=10, 
                    scaler=None):
    model.train()
    
    if logger_name == 'default':
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
    
        for inputs, targets in metric_logger.log_every(data_loader, print_freq, header):
            # image, prompt = inputs['image'].to(device), inputs['prompt'].to(device)
            # target = target.to(device)
            # inputs = {'image': image, 'prompt': prompt}
            
            # to device 
            inputs = tensor_dict_to_device(inputs, device)
            targets = tensor_dict_to_device(targets, device)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
               
                losses, preds = model.loss(inputs, targets, return_logits=True)
                loss = sum(losses.values())
                
                for metric in metrics.values():
                    metric.update(preds, targets)  
                

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(loss=loss.item(), lr=lr)
            
        metric_info_dict = dict()
        for metric_name, metric in metrics.items():
            metric_info_dict[metric_name] = metric.compute()
        print(f'Training Metrics: {metric_info_dict}')
        
        avg_loss = metric_logger.meters["loss"].global_avg
        
    # print(acc_seg) # temp statement
    # mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
    # print(f"train_MAE: {mae_info:.3f} train_maxF1: {f1_info:.3f}")
    
    elif logger_name == 'wandb':
        loss_list = []
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = tensor_dict_to_device(inputs, device)
            targets = tensor_dict_to_device(targets, device)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                losses, preds = model.loss(inputs, targets, return_logits=True)
                
                loss = sum(losses.values())
                
                for metric in metrics.values():
                    metric.update(preds, targets)
                    
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            
            lr = optimizer.param_groups[0]["lr"]
            
            if batch_idx % print_freq == 0:
                # print('Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),batch_idx / len(train_loader), loss.item()))
                print(f'Epoch: {epoch} [{batch_idx}/{len(data_loader)}]\tLoss: {loss.item():.4f}\tLR: {lr:.6f}')
                
            loss_list.append(loss.detach().item())    
            training_log = {
                'learning_rate': lr,
                "train_loss": loss.detach().item(),
            }
            logger.log(dict(train=training_log))

        metric_info_dict = dict()
        for metric_name, metric in metrics.items():
            metric_info_dict[metric_name] = metric.compute()
        print(f'Training Metrics: {metric_info_dict}')
        
        
        logger_metric_info_dict = dict()
        for metric_name, metric in metric_info_dict.items():
            logger_metric_info_dict['train_metric/' + metric_name] = metric
        logger_metric_info_dict['train_metric/epoch'] = epoch
        
        logger.log(logger_metric_info_dict)
        
        avg_loss = sum(loss_list) / len(loss_list)
        
        
    return avg_loss, lr








# The implementation code is modified from Timm (https://github.com/huggingface/pytorch-image-models/tree/main/timm
def get_parameter_groups(model: torch.nn.Module, 
                         weight_decay: float=1e-5, 
                         skip_list=(), 
                         get_num_layer=None, 
                         get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())



