
from typing import Optional


import torch
import torch.nn as nn
from torch import Tensor









from . import losses
def build_loss(loss_type: str, 
               weight: Optional[Tensor] = None, 
               reduction: str = 'mean',
               loss_name: str = None,
               loss_weight : float = 1.0,
               
               loss_args = None
               ) -> nn.Module:
    '''
    build loss layer
    
    
    
    loss config:

        loss_type (str)
        
        weight: Optional[Tensor] = None, 
        reduction: str = 'mean',
        
        #ce, bce
        pos_weight: Optional[Tensor] = None
        
        # ce
        ignore_index: int = -100, 
        label_smoothing: float = 0.0
    
    '''
    supported_losses = [item for item in dir(losses) if not (item.startswith("__") and item.endswith("__"))]
    assert loss_type in supported_losses, f'Unsupported loss type: {loss_type}, supported types are: {supported_losses}. Please add your own loss class in "src/Models/losses/__init__.py".'
    
    
    # supported_loss_types  = {
    #     'CrossEntropyLoss': CrossEntropyLoss,
    #     'BCEWithLogitsLoss': BCEWithLogitsLoss,
    #     'MSELoss': nn.MSELoss,
    # }
    
    loss_class = getattr(losses, loss_type)
    
    # build 
    if loss_args is None:
        layer = loss_class(weight = weight, reduction = reduction, loss_weight = loss_weight, loss_name=loss_name)
    else:
        layer = loss_class(weight = weight,reduction = reduction, loss_weight = loss_weight, loss_name=loss_name, **loss_args)
    
    
    
    # if loss_name == None:
    #     layer = supported_loss_types[loss_type](weight = weight,
    #                                             reduction = reduction,
    #                                             loss_weight = loss_weight,
    #                                             layer_args = layer_args)
    # else:
    #     layer = supported_loss_types[loss_type](weight = weight,
    #                                             reduction = reduction,
    #                                             loss_name = loss_name, 
    #                                             loss_weight = loss_weight,
    #                                             layer_args = layer_args)
    
    return layer

# from .metrics import *

# def build_metric(metric_type, 
#                  resize_logits, 
#                  name=None,
#                  metric_args = None):
    
#     supported_metrics = {'MAE': MAE, 
#                          'Fmeasure':Fmeasure, 
#                          'WeightedFmeasure':WeightedFmeasure,
#                          'Smeasure':Smeasure, 
#                          'Emeasure': Emeasure
#     }

#     # build metric
#     if metric_args is None:
#         metric = supported_metrics[metric_type](resize_logits=resize_logits)
#     else:
#         metric = supported_metrics[metric_type](resize_logits=resize_logits, **metric_args)
    
#     return metric

from . import metrics
def build_metric(metric_type, 
                 resize_logits, 
                 name=None,# do not remove this parameter
                 metric_args = None):
    
    supported_metrics = [item for item in dir(metrics) if not (item.startswith("__") and item.endswith("__"))]
    
    assert metric_type in supported_metrics, f'Unsupported metric type: {metric_type}, supported types are: {supported_metrics}. Please add your own metric class in "src/Models/metrics/__init__.py".'
    
    metric_type = 'metrics.' + metric_type
    
    metric_class = eval(metric_type)

    # build metric
    if metric_args is None:
        metric_instance = metric_class(resize_logits=resize_logits)
    else:
        metric_instance = metric_class(resize_logits=resize_logits, **metric_args)
    
    return metric_instance





#=========================debug================================

def debug():
    torch.manual_seed(1)
    input = torch.rand(3)
    # input = input.unsqueeze(1)
    loss_cfg = {
        'loss_type': 'BCEWithLogitsLoss',
        'weight': None,
        'reduction': 'none',
        
        
    }
    test_layer = build_loss(**loss_cfg)
    output = test_layer(input, torch.tensor([0.1, 0.2, 0.7]))
    print(output)
    
    
    
if __name__ == '__main__':
    debug()












