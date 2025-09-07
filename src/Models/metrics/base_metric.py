
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..utils import resize



class BaseMetric(object):
    '''
    only support single gpu
    '''

    def __init__(self, resize_logits: bool = False):
        self.resize_logits = resize_logits
        self.value_list = []
    
        
    def update(self, preds: Dict, gts: Dict) -> None:
        '''
        Calculate metric value
        Write a function that:
            add the calculated metric value of each batch to self.value_list
        
        Args:
            preds: Dict[Tensor]
            gt: same 
            
        Shape:
            preds: (N, 1, H, W)
            gts: (N, 1, H, W)
        
        '''
        raise NotImplementedError 
    
    def compute(self) -> float:
        mean_metric_value = sum(self.value_list) / len(self.value_list)
        self.reset_metric_value_list()
        return mean_metric_value
        
    def reset_metric_value_list(self) -> None:
        self.value_list = []
        
        

        


