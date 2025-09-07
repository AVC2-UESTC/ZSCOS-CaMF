import math
import time

import numpy as np
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .base_metric import BaseMetric

class Accuracy(BaseMetric):
    def __init__(self, resize_logits: bool = False):
        super().__init__(resize_logits)
    
    def update(self, preds: Dict, gt:torch.Tensor):
        '''
        Args: 
            pred: 0~1
            gt: indices
            
        Shape: 
            pred: (N, n_cls)
            gt: (N, )
        '''
        pred = preds['pred_clsprob']
        pred = pred.detach()
        gt = gt.detach()

       
            
        
        acc = self._cal_acc(pred, gt)
        self.value_list.extend(acc.tolist())
        
        
    def _cal_acc(self, pred, gt):
        pred = pred.argmax(dim=1)
        acc = (pred == gt).float()
        return acc



