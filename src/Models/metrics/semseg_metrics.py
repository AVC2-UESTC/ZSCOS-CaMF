import math
import time

import numpy as np
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F


import mmcv 


from ..utils import resize
from .base_metric import BaseMetric


EPS = 1e-9
# functional

def is_zero(x: torch.Tensor) -> bool:
    if x.abs() < EPS:
    
        return_value = True
    else:
        return_value = False
    
    return return_value


def precision(pred, label, threshold=128) -> torch.Tensor:
    '''
    Args:
        pred: figure ranging from 0 ~ 255
        labels: label indices {0, 1}
        threshold: 0 ~ 255
    
    Return:
        float
    
    Shape: 
        pred: (1, ..., 1, H, W) or (H, W)
        label: same as above
        
    
    '''
    
    binary_segmented_map = (pred > threshold).int()
    TP = torch.sum(binary_segmented_map * label)
    FP = torch.sum(binary_segmented_map) - TP
    
    precision_value = TP / (TP + FP + EPS)
    
    return precision_value

def recall(pred, label, threshold=128) -> torch.Tensor:
    '''
    Args:
        pred: figure ranging from 0 ~ 255
        labels: label indices {0, 1}
        threshold: 0 ~ 255
    
    Return:
        float
    
    Shape: 
        pred: (1, ..., 1, H, W) or (H, W)
        label: same as above

    '''

    binary_segmented_map = (pred > threshold).int()
    TP = torch.sum(binary_segmented_map * label)
    FN = torch.sum(label) - TP
    
    recall_value = TP / (TP + FN + EPS)
    
    return recall_value


class mIOU(BaseMetric):
    def __init__(self, resize_logits: bool = False, 
                 # args
                 
                 
                 
                 ):
        super().__init__(resize_logits)

    
    
    
    def update(self, preds: Dict, gts: Dict):
        '''
        Args: 
            pred: 0~1
            gt: indices
            
        Shape: 
            pred: (N, n_cls, H, W)
            gt: (N, n_cls, H, W)
        '''
        pred = preds['pred_mask']
        gt = gts['label_mask']
        pred = pred.detach()
        gt = gt.detach()
        batch_size, c, h, w = gt.shape

        if self.resize_logits:
            pred = resize(
                input=pred,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
        
        miou = self._cal_miou(pred, gt)
        self.value_list.extend(miou.tolist())
        
    def _cal_miou(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        '''
        Args: 
            pred: 0~1
            gt: 

        Shape: 
            pred: (N, n_cls, H, W)
            gt: (N, n_cls, H, W)
        '''
        
        intersection = pred * gt
    
        miou = (intersection.sum(dim=(1, 2, 3)) + EPS) / ((pred + gt - intersection).sum(dim=(1, 2, 3)) + EPS)
        # (b, )
        
        return miou


