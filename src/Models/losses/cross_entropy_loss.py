# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Union, List


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from .base_loss import BaseLoss
from .utils import weight_reduce_loss





# ==============================================================================
# ================================== func ======================================



def wbce_loss(pred: Tensor, target: Tensor, 
              
              weight = None,
              reduction = 'mean',
              avg_factor=None,
              
              gamma: int = 5,
              surround_size: int = 31,
              
              ):
    '''
    Args:
        pred: (N, 1, H, W)
        target: (N, H, W) or (N, 1, H, W)

    '''

    if len(target.shape) == 3:
            target = target.unsqueeze(1)
            
    padding = (surround_size - 1) // 2
    
    weit = 1 + gamma*torch.abs(F.avg_pool2d(target, kernel_size=surround_size, stride=1, padding=padding) - target)
    wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    wbce = (weit*wbce).sum(dim=(1, 2, 3)) / weit.sum(dim=(1, 2, 3))

    loss = weight_reduce_loss(wbce, weight, reduction=reduction, avg_factor=avg_factor)
    
    return loss









# ==================================== End =========================================
# ==================================================================================







#wrapper
class BCEWithLogitsLoss(nn.Module):
    '''
    Args:
        weight
        
        reduction: 'none', 'mean', 'sum'
        
        layer_args: 
            - pos_weight (Tensor):
            
            
    Shape:  
        input: (*) any shape. 
        target: same shape as input. Values between 0 and 1
            (N, H, W) is also ok
    '''
    def __init__(self, 
                 weight: Optional[Tensor] = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_bce',
                 loss_weight: float = 1.0,
                 
                 # loss_args
                 pos_weight = None) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.weight = weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
    
    def forward(self, pred: Union[Tensor, List[Tensor]], target: Union[Tensor, List[Tensor]], avg_factor=None) -> Tensor:
        
        if isinstance(pred, list) and isinstance(target, list):
            # (N, num_masks, H, W)
            bs = len(pred)
            loss = torch.zeros(bs)
            for b in range(bs):
                one_target = target[b].unsqueeze(1)
                one_pred = pred[b].unsqueeze(1)
                one_pred = one_pred.float()
                one_target = one_target.float()
                
                one_loss = F.binary_cross_entropy_with_logits(one_pred, one_target, 
                                                        weight=self.weight, 
                                                        reduction='mean',
                                                        pos_weight=self.pos_weight)
                loss[b] = one_loss
            
            loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction, avg_factor=avg_factor)
                
                
        elif isinstance(pred, Tensor) and isinstance(target, Tensor):
            pred = pred.float()
            target = target.float()
            if target.dim() == 3: #(N, H, W)
                target = target.unsqueeze(1)
                loss = F.binary_cross_entropy_with_logits(pred, target, 
                                                        weight=self.weight, 
                                                        reduction=self.reduction,
                                                        pos_weight=self.pos_weight)
            else:
                loss = F.binary_cross_entropy_with_logits(pred, target, 
                                                        weight=self.weight, 
                                                        reduction=self.reduction,
                                                        pos_weight=self.pos_weight)
        else:
            raise TypeError(f'input and target must be Tensor or List[Tensor]')
            
        
        
        return self.loss_weight * loss
        
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    
    
    
class WeightedBCEWithLogitsLoss(BaseLoss):
    def __init__(self, 
                 weight: Optional[Sequence] = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_wbce', 
                 loss_weight: float = 1, 
                 
                 # loss args
                 gamma: int = 5, 
                 surround_size: int = 31, 
                 
                 
                 
                 ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)

        self.gamma = gamma
        self.surround_size = surround_size
        
    
    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
            
        padding = (self.surround_size - 1) // 2
        
        weit = 1 + self.gamma*torch.abs(F.avg_pool2d(target, kernel_size=self.surround_size, stride=1, padding=padding) - target)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit*wbce).sum(dim=(1, 2, 3)) / weit.sum(dim=(1, 2, 3))

        loss = weight_reduce_loss(wbce, self.weight, reduction=self.reduction, avg_factor=avg_factor)
        
        return loss
    
 
class BalancedBCEWithLogitsLoss(BaseLoss):
    def __init__(self, weight = None, 
                 reduction = 'mean', 
                 loss_name = 'loss_bbce', 
                 loss_weight = 1):
        super().__init__(weight, reduction, loss_name, loss_weight)

    def loss_func(self, pred, target, avg_factor=None, **kwargs):
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
            
        count_pos = target.sum(dim=(1, 2, 3), keepdim=True) + 1e-9
        count_neg = torch.sum(1. - target, dim=(1, 2, 3), keepdim=True)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)
        bbce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=ratio, reduction='none')
        bbce = w_neg * bbce
        loss = weight_reduce_loss(bbce, self.weight, reduction=self.reduction, avg_factor=avg_factor)
        return loss

 
 
class CrossEntropyLoss(nn.Module):
    '''
    Args:
        weight

        reduction: 'none', 'mean', 'sum'

        layer_args: (Sequence)
            - ignore_index (int): Specifies a target value that is ignored
                and does not contribute to the input gradient. Default: -100
                
            - label_smoothing: float = 0.0
    
    Shape:
        Input: Shape (C), (N, C) or (N, C, d_1, d_2, ..., d_K) with K \geq 1 in the case of K-dimensional loss.
        Target: If containing class indices, shape (), (N) or (N, d_1, d_2, ..., d_K) with K \geq 1 in the case of K-dimensional loss where each value should be between [0, C). If containing class probabilities, same shape as the input and each value should be between [0, 1].
    '''
    def __init__(self, 
                 weight: Optional[Sequence] = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_ce',
                 loss_weight: float = 1.0,
                 
                 # loss_args
                 ignore_index=-100,
                 label_smoothing=0.0,
                 ) -> None:
        super().__init__()
    
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
            
        self.weight = weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.weight is not None:
            weight = torch.as_tensor(self.weight, dtype=input.dtype, device=input.device)
        else:
            weight = None
        return self.loss_weight * F.cross_entropy(input, target,
                               weight = weight,
                               reduction = self.reduction,
                               ignore_index = self.ignore_index,
                               label_smoothing = self.label_smoothing)
        
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. 

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
    
    


        

