import warnings
from typing import Optional, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import weight_reduce_loss





class BaseLoss(nn.Module):
    '''
    Args:
        reduction: 'none', 'mean', 'sum'
    
    '''
    def __init__(self, 
                 
                weight: Optional[Tensor] = None,
                reduction: str = 'mean',
                loss_name: str = 'loss_', 
                loss_weight: float = 1.0,
                 
                 
                 ) -> None:
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        
        self.loss_weight = loss_weight

        self._loss_name = loss_name
        
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
    
    
    def forward(self, pred: Tensor, target:Tensor, **kwargs) -> Tensor:
        loss = self.loss_func(pred, target, **kwargs)
        return loss * self.loss_weight
    
    
    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        raise NotImplementedError





















