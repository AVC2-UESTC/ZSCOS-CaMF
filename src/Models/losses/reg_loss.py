import warnings
from typing import Optional, Union


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.distributed as dist

import einops

from .utils import weight_reduce_loss

from .base_loss import BaseLoss

# ================ MSE loss ========================================


class MSELoss(nn.Module):
    '''
    Args:
        reduction: 'none', 'mean', 'sum'
    
    '''

    
    def __init__(self, 
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        loss_name: str = 'loss_mse', 
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.weight = weight
        self.reduction = reduction
        
        self.loss_weight = loss_weight

        self._loss_name = loss_name

        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        '''
        Args:
            input: logits (Tensor) : (N, d_i)
            target: seg_map (Tensor): (N, d_i)
        '''
        
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss * self.loss_weight
        
        
        
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
    
    
class SmoothL1Loss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
    
    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        loss = F.smooth_l1_loss(pred, target, reduction=self.reduction)
        
        return loss

    


class CosSimilarityLoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)


    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        if isinstance(pred, list) and isinstance(target, list):
            n = len(pred)
            loss = torch.tensor(0.0, device=pred[0].device)
            for i in range(n):
                value = 1.0 - F.cosine_similarity(pred[i], target[i], dim=-1, eps=1e-6) # (N, L)
                loss = loss + value.mean(dim=-1)
        # loss = loss.mean(dim=-1)
        loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction)
        
        return loss



class UALLoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)

    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        pred = torch.sigmoid(pred)
        loss = 1.0 - (2 * pred - 1).abs().pow(2)
        # (N, 1, H, W)
        loss = loss.mean(dim=(1, 2, 3)) # (N, )
        loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction)
        
        return loss


class AdaLoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1,
                 
                 # loss args:
                 token_target_ratio=0.5
                 
                 ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        self.token_target_ratio = token_target_ratio

    def loss_func(self, pred: Tensor, target: Tensor=None, avg_factor=None, **kwargs) -> Tensor:
        pred = pred.float()
        token_select = pred
        # (bs, num_layers, L, 1)
        token_mean = token_select.mean(dim=(1, 2, 3))
        token_flops_loss = (token_mean - self.token_target_ratio)**2
        
        loss = weight_reduce_loss(token_flops_loss, self.weight, reduction=self.reduction)
        
        return loss
    
    
    
    
# moe loss =========================================================
# from switch traansformer 

class SparseMoELoss(BaseLoss):
    '''
    auxiliary_loss + z_loss
    '''
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1, 
                 
                 # loss args:
                 num_experts: int = 4,
                 topk: int = 2,
                 
                 ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        self.num_experts = num_experts
        self.topk = topk
        
    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        pred = pred.float()
        router_logits = pred
        # (B, num_layers, L, num_experts)
        probs = torch.softmax(router_logits, dim=-1)
        
        P_i = probs.mean(dim=(1, 2)) # (B, num_experts) 
        
        _, selected_experts_indices = torch.topk(probs, k=self.topk, dim=-1)  
        # (B, num_layers, L, topk)
        expert_mask = F.one_hot(selected_experts_indices, self.num_experts).float()
        # (B, num_layers, L, topk, num_experts)
        
        f_i = expert_mask.mean(dim=(1, 2, 3)) # (B, num_experts)
 
        loss = self.num_experts * (P_i * f_i).sum(dim=1) # (B, )
        loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction)
        return loss
        
        
class SparseTaskMoELoss(BaseLoss):
    '''
    
    '''
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1, 
                 
                 # loss args:
                 num_experts: int = 4,
                 topk: int = 2,
                 
                 ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        self.num_experts = num_experts
        self.topk = topk
        
    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        pred = pred.float()
        router_logits = pred
        # (B, num_experts)
        probs = torch.softmax(router_logits, dim=-1)
        
        P_i = probs # (B, num_experts) 
        
        _, selected_experts_indices = torch.topk(probs, k=self.topk, dim=-1)  
        # (B, topk)
        expert_mask = F.one_hot(selected_experts_indices, self.num_experts).float()
        # (B, topk, num_experts)
        
        f_i = expert_mask.mean(dim=1) # (B, num_experts)
 
        loss = self.num_experts * (P_i * f_i).sum(dim=1) # (B, )
        loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction)
        return loss



class NMILoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1,
                 
                 
                 ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        

    def loss_func(self, pred: Tensor, target: Tensor=None, avg_factor=None, **kwargs) -> Tensor:
        attn = pred
        b, h, q, k = attn.shape
        
        
        
        pq = torch.ones([b, h, q]).to(attn.device)
        pq = F.softmax(pq, dim=-1)
        pq_ext = einops.repeat(pq, "b h q -> b h q k", k=k)
        
        pk = einops.reduce(attn * pq_ext, "b h q k -> b h k", "sum")
        pk_ext = einops.repeat(pk, "b h k -> b h q k", q=q)
        
        mi = einops.reduce(attn * pq_ext * torch.log(attn / pk_ext), "b h q k -> b h", "sum")
        eq = - einops.reduce(pq * torch.log(pq), "b h q -> b h", "sum")
        ek = - einops.reduce(pk * torch.log(pk), "b h k -> b h", "sum")
    
        nmiv = mi / torch.sqrt(eq * ek)
        loss = 1 - nmiv.mean(dim=1) # (N, )
        
        loss = weight_reduce_loss(nmiv, self.weight, reduction=self.reduction)
        
        return loss
    
    
    
class MIMRegLoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1
        
        
        ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        
    
    def loss_func(self, pred: Tensor, target: Tensor, extra_in: Tensor, avg_factor=None, **kwargs) -> Tensor:
        c = pred.shape[1]
        # loss = F.l1_loss(pred, target, reduction='none') # (b, c, h, w)
        loss = F.mse_loss(pred, target, reduction='none') # (b, c, h, w)
        # loss = (loss * extra_in).sum(dim=(1, 2, 3)) / (extra_in.sum(dim=(1, 2, 3)) + 1e-5) / c 
        loss = loss.mean(dim=(1, 2, 3)) 
        loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction)
        
        return loss



class LMIMCosSimilarityLoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1
        
        
        ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        
    
    def loss_func(self, pred: Tensor, target: Tensor, extra_in: Tensor, avg_factor=None, **kwargs) -> Tensor:
        mask = extra_in.squeeze(-1) # (B, L)
        pred = pred.float()
        target = target.float()
        
        loss = 1 - F.cosine_similarity(pred, target, dim=-1, eps=1e-6) # (B, L)
        # loss = (loss * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-6) # (B, )
        loss = loss.mean(dim=-1) # (B, )
        
        
        loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction)
        
        return loss



class LMIMEntropyLoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_', 
                 loss_weight: float = 1,
                 
                 # loss_args:
                 patch_out_dim = 768,
                 student_temp: float = 0.1,
                 teacher_temp: float = 0.07,
                 center_momentum: float = 0.9,
        
        
        ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None
        
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
    
    def loss_func(self, pred: Tensor, target: Tensor, extra_in: Tensor, avg_factor=None, **kwargs) -> Tensor:
        mask = extra_in.squeeze(-1) # (B, L)
        pred = pred.float()
        target = target.float()
        
        self.apply_center_update()
        target = F.softmax((target - self.center) / self.teacher_temp, dim=-1)
        
        loss = target * F.log_softmax(pred / self.student_temp, dim=-1) 
        loss = loss.sum(dim=-1)  # (B, L)
        loss = - torch.sum(loss * mask.float(), dim=-1) / mask.sum(dim=-1).clamp(min=1.0) # (B, )
        
        
        loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction)
        
        return loss
    
    
    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True
