import warnings
from typing import Optional, Union, List


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import weight_reduce_loss
from .base_loss import BaseLoss

# ================ IOU loss ========================================


def iou_loss(pred: Tensor, target:Tensor, 
             weight: Union[torch.Tensor, None],
             reduction: str = 'mean', 
             avg_factor: Union[int, None] = None,):
    '''
    Shape:
        pred: (N, 1, H, W)
        target: (N, 1, H, W)
    '''
    pred = torch.sigmoid(pred)
    intersection = pred * target
    
    loss = intersection.sum(dim=(1, 2, 3)) / (pred + target - intersection).sum(dim=(1, 2, 3))
    loss = 1 - loss
    
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class IOULoss(nn.Module):
    '''
    Args:
        reduction: 'none', 'mean', 'sum'
    
    '''

    
    def __init__(self, 
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        loss_name: str = 'loss_iou', 
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
            input: logits (Tensor) : (N, 1, H, W)
            target: seg_map (Tensor): (N, H, W) or (N, 1, H, W)
        '''

        if target.dim() == 3: # (N, H, W)
            target = target.unsqueeze(1)
            loss = iou_loss(pred, target, reduction=self.reduction, weight=self.weight)
        elif target.dim() == 4: # (N, 1, H, W)
            loss = iou_loss(pred, target, reduction=self.reduction, weight=self.weight)
        else:
            raise ValueError(f'Unsupported target dimension: {target.dim()}')
        
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
        
        
        

# ============================ Weighted IOU loss ====================================

def weighted_iou_loss(pred: Tensor, target: Tensor, 
                      
                      weight: Tensor = None,
                      reduction: str = 'mean',
                      avg_factor=None,
                      
                      gamma: int = 5, 
                      surround_size: int = 31
                      ) -> Tensor:
   
    if len(target.shape) == 3:
        target = target.unsqueeze(1)

    pred = torch.sigmoid(pred)
    padding = (surround_size - 1) // 2
    weit = 1 + gamma*torch.abs(F.avg_pool2d(target, kernel_size=surround_size, stride=1, padding=padding) - target)
    
    inter = ((pred * target) * weit).sum(dim=(1, 2, 3))
    union = ((pred + target) * weit).sum(dim=(1, 2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    loss = weight_reduce_loss(wiou, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss



class WeightedIoULoss(BaseLoss):
    def __init__(self, 
                 weight: Tensor | None = None, 
                 reduction: str = 'mean', 
                 loss_name: str = 'loss_wiou', 
                 loss_weight: float = 1.0, 
                 
                 # loss args
                 gamma: int = 5, 
                 surround_size: int = 31, 
                 
                 ) -> None:
        super().__init__(weight, reduction, loss_name, loss_weight)
        
        self.gamma = gamma
        self.surround_size = surround_size
        
    
    def loss_func(self, pred: Tensor, target: Tensor, avg_factor=None, **kwargs) -> Tensor:
        pred = pred.float()
        target = target.float()
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
            
        
        pred = torch.sigmoid(pred)
        padding = (self.surround_size - 1) // 2
        weit = 1 + self.gamma*torch.abs(F.avg_pool2d(target, kernel_size=self.surround_size, stride=1, padding=padding) - target)
        
        inter = ((pred * target) * weit).sum(dim=(1, 2, 3))
        union = ((pred + target) * weit).sum(dim=(1, 2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        loss = weight_reduce_loss(wiou, weight=self.weight, reduction=self.reduction, avg_factor=avg_factor)
        return loss
            
        
        
        
        
        
        
        
        
        
        
        
        
        
# ============================== Dice Loss ==========================================
        
        
def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              weight: Union[torch.Tensor, None],
              eps: float = 1e-3,
              reduction: Union[str, None] = 'mean',
              avg_factor: Union[int, None] = None,
    ) -> float:
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        
            
    '''
    Shape:
        pred: (N, 1, H, W)
        target: (N, 1, H, W)
    '''
    """
    
    # assert pred.shape[1] == 1, f'only support binary segmentation'  
    pred = torch.sigmoid(pred)
    
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_card = pred.sum(dim=(1, 2, 3))
    target_card = target.sum(dim=(1, 2, 3))
    
    dice = (2 * intersection + eps) / (pred_card + target_card + eps)

    loss = 1 - dice
    
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



class DiceLoss(nn.Module):

    def __init__(self,
                weight=None,
                reduction='mean',
                loss_name='loss_dice',
                loss_weight=1.0,
                
                # loss_args                                  
                 eps=1e-3,
                 ):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
       
        self._loss_name = loss_name

    def forward(self,
                pred: Union[Tensor, List[Tensor]],
                target: Union[Tensor, List[Tensor]],
                avg_factor=None,
                ):
        '''Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        
        Shape:
        
            input: logits (Tensor) : (N, 1, H, W)
            target: seg_map (Tensor): (N, H, W) or (N, 1, H, W)
        '''
        
        
        if isinstance(pred, list) and isinstance(target, list):
            # (N, num_masks, H, W)
            bs = len(pred)
            loss = torch.zeros(bs)
            for b in range(bs):
                one_target = target[b].unsqueeze(1)
                one_pred = pred[b].unsqueeze(1)
                
                one_loss = dice_loss(one_pred, one_target, weight=self.weight, reduction='mean', eps=self.eps, avg_factor=avg_factor)
                loss[b] = one_loss
            
            loss = weight_reduce_loss(loss, self.weight, reduction=self.reduction, avg_factor=avg_factor)
        
        elif isinstance(pred, Tensor) and isinstance(target, Tensor):
            
            if target.dim() == 3: # (N, H, W)
                target = target.unsqueeze(1)
                loss = dice_loss(pred, target, reduction=self.reduction, weight=self.weight, eps=self.eps, avg_factor=avg_factor)
            elif target.dim() == 4: # (N, num_classes, H, W)
                loss = dice_loss(pred, target, reduction=self.reduction, weight=self.weight, eps=self.eps, avg_factor=avg_factor)
            else:
                raise ValueError(f'Unsupported target dimension: {target.dim()}')
            
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
def _expand_onehot_labels_dice(pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    if num_classes == 1:
        # foreground map
        target = target.to(dtype=torch.long)
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


def dice_loss_old(pred: torch.Tensor,
              target: torch.Tensor,
              weight: Union[torch.Tensor, None],
              eps: float = 1e-10,
              reduction: Union[str, None] = 'mean',
              naive_dice: Union[bool, None] = False,
              avg_factor: Union[int, None] = None,
              ignore_index: Union[int, None] = 255) -> float:
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
            loss defined in the V-Net paper, otherwise, use the
            naive dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to 255.
    """
    if ignore_index is not None:
        num_classes = pred.shape[1]
        pred = pred[:, torch.arange(num_classes) != ignore_index, :, :]
        target = target[:, torch.arange(num_classes) != ignore_index, :, :]
        assert pred.shape[1] != 0  # if the ignored index is the only class
    input = pred.flatten(1)
    target = target.flatten(1).float()
    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



class DiceLoss_old(nn.Module):

    def __init__(self,
                weight=None,
                reduction='mean',
                loss_name='loss_dice',
                loss_weight=1.0,
                
                # loss_args
                 use_sigmoid=True,
                 activate=True,
                 
                 naive_dice=False,
                 
                 ignore_index=255,
                 eps=1e-10,
                 ):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            ignore_index (int, optional): The label index to be ignored.
                Default: 255.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_dice'.
        """

        super().__init__()
        self.weight = weight
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        one_hot_target = target
        if (pred.shape != target.shape):
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                # softmax does not work when there is only 1 class
                pred = pred.softmax(dim=1)
        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            self.weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)

        return loss

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


























