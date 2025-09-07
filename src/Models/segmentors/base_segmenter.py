import warnings
import math
import sys

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor

from scipy.optimize import linear_sum_assignment
import numpy as np

from ..builder import build_loss

from ..utils import resize


# only for semseg
def index_mask_to_onehot_mask_old(index_mask: Tensor, num_classes: int) -> List[Tensor]:
    '''
    Shape: 
        index_mask: (N, H, W)
        
        one_hot_mask: (N, num_masks, H, W)
        index_class: (N, num_masks)
    '''
    
    if index_mask.dim() == 4 and index_mask.shape[1] == 1:
        index_mask = index_mask.squeeze(1)
    index_mask = index_mask.to(torch.int64)
    channels = num_classes + 1 
    batch_size, h, w = index_mask.shape
    
    one_hot_mask = torch.zeros([batch_size, channels, h, w], dtype=index_mask.dtype, device=index_mask.device)
    index_class = torch.zeros([batch_size, channels], dtype=torch.int64, device=index_mask.device)
    class_p = torch.arange(channels, dtype=torch.int64, device=index_mask.device)
    # print(one_hot.shape)
    
    
    
    for i in range(batch_size):
        one_hot_mask_i = torch.zeros(h, w, channels, dtype=index_mask.dtype, device=index_mask.device)
        one_hot_mask_i.scatter_(dim=-1, index=index_mask[i].unsqueeze(-1), value=1)
        one_hot_mask[i] = one_hot_mask_i.permute(2, 0, 1)
        
        hist = index_mask[i].float().histc(
                bins=num_classes, min=0, max=num_classes - 1)
        index_class[i] = (hist > 0) * class_p
        
    # del one_hot_mask[:, 0]
    # del index_class[:, 0]
    one_hot_mask = one_hot_mask[:, 1:]
    index_class = index_class[:, 1:]
    
    label_mask = []
    label_class = []
    
    for i in range(batch_size):
        tgt_ids = index_class[i]
        tgt_mask = one_hot_mask[i]
        
    
        indices_nonzero = torch.nonzero(tgt_ids, as_tuple=True)[0]
        if not indices_nonzero.numel() == 0:
            tgt_ids = tgt_ids[indices_nonzero]
        # (num_masks, ) usually num_queries > num_masks
            tgt_mask = tgt_mask[indices_nonzero]
        # (num_masks, H, W),  

        label_mask.append(tgt_mask)
        label_class.append(tgt_ids)
        
    return label_mask, label_class



def index_mask_to_onehot_mask(index_mask: Tensor, num_classes: int) -> Tuple[Tensor, Tensor]:
    """
    Convert an index mask to a one-hot mask and extract class indices.

    Args:
        index_mask (Tensor): Shape (N, H, W) where N is the number of samples,
                             H is the height, and W is the width.
        num_classes (int): The number of classes for one-hot encoding.

    Returns:
        - one_hot_mask: List[Tensor] (N, num_classes, H, W).
        - index_class: List[Tensor] (N, num_masks) containing the unique class indices.
    """
    # Ensure the index_mask is of the correct type
    index_mask = index_mask.long()
    
    # Get the shape of the input
    batch_size, H, W = index_mask.shape
    
    # Use one_hot to convert index mask to one-hot encoding
    one_hot_mask = torch.nn.functional.one_hot(index_mask, num_classes=num_classes)
    
    # Rearrange the dimensions to (N, num_classes, H, W)
    one_hot_mask = one_hot_mask.permute(0, 3, 1, 2)  #(N, num_classes, H, W)
    
    label_mask = []
    label_class = []
    for i in range(batch_size):
        one_hot_mask_i = one_hot_mask[i]
        # Exclude all-zero masks by checking if any class is active
        mask_active = one_hot_mask_i.sum(dim=(1, 2)) > 0  # Sum over classes and find active masks
        one_hot_mask_i = one_hot_mask_i[mask_active]  # Filter out all-zero masks

        # Extract the unique class indices for each mask
        index_class_i = index_mask[i].unique(dim=None)  # Extracts unique classes along the last dimension
        index_class_i = index_class_i + 1  # Shift indices to start from 1
        label_mask.append(one_hot_mask_i)
        label_class.append(index_class_i)
    
    return label_mask, label_class
    # List[Tensor, ...]     Each tensor may have different shape
    


def linear_sum_assignment_with_nan(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    nan = np.isnan(cost_matrix).any()
    nan_all = np.isnan(cost_matrix).all()
    empty = cost_matrix.size == 0

    if not empty:
        if nan_all:
            print('Matrix contains all NaN values!')
        elif nan:
            print('Matrix contains NaN values!')

        if nan_all:
            cost_matrix = np.empty(shape=(0, 0))
        elif nan:
            cost_matrix[np.isnan(cost_matrix)] = 100

    return linear_sum_assignment(cost_matrix)



class BaseSegmenter(nn.Module):
    '''
    Args:

    '''
    def __init__(self, 
                threshold: float = None, 
                loss_decode=dict(
                    loss_type='CrossEntropyLoss', 
                    reduction='mean'),
                
                ignore_index: int = 255,
                align_corners: bool = False,
                
                ) -> None:
        super().__init__()

        self.threshold = threshold
        
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        
        # build loss layer
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(**loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(**loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        # self.loss_decode: loss layers in Modulelist
        
        
    def forward(self, inputs: Dict[str, Tensor]):
        '''
        Args:
            inputs: dict(
                image=...
                ...
            )
        
        return:
            dict(
                logits=...
            )
        '''
        raise NotImplementedError
    
    
    def loss(self, inputs: Dict[str, Tensor], labels: Dict[str, Tensor],
             return_logits: bool = False
             ) -> dict:
        """Forward function for training.

        Args:
            

        Returns:
            
        Shape:
            inputs: dict(
                image: (N, C, H, W)
            )
            
            labels: dict(
                label_mask: (N, out_channel, H, W)
            ) 
            
        """
        results = self.forward(inputs)
        
        seg_logits = results['logits_mask']
        
        seg_label = labels['label_mask']
        
        logits_prob = torch.sigmoid(seg_logits) # for metric computing
        
        
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:], #(N, 1, H, W)
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)
        # (N, H, W)
        
        # Calculate loss
        losses = dict()
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
            
        else:
            losses_decode = self.loss_decode
        # losses_decode: loss layer(s) in Modulelist
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in losses:
                losses[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,#(N, H, W)
                    )
            else:
                losses[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    )
               
        # losses: {
        #         
        #         'loss_name1': loss_value1
        #         ...
        #     }
        
        preds = dict(pred_mask=logits_prob)
        
        if return_logits:
            return losses, preds
        else:
            return losses
    

    def predict(self, inputs: Dict,
                return_logits: bool = False
                ) -> Dict:
        """Forward function for prediction.

        Args:
            inputs: 
            

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        img_size = inputs['image'].shape[2:]
        
        results = self.forward(inputs)
        seg_logits = results['logits_mask']
        logits_prob = torch.sigmoid(seg_logits) # for metric computing
        preds = dict(
            pred_mask=logits_prob,
        )
        
        seg_logits = resize(
            input=seg_logits,
            size=img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # (N, out_channels, H, W)
        try:
            out_channels = self.decode_head.out_channels
        except:
            out_channels = self.out_channels
        # except:
        #     raise KeyError('Cannot find the out_channels of the model')
        
        if out_channels == 1:
            seg_probs = torch.sigmoid(seg_logits)
            if self.threshold is not None:
                seg_map = (seg_probs > self.threshold).float()
            else:
                seg_map = seg_probs
            # (N, 1, H, W)
            seg_map = seg_map.squeeze(dim=1)
            #(N, H, W)
        else:
            seg_probs = F.softmax(seg_logits, dim=1)
            seg_map = torch.argmax(seg_probs, dim=1)
            # (N, H, W)
        
        outputs = dict(
            pred_mask=seg_map
        )
        
        if return_logits:
            return outputs, preds
        else:
            return outputs



class BaseSegmentor_Config():
    '''
    Args:

    '''
    def __init__(self, 
                pretrained_weights: str=None,
                finetune_weights: str=None,
                tuning_mode: str='PEFT',
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                
                align_corners: bool = False,
                
                
                ) -> None:
        self.pretrained_weights = pretrained_weights
        self.finetune_weights = finetune_weights
        self.tuning_mode = tuning_mode
        
        self.threshold = threshold
        self.loss_decode = loss_decode
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        
        self.model_class = None
        
        
    def set_model_class(self):
        '''
        E.g:
            self.model_class = Muv02_MaskDecoder_Segmentation_Adapter_SFP
            
        '''
        raise NotImplementedError
        
    # a property method to instaniate the model
    @property
    def model(self):
        self.set_model_class()
        model_args = deepcopy(self.__dict__)
        model_args.pop('pretrained_weights')
        model_args.pop('finetune_weights')
        model_args.pop('tuning_mode')
        model_args.pop('model_class')
        model_inst = self.model_class(**model_args)
        
        
        if self.pretrained_weights is not None:
            print('Pretrained weights loaded.')
            model_weights = torch.load(self.pretrained_weights, map_location='cpu')
            
            # for k, v in model_inst.state_dict().items():
            #     print(k)
            
            # for k, v in model_weights.items():
            #     try:
            #         print(k)
            #         print(model_inst.state_dict()[k].shape)
            #     except KeyError:
            #         print(f'{k} not in model state dict')
            #         for k2, v2 in model_inst.state_dict().items():
            #             print(k2)
            #         break
            
            load_weights_dict = {k: v for k, v in model_weights.items()
                                if model_inst.state_dict()[k].numel() == v.numel()}
            
            msg = model_inst.load_state_dict(load_weights_dict, strict=False)
            
            
            if self.tuning_mode == 'PEFT':
                print('Start Parameter-Efficient Finetuning')
                weights_tosave_keys = msg.missing_keys
                
                if self.finetune_weights is not None:
                    print('Finetune weights loaded')
                    model_finetune_weights = torch.load(self.finetune_weights, map_location='cpu')
                    load_weights_dict = {k: v for k, v in model_finetune_weights.items()
                                 if model_inst.state_dict()[k].numel() == v.numel()}
                    model_inst.load_state_dict(load_weights_dict, strict=False)
                    
                # freeze all but finetune and head
                for name, p in model_inst.named_parameters():
                    if name in weights_tosave_keys:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    
            elif self.tuning_mode == 'Full':
                print('Start Full Finetuning')
                weights_tosave_keys = model_inst.state_dict().keys()
                
                if self.finetune_weights is not None:
                    print('Finetune weights loaded')
                    model_finetune_weights = torch.load(self.finetune_weights, map_location='cpu')
                    load_weights_dict = {k: v for k, v in model_finetune_weights.items()
                                 if model_inst.state_dict()[k].numel() == v.numel()}
                    model_inst.load_state_dict(load_weights_dict, strict=False)
                
        else:
            print('No pretrained weights. Start training from scratch.')
            weights_tosave_keys = model_inst.state_dict().keys()
        
        return model_inst, weights_tosave_keys




class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, 
                 cost_class: float = 1, 
                 cost_mask: float = 1, 
                 cost_dice: float = 1, 
                 num_points: int = 12544):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["logits_class"].shape[:2]
        
        

        indices = []
        
        # Iterate through batch size
        for b in range(bs):

            
            # we need remove the background class......
            
            out_prob = outputs["logits_class"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets['label_class'][b] # (num_masks, )
            
            out_mask = outputs["logits_mask"][b]  # [num_queries, H, W]
            tgt_mask = targets["label_mask"][b]  # (num_masks, H, W)
            
            
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids-1]
            

            # out_mask = out_mask[:, None]
            # tgt_mask = tgt_mask[:, None]
            out_mask = out_mask.unsqueeze(1)
            tgt_mask = tgt_mask.unsqueeze(1)
            
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            
            # get gt labels
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            tgt_mask = self._point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = self._point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            # (N, num_points)

            with torch.cuda.amp.autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                cost_mask = self.batch_sigmoid_ce_loss(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                cost_dice = self.batch_dice_loss(out_mask, tgt_mask)
            
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            # indices.append(linear_sum_assignment(C))
            
            # debug hook
            try:
                re = linear_sum_assignment_with_nan(C)
            except ValueError:
                # print all elements of tensor
                torch.save(C, 'debugC.pt')
                torch.save(cost_mask, 'debugcost_mask.pt')
                torch.save(cost_class, 'debugcost_class.pt')
                torch.save(cost_dice, 'debugcost_dice.pt')
                print('ValueError, Debug log saved')
                sys.exit()
            indices.append(re)
            
            # for each batch: e.g. ([1, 3, 4, 6], [2, 1, 3, 0])
            # indices: (N, num_masks, num_masks)

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    @staticmethod
    def _point_sample(input, point_coords, **kwargs):
        """
        A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
        Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
        [0, 1] x [0, 1] square.

        Args:
            input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
            point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
            [0, 1] x [0, 1] normalized point coordinates.

        Returns:
            output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
                features for points in `point_coords`. The features are obtained via bilinear
                interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
        """
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            point_coords = point_coords.unsqueeze(2) # (N, P, 1, 2)
        output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
        if add_dim:
            output = output.squeeze(3)
        return output
    
    @staticmethod
    def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1e-4) / (denominator + 1e-4)
        return loss
    
    @staticmethod
    def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        Returns:
            Loss tensor
        """
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )
        
        return loss / hw


class BaseMaskSegmenter(nn.Module):
    '''
    Args:

    '''
    def __init__(self, 
                num_classes: int,
                threshold: float = None, 
                
                
                loss_decode=dict(
                    loss_type='CrossEntropyLoss', 
                    reduction='mean'),
                
                ignore_index: int = 255,
                align_corners: bool = False,
                
                ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold
        
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        
        # build loss layer
        if isinstance(loss_decode, dict):
            self.loss_decode = build_loss(**loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(build_loss(**loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')
        # self.loss_decode: loss layers in Modulelist
        
        self.matcher = HungarianMatcher(
            cost_class=1, 
            cost_mask=1, 
            cost_dice=1,
            num_points=12544
        )
        
        
    def forward(self, inputs: Dict[str, Tensor]):
        '''
        Args:
            inputs: dict(
                image=...
                ...
            )
        
        return:
            dict(
                logits_class=..., 
                logits_mask=...,
            )
        '''
        raise NotImplementedError
    
    
    def loss(self, inputs: Dict[str, Tensor], labels: Dict[str, Tensor],
             return_logits: bool = False
             ) -> dict:
        """Forward function for training.

        Args:
            

        Returns:
            
        Shape:
            inputs: dict(
                image: (N, C, H, W)
            )
            
            labels: dict(
                label_mask: (N, H, W)
            ) 
            
        """
        results = self.forward(inputs)
        
        logits_class = results['logits_class'] # (N, n_q, num_classes + 1)
        bs, n_q = logits_class.shape[0:2]
        logits_mask = results['logits_mask'] # (N, n_q, h, w)
        
        label_mask = labels['label_mask'] # (N, H, W) indices
        img_size = label_mask.shape[-2:]
        
        label_mask, label_class = index_mask_to_onehot_mask(label_mask, num_classes=self.num_classes)
        # List[Tensor]: (N, num_masks, H, W), (N, num_masks)
        
        
        labels_f_matcher = dict(
            label_class=label_class, # (N, num_masks)
            label_mask=label_mask, # (N, num_masks, H, W), 
        )
        
        
        
        if self.num_classes == 1:
            pred_class = F.softmax(logits_class, dim=-1)[:, :, 1:]
            pred_masks = torch.sigmoid(logits_mask)
            # logits_prob = torch.einsum('bqn,bqhw->bnhw', pred_class, pred_masks) 
            max_index = pred_class.argmax(dim=1) # (N, 1)
            max_index = max_index.squeeze(dim=1)
            sqe_index = torch.arange(0, bs, dtype=torch.int64, device=logits_mask.device)
            logits_prob = pred_masks[sqe_index, max_index].unsqueeze(1)
            
            # 
            
            
            # print(torch.max(logits_prob))
        elif self.num_classes > 1:
            pred_class = F.softmax(logits_class, dim=-1)[:, :, 1:]
            pred_masks = torch.sigmoid(logits_mask)
            score_map = torch.einsum('bqn,bqhw->bnhw', pred_class, pred_masks)
            pred = score_map.argmax(dim=1)  # Shape: [N, H, W]
        else:
            raise NotImplementedError
        
        # resize mask 
        logits_mask = resize(
            input=logits_mask,
            size=img_size,
            mode='bilinear',
            align_corners=self.align_corners
        )
        # (N, n_q, H, W)
        
        logits_f_matcher = dict(
            logits_class=logits_class[:, :, 1:], # (N, n_q, num_classes)
            logits_mask=logits_mask # (N, n_q, H, W)
        )
        
        indices = self.matcher(logits_f_matcher, labels_f_matcher)
        # (N, num_masks, num_masks)
        # for each batch: e.g. ([1, 3, 4, 6], [2, 1, 3, 0])
        
        bs = len(indices)
        
        label_mask_f_bp = []
        logits_mask_f_bp = []
        label_cls_f_bp = torch.zeros([bs, n_q], dtype=torch.int64, device=logits_mask.device)
        # (N, n_q)
        
        for b in range(bs):
            query_ids, label_msk_ids = indices[b]
            selected_query_msk = logits_mask[b][query_ids]
            selected_label_msk = label_mask[b][label_msk_ids].float()
            # (num_masks, H, W)
            selected_label_cls = label_class[b][label_msk_ids]
            # (num_masks, )
            
            label_cls_f_bp[b][query_ids] = selected_label_cls
            
            label_mask_f_bp.append(selected_label_msk)
            logits_mask_f_bp.append(selected_query_msk)
        # label_cls_f_bp: Tensor, (N, n_q)
        # label_mask_f_bp: List[Tensor], (N, num_masks, H, W)
        # logits_mask_f_bp: List[Tensor], (N, num_masks, H, W)
        
           
        # Calculate loss
        losses = dict()
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
            
        else:
            losses_decode = self.loss_decode
        # losses_decode: loss layer(s) in Modulelist
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in losses:
                if loss_decode.loss_name.startswith('class_'):
                    losses[loss_decode.loss_name] = loss_decode(
                        logits_class.permute(0, 2, 1), # (N, num_classes + 1, n_q))
                        label_cls_f_bp # (N, n_q)
                        )
                elif loss_decode.loss_name.startswith('mask_'):
                    losses[loss_decode.loss_name] = loss_decode(
                        logits_mask_f_bp, # List[Tensor], (N, num_masks, H, W)
                        label_mask_f_bp # List[Tensor], (N, num_masks, H, W)
                    )
                else:
                    raise ValueError(f'loss_name must start with "class_" or "mask_", \
                        but got {loss_decode.loss_name}')
                    
            else:
                if loss_decode.loss_name.startswith('class_'):
                    losses[loss_decode.loss_name] += loss_decode(
                        logits_class.permute(0, 2, 1), # (N, num_classes + 1, n_q))
                        label_cls_f_bp # (N, n_q)
                        )
                elif loss_decode.loss_name.startswith('mask_'):
                    losses[loss_decode.loss_name] += loss_decode(
                        logits_mask_f_bp, # List[Tensor], (N, num_masks, H, W)
                        label_mask_f_bp # List[Tensor], (N, num_masks, H, W)
                    )
                else:
                    raise ValueError(f'loss_name must start with "class_" or "mask_", \
                        but got {loss_decode.loss_name}')
               
        # losses: {
        #         
        #         'loss_name1': loss_value1
        #         ...
        #     }
        
        preds = dict(pred_mask=logits_prob)
        
        if return_logits:
            return losses, preds
        else:
            return losses
    

    def predict(self, inputs: Dict,
                return_logits: bool = False
                ) -> Dict:
        """Forward function for prediction.

        Args:
            inputs: 
            

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        img_size = inputs['image'].shape[2:]
        bs = inputs['image'].shape[0]
        results = self.forward(inputs)
        
        logits_class = results['logits_class'] # (N, n_q, num_classes + 1)
        n_q = logits_class.shape[1]
        logits_mask = results['logits_mask'] # (N, n_q, h, w)
        
        
        if self.num_classes == 1:
            pred_class = F.softmax(logits_class, dim=-1)[:, :, 1:]
            pred_masks = torch.sigmoid(logits_mask)
            # logits_prob = torch.einsum('bqn,bqhw->bnhw', pred_class, pred_masks) 
            max_index = pred_class.argmax(dim=1) # (N, 1)
            max_index = max_index.squeeze(dim=1)
            sqe_index = torch.arange(0, bs, dtype=torch.int64, device=logits_mask.device)
            logits_prob = pred_masks[sqe_index, max_index].unsqueeze(1)
            
            
            
            
            # for metric computing
            # (N, num_classes, h, w)
        else:
            raise NotImplementedError
        
        preds = dict(
            pred_mask=logits_prob,
        )
        
        seg_logits = resize(
            input=logits_mask,
            size=img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # (N, num_classes, H, W)
        if self.num_classes == 1:
            seg_probs = torch.sigmoid(seg_logits)
            pred_class = F.softmax(logits_class, dim=-1)[:, :, 1:]
            
            # seg_probs = torch.einsum('bqn,bqhw->bnhw', pred_class, seg_probs) 
            max_index = pred_class.argmax(dim=1) # (N, 1)
            max_index = max_index.squeeze(dim=1)
            sqe_index = torch.arange(0, bs, dtype=torch.int64, device=logits_mask.device)
            seg_probs = seg_probs[sqe_index, max_index].unsqueeze(1)
            
            # seg_probs = torch.sigmoid(seg_logits) # <-------------------------------------------- Hook
            
            if self.threshold is not None:
                seg_map = (seg_probs > self.threshold).float()
            else:
                seg_map = seg_probs
            # (N, 1, H, W)
            # seg_map = seg_map.squeeze(dim=1)
            #(N, H, W)
        else:
            raise NotImplementedError
            # seg_probs = F.softmax(seg_logits, dim=1)
            # seg_map = torch.argmax(seg_probs, dim=1)
            # (N, H, W)
        
        output = dict(
            pred_mask=seg_map
        )
        
        if return_logits:
            return output, preds
        else:
            return output


class BaseMaskSegmentor_Config():
    '''
    Args:

    '''
    def __init__(self, 
                pretrained_weights: str=None,
                finetune_weights: str=None,
                tuning_mode: str='PEFT',
                
                num_classes: int=1,
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                
                align_corners: bool = False,
                
                
                ) -> None:
        self.pretrained_weights = pretrained_weights
        self.finetune_weights = finetune_weights
        self.tuning_mode = tuning_mode
        
        self.num_classes = num_classes
        self.threshold = threshold
        self.loss_decode = loss_decode
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        
        self.model_class = None
        
        
    def set_model_class(self):
        '''
        E.g:
            self.model_class = Muv02_MaskDecoder_Segmentation_Adapter_SFP
            
        '''
        raise NotImplementedError
        
    # a property method to instaniate the model
    @property
    def model(self):
        self.set_model_class()
        model_args = deepcopy(self.__dict__)
        model_args.pop('pretrained_weights')
        model_args.pop('finetune_weights')
        model_args.pop('tuning_mode')
        model_args.pop('model_class')
        model_inst = self.model_class(**model_args)
        
        
        if self.pretrained_weights is not None:
            print('Pretrained weights loaded.')
            model_weights = torch.load(self.pretrained_weights, map_location='cpu')
            
            load_weights_dict = {k: v for k, v in model_weights.items()
                                if model_inst.state_dict()[k].numel() == v.numel()}
            
            msg = model_inst.load_state_dict(load_weights_dict, strict=False)
            
            
            if self.tuning_mode == 'PEFT':
                print('Start Parameter-Efficient Finetuning')
                weights_tosave_keys = msg.missing_keys
                
                if self.finetune_weights is not None:
                    print('Finetune weights loaded')
                    model_finetune_weights = torch.load(self.finetune_weights, map_location='cpu')
                    load_weights_dict = {k: v for k, v in model_finetune_weights.items()
                                 if model_inst.state_dict()[k].numel() == v.numel()}
                    model_inst.load_state_dict(load_weights_dict, strict=False)
                    
                # freeze all but finetune and head
                for name, p in model_inst.named_parameters():
                    if name in weights_tosave_keys:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    
            elif self.tuning_mode == 'Full':
                print('Start Full Finetuning')
                weights_tosave_keys = model_inst.state_dict().keys()
                
                if self.finetune_weights is not None:
                    print('Finetune weights loaded')
                    model_finetune_weights = torch.load(self.finetune_weights, map_location='cpu')
                    load_weights_dict = {k: v for k, v in model_finetune_weights.items()
                                 if model_inst.state_dict()[k].numel() == v.numel()}
                    model_inst.load_state_dict(load_weights_dict, strict=False)
                
        else:
            print('No pretrained weights. Start training from scratch.')
            weights_tosave_keys = model_inst.state_dict().keys()
        
        return model_inst, weights_tosave_keys