import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor


from ..backbones.eva02 import EVA_02_VisionTransformer
from ..decode_heads import SegformerHead
from ..decode_heads.segformer_head import SegformerHead_single_input
from ..losses import accuracy
from ..builder import build_loss

from ..utils import resize



class EVA_02_Segmentation(nn.Module):
    def __init__(self, 
                backbone_cfg: dict, 
                decode_head_cfg: dict,
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                
                
                ) -> None:
        super().__init__()
        
        self.backbone = EVA_02_VisionTransformer(**backbone_cfg)
        
        self.decode_head = SegformerHead_single_input(**decode_head_cfg)
        
        self.ignore_index = ignore_index
        self.align_corners = decode_head_cfg['align_corners']
        
        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            threshold = 0.3
            warnings.warn('threshold is not defined for binary, and defaults'
                          'to 0.3')
            
        self.threshold = threshold
            
            
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
        
        
    
    def forward(self, x):
        outs = self.backbone(x) 
        # (N, L+1, C)
        
        
        outs = outs[:, 1:, :]
        N, L, C = outs.shape
        h = int(math.sqrt(L))
        w = h
        
        outs = [outs.reshape(N, h, w, C).permute(0, 3, 1, 2),]
        
        out = self.decode_head(outs)
        # logits: (N, out_channel, H, W)
        
        return out
        
        
    
    def loss(self, inputs: Tensor, target: Tensor,
             return_logits: bool = False
             ) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            
            Target (list[:obj:`SegDataSample`]): The seg map. 

        Returns:
            dict[str, Tensor]: a dictionary of loss components
            
        Shape:
            inputs: (N, C, H, W)
            
            target: (N, 1, H, W)
            
            
            
        
        """
        seg_logits = self.forward(inputs)
        logits_prob = torch.sigmoid(seg_logits)
        
        # Calculate loss
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=target.shape[2:], #(N, 1, H, W)
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg_label = target.squeeze(1)
        # (N, H, W)
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        # losses_decode: loss layer(s) in Modulelist
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,#(N, H, W)
                    )
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    )
                
        
        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        # losses: {
        #         'acc_seg': acc_value
            
        #         'loss_name1': loss_value1
        #         ...
        #     }
        
        losses = loss
        
        if return_logits:
            return losses, logits_prob
        else:
            return losses
    

    def logits(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)
        #(N, out_channels, H/4, W/4)
        # raw: without sigmoid
    

    def predict(self, inputs: Tensor,
                return_logits: bool = False
                ) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tensor): 
            

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        img_size = inputs.shape[2:]
        
        seg_logits = self.forward(inputs)
        logits_prob = torch.sigmoid(seg_logits)
        seg_logits = resize(
            input=seg_logits,
            size=img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # (N, out_channels, H, W)
        if self.decode_head.out_channels == 1:
            seg_probs = torch.sigmoid(seg_logits)
            seg_map = (seg_probs > self.threshold).long()
            # (N, 1, H, W)
            seg_map = seg_map.squeeze(dim=1)
            #(N, H, W)
        else:
            seg_probs = F.softmax(seg_logits, dim=1)
            seg_map = torch.argmax(seg_probs, dim=1)
            # (N, H, W)
        
        if return_logits:
            return seg_map, logits_prob
        else:
            return seg_map
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'backbone.pos_embed', 'backbone.cls_token'}





# ================== Config ========================

class EVA_02_Segmentation_Config():
    '''
    
    '''
    def __init__(self, 
                backbone_cfg: dict, 
                decode_head_cfg: dict,
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,) -> None:
        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        self.threshold = threshold
        self.loss_decode = loss_decode
        self.ignore_index = ignore_index
        
    # a property method to instaniate the model
    @property
    def model(self):
        return EVA_02_Segmentation(**self.__dict__) 



def EVA_02_Segmentation_base_fgseg_cfg():
    args = EVA_02_Segmentation_Config(
        backbone_cfg = dict(
            img_size=352,
            patch_size=16,
            embed_dim=768, 
            depth=12, 
            num_heads=12, 
            mlp_ratio=4*2/3, 
            qkv_bias=True,
            norm_cfg=dict(
                norm_type='LayerNorm', 
                layer_args=dict(
                    eps=1e-6
                )
            ), 
            subln=True,
            xattn=True,
            naiveswiglu=True,
            rope=True, 
            pt_hw_seq_len=22,   # 352//16
            intp_freq=True,
        ), 
        
        decode_head_cfg = dict(
            in_channels=[768,],#
            channels=256,
            num_classes=2,
            out_channels=1,
                
            norm_cfg=dict(
                norm_type='BatchNorm2d',
                requires_grad=True,
                layer_args=dict(
                    eps=1e-5, 
                    momentum=0.1, 
                    affine=True,
                    track_running_stats=True
                )
                ),
                
                in_index=[0,],  
                align_corners=False,
                
                resize_ratio=4
        ), 
        
        threshold=0.5,
        loss_decode=dict(
            loss_type='BCEWithLogitsLoss',
            reduction = 'mean',
        ),
    )

    return args


def EVA_02_Segmentation_large_fgseg_cfg():
    args = EVA_02_Segmentation_Config(
        backbone_cfg = dict(
            img_size=384,
            patch_size=16,
            embed_dim=1024, 
            depth=24, 
            num_heads=16, 
            mlp_ratio=4*2/3, 
            qkv_bias=True,
            norm_cfg=dict(
                norm_type='LayerNorm', 
                layer_args=dict(
                    eps=1e-6
                )
            ), 
            subln=True,
            xattn=True,
            naiveswiglu=True,
            rope=True, 
            pt_hw_seq_len=22,   # 352//16
            intp_freq=True,
        ), 
        
        decode_head_cfg = dict(
            in_channels=[1024,],#
            channels=256,
            num_classes=2,
            out_channels=1,
                
            norm_cfg=dict(
                norm_type='BatchNorm2d',
                requires_grad=True,
                layer_args=dict(
                    eps=1e-5, 
                    momentum=0.1, 
                    affine=True,
                    track_running_stats=True
                )
                ),
                
                in_index=[0,],  
                align_corners=False,
                
                resize_ratio=4
        ), 
        
        threshold=0.5,
        loss_decode=dict(
            loss_type='BCEWithLogitsLoss',
            reduction = 'mean',
        ),
    )

    return args















