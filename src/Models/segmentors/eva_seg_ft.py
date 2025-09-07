import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor

from einops import rearrange

from ..decode_heads.segformer_head import SegformerHead_single_input, SegformerHead
from ..losses import accuracy
from ..builder import build_loss

from ..utils import resize

from ..backbones.eva02_ft import (EVA_02_VisionTransformer_Adapter, EVA_02_VisionTransformer_Adapter_SFP, 
                                EVA_02_VisionTransformer_Mona, EVA_02_VisionTransformer_Mona_SFP,)
from .eva_seg import EVA_02_Segmentation, EVA_02_Segmentation_Config

from ..segmentors import BaseSegmenter, BaseSegmentor_Config


class EVA_02_Segmentation_Adapter(BaseSegmenter):
    '''
    Args:

    '''
    def __init__(self, 
                 backbone_cfg: Dict, 
                 decode_head_cfg: Dict, 
                 
                 threshold: float = None, 
                 loss_decode: Dict = dict(loss_type='CrossEntropyLoss', 
                                  reduction='mean'),  
                 ignore_index: int = 255, 
                 align_corners: bool = False) -> None:
        super().__init__(threshold=threshold, loss_decode=loss_decode, ignore_index=ignore_index, align_corners=align_corners)
        
        self.backbone = EVA_02_VisionTransformer_Adapter(**backbone_cfg)
        
        self.decode_head = SegformerHead_single_input(**decode_head_cfg)
        
        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')
            
            
    def forward(self, inputs: Dict):
        prompt = inputs['prompt']
        x = inputs['image']
        
        outs = self.backbone(x) 
        outs = outs[:, 1:, :] # remove cls token
        N, L, C = outs.size()
        h = w = int(L ** 0.5)
        outs = rearrange(outs, 'b (h w) c -> b c h w', h=h, w=w)
        outs = [outs, ]
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        
        
        results = self.decode_head(outs)
        # results: dict(
        #   logits_mask: (N, out_channel, H, W), 
        # )
        # 
        
        return results

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'backbone.pos_embed', 'backbone.cls_token'}

        

# ================== Config ========================

class EVA_02_Segmentation_Adapter_Config(BaseSegmentor_Config):
    '''
    
    '''
    def __init__(self, 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'PEFT', 
                 
                backbone_cfg: dict = None, 
                decode_head_cfg: dict = None,
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)        
        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
        
    def set_model_class(self):
        self.model_class = EVA_02_Segmentation_Adapter




from ..backbones.eva02_ft_evp import EVA02EVP
class EVA_02_Segmentation_EVP(BaseSegmenter):
    '''
    Args:

    '''
    def __init__(self, 
                 backbone_cfg: Dict, 
                 decode_head_cfg: Dict, 
                 
                 threshold: float = None, 
                 loss_decode: Dict = dict(loss_type='CrossEntropyLoss', 
                                  reduction='mean'),  
                 ignore_index: int = 255, 
                 align_corners: bool = False) -> None:
        super().__init__(threshold=threshold, loss_decode=loss_decode, ignore_index=ignore_index, align_corners=align_corners)
        
        self.backbone = EVA02EVP(**backbone_cfg)
        
        self.decode_head = SegformerHead_single_input(**decode_head_cfg)
        
        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')
            
            
    def forward(self, inputs: Dict):
        # prompt = inputs['prompt']
        x = inputs['image']
        
        outs = self.backbone(x) 
        outs = outs[:, 1:, :] # remove cls token
        N, L, C = outs.size()
        h = w = int(L ** 0.5)
        outs = rearrange(outs, 'b (h w) c -> b c h w', h=h, w=w)
        outs = [outs, ]
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        
        
        results = self.decode_head(outs)
        # results: dict(
        #   logits_mask: (N, out_channel, H, W), 
        # )
        # 
        
        return results

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'backbone.pos_embed', 'backbone.cls_token'}

        

# ================== Config ========================

class EVA_02_Segmentation_EVP_Config(BaseSegmentor_Config):
    '''
    
    '''
    def __init__(self, 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'PEFT', 
                 
                backbone_cfg: dict = None, 
                decode_head_cfg: dict = None,
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)        
        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
        
    def set_model_class(self):
        self.model_class = EVA_02_Segmentation_EVP








# ============== AdapterFormer + ViTDet =======================

class EVA_02_Segmentation_Adapter_SFP(nn.Module):
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
        
        self.backbone = EVA_02_VisionTransformer_Adapter_SFP(**backbone_cfg)
        
        self.decode_head = SegformerHead(**decode_head_cfg)
        
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
        # List: [Tensor, ...]

        
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


# ============== Config =====================

class EVA_02_Segmentation_Adapter_SFP_Config():
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
        return EVA_02_Segmentation_Adapter_SFP(**self.__dict__) 



def EVA_02_Segmentation_Adapter_SFP_base_fgseg_cfg():
    args = EVA_02_Segmentation_Adapter_SFP_Config(
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
            
            ft_cfg=[
                dict(
                    type='backbone_ft',
                    bottleneck=64, 
                    adapter_scalar='learnable_scalar',#'0.1', 
                    
                    act_cfg=dict(
                        act_type='ReLU', 
                        layer_args=dict(inplace=True)
                    ),     
                    adapter_layernorm_option='none',
                            
                    dropout_layer = dict(
                    drop_type='Dropout',
                    drop_prob=0.0,
                    inplace=False)
                ),
                dict(
                    type='neck_ft',
                    out_channels=256,
                    scale_factors=[4.0, 2.0, 1.0, 0.5],
                    norm_cfg=dict(
                        norm_type='LayerNorm2d'   
                    ),
                )
            ]
        ), 
        
        decode_head_cfg = dict(
            in_channels=[256, 256, 256, 256],#
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
                
            in_index=[0, 1, 2, 3],  
            align_corners=False,
                
        ), 
        
        threshold=0.5,
        loss_decode=dict(
            loss_type='BCEWithLogitsLoss',
            reduction = 'mean',
        ),
    )

    return args


def EVA_02_Segmentation_Adapter_SFP_large_fgseg_cfg():
    args = EVA_02_Segmentation_Adapter_SFP_Config(
        backbone_cfg = dict(
            img_size=384,
            # img_size=224,
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
            # pt_hw_seq_len=14,
            intp_freq=True,
            
            ft_cfg=[
                dict(
                    type='backbone_ft',
                    bottleneck=64, 
                    adapter_scalar=0.1,#'0.1', 
                    
                    act_cfg=dict(
                        act_type='ReLU', 
                        layer_args=dict(inplace=True)
                    ),     
                    adapter_layernorm_option='none',
                            
                    dropout_layer = dict(
                    drop_type='Dropout',
                    drop_prob=0.0,
                    inplace=False)
                ),
                dict(
                    type='neck_ft',
                    out_channels=256,
                    scale_factors=[4.0, 2.0, 1.0, 0.5],
                    norm_cfg=dict(
                        norm_type='LayerNorm2d'   
                    ),
                )
            ]
        ), 
        
        decode_head_cfg = dict(
            in_channels=[256, 256, 256, 256],#
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
                
            in_index=[0, 1, 2, 3],  
            align_corners=False,
                
        ), 
        
        threshold=0.5,
        loss_decode=dict(
            loss_type='BCEWithLogitsLoss',
            reduction = 'mean',
        ),
    )
    
    return args



# ======================================================================
# ================= Mona ===============================================

class EVA_02_Segmentation_Mona(EVA_02_Segmentation):
    def __init__(self, 
                 backbone_cfg: dict, 
                 decode_head_cfg: dict, 
                 threshold=None, 
                 loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                 ignore_index=255) -> None:
        super(EVA_02_Segmentation, self).__init__()
        
        self.backbone = EVA_02_VisionTransformer_Mona(**backbone_cfg)
        
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
    

class EVA_02_Segmentation_Mona_SFP(EVA_02_Segmentation):
    def __init__(self, 
                 backbone_cfg: dict, 
                 decode_head_cfg: dict, 
                 threshold=None, 
                 loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                 ignore_index=255) -> None:
        super(EVA_02_Segmentation, self).__init__()
        
        self.backbone = EVA_02_VisionTransformer_Mona_SFP(**backbone_cfg)
        
        self.decode_head = SegformerHead(**decode_head_cfg)
        
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
        # List: [Tensor, ...]

        
        out = self.decode_head(outs)
        # logits: (N, out_channel, H, W)
        
        return out


# ========================= Config =====================================

class EVA_02_Segmentation_Mona_Config(EVA_02_Segmentation_Config):
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
        
        super().__init__(
            backbone_cfg=backbone_cfg,
            decode_head_cfg=decode_head_cfg,
            threshold=threshold,
            loss_decode=loss_decode,
            ignore_index=ignore_index
        )
        
    # a property method to instaniate the model
    @property
    def model(self):
        return EVA_02_Segmentation_Mona(**self.__dict__) 
    

class EVA_02_Segmentation_Mona_SFP_Config(EVA_02_Segmentation_Config):
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
        super().__init__(
            backbone_cfg=backbone_cfg,
            decode_head_cfg=decode_head_cfg,
            threshold=threshold,
            loss_decode=loss_decode,
            ignore_index=ignore_index
        )
        
    # a property method to instaniate the model
    @property
    def model(self):
        return EVA_02_Segmentation_Mona_SFP(**self.__dict__) 


def EVA_02_Segmentation_Mona_large_fgseg_cfg():
    args = EVA_02_Segmentation_Mona_Config(
        backbone_cfg = dict(
            img_size=352,
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
            
            ft_cfg=dict(
                bottleneck=64, 
                adapter_scalar='learnable_scalar', 
            
                act_cfg = dict(act_type='GELU'), 
        
                adapter_layernorm_option="in",
                norm_cfg = dict(norm_type='LayerNorm'),
                
                dropout_layer = dict(
                    drop_type='Dropout',
                    drop_prob=0.1,
                    inplace=False
                ),
            ),
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

def EVA_02_Segmentation_Mona_SFP_large_fgseg_cfg():
    args = EVA_02_Segmentation_Mona_SFP_Config(
        backbone_cfg = dict(
            img_size=352,
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
            
            ft_cfg=[
                dict(
                    type='backbone_ft',
                    bottleneck=64, 
                    adapter_scalar='learnable_scalar', 
                
                    act_cfg = dict(act_type='GELU'), 
            
                    adapter_layernorm_option="in",
                    norm_cfg = dict(norm_type='LayerNorm'),
                    
                    dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.1,
                        inplace=False
                    ),
                ),
                dict(
                    type='neck_ft',
                    out_channels=256,
                    scale_factors=[4.0, 2.0, 1.0, 0.5],
                    norm_cfg=dict(
                        norm_type='LayerNorm2d'   
                    ),
                )
            ]
        ), 
        
        decode_head_cfg = dict(
            in_channels=[256, 256, 256, 256],#
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
                
            in_index=[0, 1, 2, 3],  
            align_corners=False,
                
        ), 
        
        threshold=0.5,
        loss_decode=dict(
            loss_type='BCEWithLogitsLoss',
            reduction = 'mean',
        ),
    )
    
    return args


# ==================== End =============================================
# ======================================================================



# ====================== deprecated ===============================================================================
#==================================================================================================================
# ============== InvertedSwiGLU_Adapter + ViTDet ==================================================================

from ..backbones.eva02_ft import EVA_02_VisionTransformer_InvertedSwiGLUAdapter_SFP
class EVA_02_Segmentation_InvertedSwiGLUAdapter_SFP(nn.Module):
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
        
        self.backbone = EVA_02_VisionTransformer_InvertedSwiGLUAdapter_SFP(**backbone_cfg)
        
        self.decode_head = SegformerHead(**decode_head_cfg)
        
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
        # List: [Tensor, ...]

        
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


# ============== Config =====================

class EVA_02_Segmentation_InvertedSwiGLUAdapter_SFP_Config():
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
        return EVA_02_Segmentation_InvertedSwiGLUAdapter_SFP(**self.__dict__) 



def EVA_02_Segmentation_base_InvertedSwiGLUAdapter_SFP_fgseg_cfg():
    args = EVA_02_Segmentation_InvertedSwiGLUAdapter_SFP_Config(
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
            
            ft_cfg=[
                dict(
                    type='backbone_ft',
                    bottleneck=48, 
                    subln=True,
                    adapter_scalar='learnable_scalar',#'0.1', 
                    
                    act_cfg=dict(
                        act_type='SiLU', 
                    ),     
                    adapter_layernorm_option='none',
                            
                    dropout_layer = dict(
                    drop_type='Dropout',
                    drop_prob=0.0,
                    inplace=False)
                ),
                dict(
                    type='neck_ft',
                    out_channels=256,
                    scale_factors=[4.0, 2.0, 1.0, 0.5],
                    norm_cfg=dict(
                        norm_type='LayerNorm2d'   
                    ),
                )
            ]
        ), 
        
        decode_head_cfg = dict(
            in_channels=[256, 256, 256, 256],#
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
                
            in_index=[0, 1, 2, 3],  
            align_corners=False,
                
        ), 
        
        threshold=0.5,
        loss_decode=dict(
            loss_type='BCEWithLogitsLoss',
            reduction = 'mean',
        ),
    )

    return args


def EVA_02_Segmentation_large_InvertedSwiGLUAdapter_SFP_fgseg_cfg():
    args = EVA_02_Segmentation_InvertedSwiGLUAdapter_SFP_Config(
        backbone_cfg = dict(
            img_size=352,
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
            
            ft_cfg=[
                dict(
                    type='backbone_ft',
                    bottleneck=32, 
                    subln=True,
                    adapter_scalar='learnable_scalar',#'0.1', 
                    
                    act_cfg=dict(
                        act_type='SiLU', 
                    ),     
                    adapter_layernorm_option='none',
                            
                    
                ),
                dict(
                    type='neck_ft',
                    out_channels=256,
                    scale_factors=[4.0, 2.0, 1.0, 0.5],
                    norm_cfg=dict(
                        norm_type='LayerNorm2d'   
                    ),
                )
            ]
        ), 
        
        decode_head_cfg = dict(
            in_channels=[256, 256, 256, 256],#
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
                
            in_index=[0, 1, 2, 3],  
            align_corners=False,
                
        ), 
        
        threshold=0.5,
        loss_decode=dict(
            loss_type='BCEWithLogitsLoss',
            reduction = 'mean',
        ),
    )
    
    return args












