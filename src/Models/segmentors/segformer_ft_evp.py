
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor

from functools import partial


from ..losses import accuracy
from ..builder import build_loss

from ..utils import resize

from ..backbones.mit_ft_evp import MixVisionTransformerEVP
from  ..decode_heads.decode_head import BaseDecodeHead

from ..utils import ConvModule, resize

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(BaseDecodeHead):
    def __init__(self, 
                 interpolate_mode='bilinear', 
                 **kwargs
                 ):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        
        assert num_inputs == len(self.in_index)
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        embedding_dim = self.channels
        
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            conv_cfg=dict(conv_type = 'Conv2d'),
            norm_cfg=dict(norm_type='BatchNorm2d', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.out_channels, kernel_size=1)



    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x



from .base_segmenter import BaseSegmenter, BaseSegmentor_Config



class Segformer_EVP(BaseSegmenter):
    def __init__(self, 
                backbone_cfg: dict, 
                decode_head_cfg: dict,
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                align_corners: bool = False
                
                ) -> None:
        super().__init__(threshold=threshold, loss_decode=loss_decode, ignore_index=ignore_index, align_corners=align_corners)
        
        self.backbone = MixVisionTransformerEVP(**backbone_cfg)
        
        self.decode_head = SegFormerHead(**decode_head_cfg)
        
        
        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')
            
            
        
        # self.loss_decode: loss layers in Modulelist


    def forward(self, inputs: Dict):
        # x = inputs['image']
        outs = self.backbone(inputs) 
        # 4 feature maps
        
        out = self.decode_head(outs)
        # logits: (N, out_channel, H, W)
        
        return dict(logits_mask=out)
    




class Segformer_EVP_Config(BaseSegmentor_Config):
    def __init__(self, 
                 pretrained_weights = None, 
                 finetune_weights = None, 
                 tuning_mode = 'PEFT', 
                 
                 backbone_cfg: Dict = None, 
                 decode_head_cfg: Dict = None,
                 
                 threshold=None, 
                 loss_decode=..., 
                 ignore_index=255, 
                 align_corners = False):
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)
    #
        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
    # a property method to instaniate the model
    def set_model_class(self):
        self.model_class = Segformer_EVP



def Segformer_EVP_fgseg_cfg():
    args = Segformer_EVP_Config(
        backbone_cfg=dict(
            # img_size=352, 
            img_size=384,
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 8, 27, 3], 
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, 
            drop_path_rate=0.1,
            
            scale_factor=4, 
            input_type='fft', 
            freq_nums=0.25,
            prompt_type='highpass',
            tuning_stage='1234',
            handcrafted_tune=True,
            embedding_tune=True,
            adaptor='adaptor'  
        ),
        
        decode_head_cfg = dict(
            in_channels=[64, 128, 320, 512],#
            channels=768,
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
        
        threshold=None,
        loss_decode=[
            dict(
                loss_type='BCEWithLogitsLoss',
                reduction = 'mean',
                loss_weight=1.0,
                loss_name = 'mask_loss_bce',
            ),
            dict(
                loss_type='DiceLoss',
                reduction='mean',
                loss_weight=0.5,
                loss_name='mask_loss_dice',
            ),   
        ],
        
    )
    
    return args








