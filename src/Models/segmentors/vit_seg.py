import warnings
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor
from einops import rearrange

from .base_segmenter import BaseSegmenter, BaseSegmentor_Config
from ..backbones.vit_ft import VisionTransformer_Adapter, VisionTransformer_Adapter_SFP
# from ..backbones.cmae_vit_ft import CMAE_VisionTransformer_Adapter_SFP

from ..decode_heads.decode_head import SimpleMaskDecoder
from ..decode_heads.segformer_head import SegformerHead_single_input, SegformerHead

from ..utils import resize



class ViT_FGSeg(BaseSegmenter):
    def __init__(self, 
                 backbone_cfg: Dict, 
                 decode_head_cfg: Dict,
                 
                 threshold: float = None, 
                 loss_decode = None, 
                 ignore_index: int = 255, 
                 align_corners: bool = False) -> None:
        super().__init__(threshold, loss_decode, ignore_index, align_corners)
        
        self.backbone = VisionTransformer_Adapter_SFP(**backbone_cfg)
        # self.backbone = CMAE_VisionTransformer_Adapter_SFP(**backbone_cfg)
        self.decode_head = SegformerHead(**decode_head_cfg)
        
        
        
    def forward(self, inputs: Tensor) -> Tensor:
        img = inputs['image']
        # enc_out, hw_size = self.backbone(img) 
        
        # enc_out = rearrange(enc_out, 'b (h w) c -> b c h w', h=hw_size[0], w=hw_size[1])
        # outputs = self.decode_head([enc_out, ])
        
        enc_out = self.backbone(img) 
        
        outputs = self.decode_head(enc_out)
        out_dict = dict(
            logits_mask = outputs
        )
        return out_dict
    
    
    def visualize_backbone_attn(self, inputs, layer):
        img = inputs['image']
        h, w = img.shape[2:]
        patch_size = self.backbone.patch_size
        grid_size = h // patch_size
        
        outs = self.backbone.get_last_selfattention(img, layer)
        nh = outs.shape[1]
        attn = outs[0, :, 0, 1:]
        attn = attn.reshape(nh, grid_size, grid_size)
        attn = resize(attn.unsqueeze(0), size=(h, w), mode='nearest')
        return dict(
            attn_map=attn
        )
    
    
class ViT_FGSeg_Config(BaseSegmentor_Config):
    def __init__(self, 
                 backbone_cfg: Dict,
                 decode_head_cfg: Dict,
                 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'PEFT', 
                 
                 threshold=None, 
                 loss_decode=None, 
                 ignore_index=255, 
                 align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)

        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
    def set_model_class(self):
        self.model_class = ViT_FGSeg















