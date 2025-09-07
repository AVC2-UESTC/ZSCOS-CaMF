
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict

from .VST.ImageDepthNet import ImageDepthNet

from .base_segmenter import BaseSegmenter, BaseSegmentor_Config



class VST_Seg(BaseSegmenter):
    def __init__(self, 
                model_cfg = Dict,
                # dict(
                #     channel = 32, 
                #     imagenet_pretrained = False
                # )
                
                threshold: float = None, 
                loss_decode=dict(
                    loss_type='CrossEntropyLoss', 
                    reduction='mean'),
                
                ignore_index: int = 255,
                align_corners: bool = False,):
        super().__init__(threshold, loss_decode, ignore_index, align_corners)
        self.network = ImageDepthNet(**model_cfg)
        
        self.out_channels = 1
        
        
    def forward(self, inputs: Dict):
        # x = inputs['image']
        h, w = inputs['image'].shape[2:]
        out = self.network(inputs)
        out = self.network.postprocessor(out, h)
        
        results = dict(
            logits_mask = out,
        )
        
        return results
    
    
    
    
class VST_Seg_Config(BaseSegmentor_Config):
    def __init__(self, 
                 pretrained_weights = None, 
                 finetune_weights = None, 
                 tuning_mode = 'PEFT', 
                 
                 model_cfg = None,
                 
                 threshold=None, 
                 loss_decode=..., 
                 ignore_index=255, 
                 align_corners = False):
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)

        self.model_cfg = model_cfg
        
    def set_model_class(self):
        self.model_class = VST_Seg