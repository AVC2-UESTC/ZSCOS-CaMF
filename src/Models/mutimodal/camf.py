import warnings
import math

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor

from ..builder import build_loss

from ..utils import resize

from ..segmentors import BaseSegmenter, BaseSegmentor_Config


from ..backbones.eva02_ft import EVA_02_VisionTransformer_Adapter_SFP
from ..decode_heads.CAMF_decoder import camf_decoder

class CAMF(BaseSegmenter):
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
        
        self.backbone = EVA_02_VisionTransformer_Adapter_SFP(**backbone_cfg)
        
        self.decode_head = camf_decoder(**decode_head_cfg)
        
        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')
            
            
    def forward(self, inputs: Dict):
        prompt = inputs['prompt']
        x = inputs['image']
        
        outs = self.backbone(x) 
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        
        
        results = self.decode_head(outs, prompt)
        # results: dict(
        #   logits: (N, out_channel, H, W), 
        # )
        # 
        
        return results
    
    def infer_forward(self, inputs: Dict):
        x = inputs['image']
        
        outs = self.backbone(x) 
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        
        # no caption
        # prompt = torch.zeros_like(prompt)
        # random caption
        # prompt = prompt + torch.randn_like(prompt)
        # prompt = torch.rand_like(prompt)
        # prompt = inputs['prompt']
        # prompt = torch.randint_like(prompt, 0, 50272)
        # prompt = torch.zeros_like(prompt)
        # prompt = None
        prompt = "query"
        results = self.decode_head.infer_forward(outs, prompt_idx=prompt)
        # results: dict(
        #   logits: (N, out_channel, H, W), 
        # )
        # 
        
        return results
    
    # def tm_visual(self, inputs: Dict):
    #     x = inputs['image']
        
    #     outs = self.backbone(x) 
    #     # List: [Tensor, ...]
    #     #   Tensor: (N, c, h, w). c = 256
        
    #     prompt = None
    #     results = self.decode_head.get_matched_token(outs, prompt_idx=prompt)
    #     # results: dict(
    #     #   tm_map: (N, out_channel, H, W), 
    #     # )
    #     # 
        
    #     return results
    
    
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
        
        logits_img_txt = results['logits_img_txt']
        logits_img_q = results['logits_img_q']
        
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
            if loss_decode.loss_name.startswith('mask_'):
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
            elif loss_decode.loss_name.startswith('alg_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        logits_img_q,
                        logits_img_txt,#(N, H, W)
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        logits_img_q,
                        logits_img_txt,
                        )
            elif loss_decode.loss_name.startswith('maskreg_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_logits,#(N, H, W)
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_logits,
                        )
            else:
                raise ValueError(f'loss name: {loss_decode.loss_name} is not supported')
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
        
        results = self.infer_forward(inputs)
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
        if self.decode_head.out_channels == 1:
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
        


class CAMF_Config(BaseSegmentor_Config):
    def __init__(self, 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'PEFT', 
                 
                 backbone_cfg: dict=None, 
                 decode_head_cfg: dict=None,
                 
                 threshold=None, 
                 loss_decode=..., 
                 ignore_index=255, 
                 align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)

        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
    def set_model_class(self):
        self.model_class = CAMF
        
        
        
        
        
        
from ..backbones.eva02_ft import EVA_02_VisionTransformer_LoRA_SFP

        
        
class CAMF_LoRA(BaseSegmenter):
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
        
        self.backbone = EVA_02_VisionTransformer_LoRA_SFP(**backbone_cfg)
        
        # self.decode_head = Muv_decoder_v3_Multi_scale(**decode_head_cfg)
        self.decode_head = camf_decoder(**decode_head_cfg)
        
        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')
            
            
    def forward(self, inputs: Dict):
        prompt = inputs['prompt']
        x = inputs['image']
        
        outs = self.backbone(x) 
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        
        # no caption
        # prompt = torch.zeros_like(prompt)
        # random caption
        # prompt = prompt + torch.randn_like(prompt)
        # prompt = torch.rand_like(prompt)
        
        results = self.decode_head(outs, prompt)
        # results: dict(
        #   logits: (N, out_channel, H, W), 
        # )
        # 
        
        return results
    
    def infer_forward(self, inputs: Dict):
        x = inputs['image']
        
        outs = self.backbone(x) 
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        
        # no caption
        # prompt = torch.zeros_like(prompt)
        # random caption
        # prompt = prompt + torch.randn_like(prompt)
        # prompt = torch.rand_like(prompt)
        # prompt = inputs['prompt']
        prompt = None
        results = self.decode_head.infer_forward(outs, prompt_idx=prompt)
        # results: dict(
        #   logits: (N, out_channel, H, W), 
        # )
        # 
        
        return results
    
    
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
        
        logits_img_txt = results['logits_img_txt']
        logits_img_q = results['logits_img_q']
        
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
            if loss_decode.loss_name.startswith('mask_'):
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
            elif loss_decode.loss_name.startswith('alg_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        logits_img_q,
                        logits_img_txt,#(N, H, W)
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        logits_img_q,
                        logits_img_txt,
                        )
            elif loss_decode.loss_name.startswith('maskreg_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_logits,#(N, H, W)
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_logits,
                        )
            else:
                raise ValueError(f'loss name: {loss_decode.loss_name} is not supported')
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
        
        results = self.infer_forward(inputs)
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
        if self.decode_head.out_channels == 1:
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
        


class CAMF_LoRA_Config(BaseSegmentor_Config):
    def __init__(self, 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'PEFT', 
                 
                 backbone_cfg: dict=None, 
                 decode_head_cfg: dict=None,
                 
                 threshold=None, 
                 loss_decode=..., 
                 ignore_index=255, 
                 align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)

        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
    def set_model_class(self):
        self.model_class = CAMF_LoRA
        
        
# from ..backbones.eva02_ft import EVA_02_VisionTransformer_LoRA_Adapter_SFP
        
      
# class CaMP_LoRA_Adapter(BaseSegmenter):
#     '''
#     Args:

#     '''
#     def __init__(self, 
#                  backbone_cfg: Dict, 
#                  decode_head_cfg: Dict, 
                 
#                  threshold: float = None, 
#                  loss_decode: Dict = dict(loss_type='CrossEntropyLoss', 
#                                   reduction='mean'),  
#                  ignore_index: int = 255, 
#                  align_corners: bool = False) -> None:
#         super().__init__(threshold=threshold, loss_decode=loss_decode, ignore_index=ignore_index, align_corners=align_corners)
        
#         self.backbone = EVA_02_VisionTransformer_LoRA_Adapter_SFP(**backbone_cfg)
        
#         # self.decode_head = Muv_decoder_v3_Multi_scale(**decode_head_cfg)
#         self.decode_head = Muv_decoder_v4(**decode_head_cfg)
        
#         out_channels = decode_head_cfg['out_channels']
        
#         if out_channels == 1 and threshold is None:
#             # threshold = 0.3
#             warnings.warn('threshold is not defined for binary')
            
            
#     def forward(self, inputs: Dict):
#         prompt = inputs['prompt']
#         x = inputs['image']
        
#         outs = self.backbone(x) 
#         # List: [Tensor, ...]
#         #   Tensor: (N, c, h, w). c = 256
        
        
#         # no caption
#         # prompt = torch.zeros_like(prompt)
#         # random caption
#         # prompt = prompt + torch.randn_like(prompt)
#         # prompt = torch.rand_like(prompt)
        
#         results = self.decode_head(outs, prompt)
#         # results: dict(
#         #   logits: (N, out_channel, H, W), 
#         # )
#         # 
        
#         return results
    
#     def infer_forward(self, inputs: Dict):
#         x = inputs['image']
        
#         outs = self.backbone(x) 
#         # List: [Tensor, ...]
#         #   Tensor: (N, c, h, w). c = 256
        
        
#         # no caption
#         # prompt = torch.zeros_like(prompt)
#         # random caption
#         # prompt = prompt + torch.randn_like(prompt)
#         # prompt = torch.rand_like(prompt)
#         # prompt = inputs['prompt']
#         prompt = None
#         results = self.decode_head.infer_forward(outs, prompt_idx=prompt)
#         # results: dict(
#         #   logits: (N, out_channel, H, W), 
#         # )
#         # 
        
#         return results
    
    
#     def loss(self, inputs: Dict[str, Tensor], labels: Dict[str, Tensor],
#              return_logits: bool = False
#              ) -> dict:
#         """Forward function for training.

#         Args:
            

#         Returns:
            
#         Shape:
#             inputs: dict(
#                 image: (N, C, H, W)
#             )
            
#             labels: dict(
#                 label_mask: (N, out_channel, H, W)
#             ) 
            
#         """
#         results = self.forward(inputs)
        
#         logits_img_txt = results['logits_img_txt']
#         logits_img_q = results['logits_img_q']
        
#         seg_logits = results['logits_mask']
        
#         seg_label = labels['label_mask']
        
#         logits_prob = torch.sigmoid(seg_logits) # for metric computing
        
        
#         seg_logits = resize(
#             input=seg_logits,
#             size=seg_label.shape[2:], #(N, 1, H, W)
#             mode='bilinear',
#             align_corners=self.align_corners)
        
#         seg_label = seg_label.squeeze(1)
#         # (N, H, W)
        
#         # Calculate loss
#         losses = dict()
        
#         if not isinstance(self.loss_decode, nn.ModuleList):
#             losses_decode = [self.loss_decode]
            
#         else:
#             losses_decode = self.loss_decode
#         # losses_decode: loss layer(s) in Modulelist
        
#         for loss_decode in losses_decode:
#             if loss_decode.loss_name.startswith('mask_'):
#                 if loss_decode.loss_name not in losses:
#                     losses[loss_decode.loss_name] = loss_decode(
#                         seg_logits,
#                         seg_label,#(N, H, W)
#                         )
#                 else:
#                     losses[loss_decode.loss_name] += loss_decode(
#                         seg_logits,
#                         seg_label,
#                         )
#             elif loss_decode.loss_name.startswith('alg_'):
#                 if loss_decode.loss_name not in losses:
#                     losses[loss_decode.loss_name] = loss_decode(
#                         logits_img_q,
#                         logits_img_txt,#(N, H, W)
#                         )
#                 else:
#                     losses[loss_decode.loss_name] += loss_decode(
#                         logits_img_q,
#                         logits_img_txt,
#                         )
#             elif loss_decode.loss_name.startswith('maskreg_'):
#                 if loss_decode.loss_name not in losses:
#                     losses[loss_decode.loss_name] = loss_decode(
#                         seg_logits,
#                         seg_logits,#(N, H, W)
#                         )
#                 else:
#                     losses[loss_decode.loss_name] += loss_decode(
#                         seg_logits,
#                         seg_logits,
#                         )
#             else:
#                 raise ValueError(f'loss name: {loss_decode.loss_name} is not supported')
#         # losses: {
#         #         
#         #         'loss_name1': loss_value1
#         #         ...
#         #     }
        
#         preds = dict(pred_mask=logits_prob)
        
#         if return_logits:
#             return losses, preds
#         else:
#             return losses
    

#     def predict(self, inputs: Dict,
#                 return_logits: bool = False
#                 ) -> Dict:
#         """Forward function for prediction.

#         Args:
#             inputs: 
            

#         Returns:
#             Tensor: Outputs segmentation logits map.
#         """
#         img_size = inputs['image'].shape[2:]
        
#         results = self.infer_forward(inputs)
#         seg_logits = results['logits_mask']
#         logits_prob = torch.sigmoid(seg_logits) # for metric computing
#         preds = dict(
#             pred_mask=logits_prob,
#         )
        
#         seg_logits = resize(
#             input=seg_logits,
#             size=img_size,
#             mode='bilinear',
#             align_corners=self.align_corners)
#         # (N, out_channels, H, W)
#         if self.decode_head.out_channels == 1:
#             seg_probs = torch.sigmoid(seg_logits)
#             if self.threshold is not None:
#                 seg_map = (seg_probs > self.threshold).float()
#             else:
#                 seg_map = seg_probs
#             # (N, 1, H, W)
#             seg_map = seg_map.squeeze(dim=1)
#             #(N, H, W)
#         else:
#             seg_probs = F.softmax(seg_logits, dim=1)
#             seg_map = torch.argmax(seg_probs, dim=1)
#             # (N, H, W)
        
#         outputs = dict(
#             pred_mask=seg_map
#         )
        
#         if return_logits:
#             return outputs, preds
#         else:
#             return outputs
        


# class CaMP_LoRA_Adapter_Config(BaseSegmentor_Config):
#     def __init__(self, 
#                  pretrained_weights: str = None, 
#                  finetune_weights: str = None, 
#                  tuning_mode: str = 'PEFT', 
                 
#                  backbone_cfg: dict=None, 
#                  decode_head_cfg: dict=None,
                 
#                  threshold=None, 
#                  loss_decode=..., 
#                  ignore_index=255, 
#                  align_corners: bool = False) -> None:
#         super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)

#         self.backbone_cfg = backbone_cfg
#         self.decode_head_cfg = decode_head_cfg
        
#     def set_model_class(self):
#         self.model_class = CaMP_LoRA_Adapter
        