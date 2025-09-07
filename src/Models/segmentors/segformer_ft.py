import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torch import Tensor



from ..backbones import MixVisionTransformer_Adapter
from ..decode_heads import SegformerHead
from ..losses import accuracy
from ..builder import build_loss

from ..utils import resize


'''

SegFormer 的基本数据流：

输入图像首先通过选择的 backbone（例如 mit_b0、mit_b1 等）进行特征提取。
这些特征（通常是多尺度的）被传递给 SegFormerHead。
在 SegFormerHead 中，特征首先通过 MLP 被转化为所需的嵌入维度。
这些嵌入特征然后被上采样并融合。
最后，融合的特征通过一个卷积层得到最终的语义分割结果。

'''

class SegFormer_Adapter(nn.Module):
    '''
    Args: 
        backbone_cfg (dict):
            Default: dict(
                in_channels=3,
                embed_dims=32,
                num_stages=4,
                num_layers=[2, 2, 2, 2],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1
            )
        
        decode_head_cfg (dict):
            Default: dict(
                in_channels=[32, 64, 160, 256],
                 channels=256,
                 num_classes=19,
                 out_channels=19,
                 dropout_ratio=0.1,
                 norm_cfg={ 
                           'norm_type': 'BatchNorm2d', 
                           'requires_grad': True,
                           'eps': 1e-5,
                           'momentum': 0.1, 
                           'affine': True,
                           'track_running_stats': True
                       },
                 act_cfg={'act_type': 'ReLU', 
                          'inplace': False},
                 in_index=[0, 1, 2, 3],
                  
                 align_corners=False
            )
            
            threshold: for binary segmentation
                Default: None

            loss_decode (dict): loss layer config
                Default: dict(
                    loss_type = 'CrossEntropyLoss',  
                    reduction = 'mean',
                )
            
            ignore_index (int): make sure: ignore_index == loss_decode['layer_args'][0]
                i.e:    ignore_index = 255
                        loss_decode = dict(
                            ...
                            layer_args = [255, _]
                        )
                        
                Default: 255
         
                
    Shape:
        loss mode:
            inputs: inputs: (N, C, H, W)
                    label: (N, 1, H, W) indices
                        in binary segmentation:
                            (N, 1, H, W) ranging 0 ~ 1
            
            outputs: {
                'acc_seg': acc_value,
            
                'loss_name1': loss_value1,
                ...
                }

        predict mode:
            inputs: (N, C, H, W)
            
            outputs: ()
                
    
    '''

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
        
        self.backbone = MixVisionTransformer_Adapter(**backbone_cfg)
        
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
        # 4 feature maps
        
        out = self.decode_head(outs)
        # logits: (N, out_channel, H, W)
        
        return out
        
        
    
    def loss(self, inputs: Tensor, target: Tensor,
             return_logits = False
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
    

    def predict(self, inputs: Tensor,
                return_logits = False
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
        if self.out_channels == 1:
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
        
        
        
    
#=======================config=================================

class segformer_adapter_config():
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
        return SegFormer_Adapter(**self.__dict__)
    


def segformer_adapter_b0_cfg():
        
    segformer_b0_args = segformer_adapter_config(
            backbone_cfg = dict(
                in_channels=3,
                embed_dims=32,
                num_stages=4,
                num_layers=[2, 2, 2, 2],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1, 
                
                ft_cfg=dict(
                     bottleneck=64, 
                     adapter_scalar='0.1', 
                     
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
            ),
    
            decode_head_cfg = dict(
                in_channels=[32, 64, 160, 256],
                 channels=256,
                 num_classes=4,
                 out_channels=4,
                #  threshold=0.5,
                 dropout_ratio=0.1,
                 norm_cfg={ 
                           'norm_type': 'BatchNorm2d', 
                           'requires_grad': True,
                           'eps': 1e-5,
                           'momentum': 0.1, 
                           'affine': True,
                           'track_running_stats': True
                       },
                 act_cfg={'act_type': 'ReLU', 
                          'inplace': False},
                 in_index=[0, 1, 2, 3],
                  # possibly mutiple inputs with different sizes, channels
                 
                #  loss_decode=dict(
                #      loss_type='BCEWithLogitsLoss',
                #      reduction = 'mean',
                     
                #      ),
                #  ignore_index=255,
                 align_corners=False,
            ), 

            
            loss_decode=dict(
                loss_type='CrossEntropyLoss',
                reduction = 'mean',
                layer_args = [255, 0.0]
            ),
            
            ignore_index=255,
        
        )
        
    return segformer_b0_args



def segformer_adapter_b0_sod_cfg():
        
    segformer_b0_args = segformer_adapter_config(
            backbone_cfg = dict(
                in_channels=3,
                embed_dims=32,
                num_stages=4,
                num_layers=[2, 2, 2, 2],
                num_heads=[1, 2, 5, 8],
                patch_sizes=[7, 3, 3, 3],
                sr_ratios=[8, 4, 2, 1],
                out_indices=(0, 1, 2, 3),
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.1, 
                
                ft_cfg=dict(
                     bottleneck=64, 
                     adapter_scalar='0.1', 
                     
                     act_cfg=dict(
                        act_type='ReLU', 
                        inplace=False
                     ),     
                     adapter_layernorm_option='none',
                               
                     dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
                     ),
            ),
    
            decode_head_cfg = dict(
                in_channels=[32, 64, 160, 256],
                 channels=256,
                 num_classes=2,
                 out_channels=1,
                #  threshold=0.5,
                 dropout_ratio=0.1,
                 norm_cfg={ 
                           'norm_type': 'BatchNorm2d', 
                           'requires_grad': True,
                           'eps': 1e-5,
                           'momentum': 0.1, 
                           'affine': True,
                           'track_running_stats': True
                       },
                 act_cfg={'act_type': 'ReLU', 
                          'inplace': False},
                 in_index=[0, 1, 2, 3],
                  # possibly mutiple inputs with different sizes, channels
                 
                
                 align_corners=False,
            ), 

            threshold=0.5,
            loss_decode=dict(
                loss_type='BCEWithLogitsLoss',
                reduction = 'mean',
            ),
            
        
        )
        
    return segformer_b0_args

# ============ b2 ================================

def segformer_adapter_b2_sod_cfg():
        
    segformer_b2_args = segformer_adapter_config(
            backbone_cfg = dict(
                in_channels=3,
                embed_dims=64,#
                num_layers=[3, 4, 6, 3],#
                num_heads=[1, 2, 5, 8],#

                drop_path_rate=0.1,
                
                ft_cfg=dict(
                     bottleneck=64, 
                     adapter_scalar='learnable_scalar',#'0.1', 
                     
                     act_cfg=dict(
                        act_type='ReLU', 
                        inplace=False
                     ),     
                     adapter_layernorm_option='none',
                               
                     dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
                     ),
            ),
    
            decode_head_cfg = dict(
                in_channels=[64, 128, 320, 512],#
                 channels=256,
                 num_classes=2,
                 out_channels=1,
                 
                 norm_cfg={ 
                           'norm_type': 'BatchNorm2d', 
                           'requires_grad': True,
                           'eps': 1e-5,
                           'momentum': 0.1, 
                           'affine': True,
                           'track_running_stats': True
                       },
                 in_index=[0, 1, 2, 3],  
                 align_corners=False,
            ), 
            
            threshold=0.5,
            loss_decode=dict(
                loss_type='BCEWithLogitsLoss',
                reduction = 'mean',
            ),
            
        
        )
        
    return segformer_b2_args


# =======================b4===========================

def segformer_adapter_b4_sod_cfg():
        
    segformer_b4_args = segformer_adapter_config(
            backbone_cfg = dict(
                in_channels=3,
                embed_dims=64,#
                num_layers=[3, 8, 27, 3],#
                num_heads=[1, 2, 5, 8],#

                drop_path_rate=0.1,
                
                ft_cfg=dict(
                     bottleneck=64, 
                     adapter_scalar='learnable_scalar',#'0.1', 
                     
                     act_cfg=dict(
                        act_type='ReLU', 
                        inplace=False
                     ),     
                     adapter_layernorm_option='none',
                               
                     dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
                     ),
            ),
    
            decode_head_cfg = dict(
                in_channels=[64, 128, 320, 512],#
                 channels=256,
                 num_classes=2,
                 out_channels=1,
                 
                 norm_cfg={ 
                           'norm_type': 'BatchNorm2d', 
                           'requires_grad': True,
                           'eps': 1e-5,
                           'momentum': 0.1, 
                           'affine': True,
                           'track_running_stats': True
                       },
                 in_index=[0, 1, 2, 3],  
                 align_corners=False,
            ), 
            
            threshold=0.5,
            loss_decode=dict(
                loss_type='BCEWithLogitsLoss',
                reduction = 'mean',
            ),
            
        
        )
        
    return segformer_b4_args


# ================================= Module ==============================

from ..backbones.mit_IMGNetPretrined_ft import MixVisionTransformer_Adapter as mit_IMG_Adapter
from .segformer import SegFormer
from .base_segmenter import BaseSegmenter, BaseSegmentor_Config

from typing import Dict
from einops import rearrange

class SegFormer_IMG_Adapter(BaseSegmenter):
    def __init__(self, 
                 backbone_cfg: dict, 
                 decode_head_cfg: dict,
                  
                 threshold=None, 
                 loss_decode=dict(loss_type='CrossEntropyLoss', 
                                  reduction='mean'), 
                 ignore_index=255, 
                 align_corners: bool = False) -> None:
        super().__init__(threshold, loss_decode, ignore_index, align_corners)
                
        self.backbone = mit_IMG_Adapter(**backbone_cfg)
        
        self.decode_head = SegformerHead(**decode_head_cfg)
        
            
            
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
        
        
    def forward(self, inputs: Dict[str, Tensor])->Dict:
        x = inputs['image']
        
        enc_out = self.backbone(x)
       
        dec_out = self.decode_head(enc_out)
        # 
        results = dict(
            logits_mask= dec_out,
        )
        
        return results
 

# =================== Config =========================

class Segformer_IMG_Adapter_Config(BaseSegmentor_Config):
    '''
    
    '''
    def __init__(self, 
            pretrained_weights: str = None, 
            finetune_weights: str = None, 
            tuning_mode: str = 'PEFT', 
            
            backbone_cfg: dict = None, 
            decode_head_cfg: dict = None,
                
            threshold=None, 
            loss_decode=None, 
            ignore_index=255, 
            align_corners: bool = True) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)
        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
        
    def set_model_class(self):
        self.model_class = SegFormer_IMG_Adapter
     




def segformerIMG_Adapter_b4_sod_cfg():
    segformer_b4_args = Segformer_IMG_Adapter_Config(
            backbone_cfg = dict(
                in_chans=3, 
                 embed_dims=64,
                 num_heads=[1, 2, 5, 8], 
                 patch_size=[7, 3, 3, 3], 
                 strides=[4, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=True, 
                 drop_rate=0.,
                 drop_path_rate=0.1, 
                 norm_cfg=dict(
                     norm_type='LayerNorm', 
                     requires_grad=True),
                 depths=[3, 8, 27, 3], 
                 sr_ratios=[8, 4, 2, 1],
                 
                 ft_cfg=dict(
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
                 
            ),
    
            decode_head_cfg = dict(
                in_channels=[64, 128, 320, 512],#
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
            
            threshold=None,
            loss_decode=[
            dict(
                loss_type='BCEWithLogitsLoss',
                reduction = 'mean',
                loss_weight=1.0,
            ),
            dict(
                loss_type='DiceLoss',
                reduction='mean',
                loss_weight=0.5,
            )
        ]          
        )
        
    return segformer_b4_args





