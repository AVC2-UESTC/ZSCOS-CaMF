import warnings

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

from ..utils import resize 
from .. builder import build_loss
from ..losses import accuracy


from ..utils.ffn import MLP_FFN, ConvMLP_FFN
from ..utils import (build_norm_layer,
                     resize)

class SimpleMaskDecoder(nn.Module):
    def __init__(self, 
                 
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 
                 scale_factor: float,
                 dropout_ratio=0.1,
                 norm_cfg=dict(norm_type='LayerNorm2d', 
                               ),
                 
                 act_cfg=dict(
                    act_type='GELU',
                ),
                 interpolate_mode='bilinear', 
                 
                 ) -> None:
        super().__init__()
        
        self.scale_factor = scale_factor
        self.interpolate_mode = interpolate_mode
        self.out_channels = num_classes
        
        
        self.mlp1 = ConvMLP_FFN(in_features=in_channels, 
                            hidden_features=channels, 
                            out_features=channels, 
                            act_cfg=act_cfg, 
                            drop=dropout_ratio,
                            )
        self.norm = build_norm_layer(channels, **norm_cfg)[1]
        
        self.mlp2 = ConvMLP_FFN(
            in_features=channels, 
            hidden_features=channels,
            out_features=num_classes,
            act_cfg=act_cfg,
        )
        
        
    def forward(self, x: Tensor, hw_size: Tuple[int, int] = None) -> Tensor:
        '''
        Args:
            x (Tensor): Input feature map with shape (N, L, C).
            
        Returns:
            Tensor: (N, num_classes, h, w)
            
        
        '''
        
        if hw_size is not None:
            h, w = hw_size
        else:
            h = int(x.shape[1] ** 0.5)
            w = h 
            
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        x = self.norm(self.mlp1(x))
        
        x = resize(x, scale_factor=self.scale_factor, mode=self.interpolate_mode, align_corners=False)
        # x = rearrange(x, 'b c h w -> b (h w) c')
        
        x = self.mlp2(x)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        outputs = dict(
            logits_mask=x
        )
        
        return outputs
        
        
        



class BaseDecodeHead(nn.Module):
    """Base class for BaseDecodeHead.

    1. The ``init_weights`` method is used to initialize decode_head's
    model parameters. After segmentor initialization, ``init_weights``
    is triggered when ``segmentor.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of decode_head,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()
    
    

    3. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) the decode_head model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict segmentation results
    including post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    Args:
        in_channels (int|Sequence[int]): Input channels.
        
        channels (int): Channels after modules, before conv_seg.
        
        num_classes (int): Number of classes.
        
        out_channels (int): Output channels of conv_seg. Default: None.
            For binary segmentation, set num_classes to 2 and out_channels to 1.
        
        threshold (float): Threshold for binary segmentation in the case of
            `num_classes==1`. Default: None.
        
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        
        conv_cfg (dict|None): Config of conv layers. Default: None.
        
        norm_cfg (dict|None): Config of norm layers. Default: None.
        
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        
        in_index (int|Sequence[int]): Input feature index. Default: -1
        
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        
        loss_decode (dict | Sequence[dict]): Config of decode loss.
            The `loss_name` is property of corresponding loss function which
            could be shown in training log. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
             e.g. dict(type='CrossEntropyLoss'),
             [dict(type='CrossEntropyLoss', loss_name='loss_ce'),
              dict(type='DiceLoss', loss_name='loss_dice')]
            Default: dict(type='CrossEntropyLoss').
        
        ignore_index: 255
            bce loss does not require ignore_index
            make sure ignore_index = loss_decode[loss_type].layer_args[0] if using cross entropy loss.
        i.e:    ignore_index = 255
                loss_decode = {
                    'loss_type': 'CrossEntropyLoss', 
                    ...
                    'layer_args': [255, _]
                }
        
        
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        
        
    """

    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 out_channels=None,
                #  threshold=None,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(
                    act_type='ReLU',
                    layer_args=dict(inplace=False)     
                ),
                 in_index=-1,
                 input_transform=None, # possibly mutiple inputs with different sizes, channels
                 
                #  loss_decode=dict(
                #      loss_type='CrossEntropyLoss',
                #      reduction = 'mean',
                     
                #      ),
                 align_corners=False, # for resize
                 
                 ):
        super().__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        

        self.align_corners = align_corners
        
        if out_channels is None:
            if num_classes == 2:
                warnings.warn('For binary segmentation, we suggest using'
                              '`out_channels = 1` to define the output'
                              'channels of segmentor, and use `threshold`'
                              'to convert `seg_logits` into a prediction'
                              'applying a threshold')
            out_channels = num_classes

        if out_channels != num_classes and out_channels != 1:
            raise ValueError(
                'out_channels should be equal to num_classes,'
                'except binary segmentation set out_channels == 1 and'
                f'num_classes == 2, but got out_channels={out_channels}'
                f'and num_classes={num_classes}')

        # if out_channels == 1 and threshold is None:
        #     threshold = 0.3
        #     warnings.warn('threshold is not defined for binary, and defaults'
        #                   'to 0.3')
            
        self.num_classes = num_classes
        self.out_channels = out_channels
        # self.threshold = threshold
        
        
        # if isinstance(loss_decode, dict):
        #     self.loss_decode = build_loss(**loss_decode)
        # elif isinstance(loss_decode, (list, tuple)):
        #     self.loss_decode = nn.ModuleList()
        #     for loss in loss_decode:
        #         self.loss_decode.append(build_loss(**loss))
        # else:
        #     raise TypeError(f'loss_decode must be a dict or sequence of dict,\
        #         but got {type(loss_decode)}')
        # # self.loss_decode: loss layers in Modulelist
        

        self.conv_seg = nn.Conv2d(channels, self.out_channels, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        
        
        
        
    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels
            
            
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

        
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output
    
    
    
    
'''
    def loss(self, inputs: Tuple[Tensor], target: Tensor,
             ) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            
            Target (list[:obj:`SegDataSample`]): The seg map. 

        Returns:
            dict[str, Tensor]: a dictionary of loss components
            
        Shape:
            inputs: (im_feature 1, ..., im_feature N)
                im_feature: (N, C, H_i, W_i)
            
            target: (N, 1, H, W)
            
            losses: {
                'acc_seg': acc_value
            
                'loss_name1': loss_value1
                ...
            }
            
        
        """
        seg_logits = self.forward(inputs)
        
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
        
        losses = loss
        return losses


    def predict(self, inputs: Tuple[Tensor],
                ) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        seg_logits = self.forward(inputs)
        seg_logits = resize(
            input=seg_logits,
            size=batch_img_metas[0]['img_shape'],
            mode='bilinear',
            align_corners=self.align_corners)
        # (N, C, H, W)
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
        

        return seg_map
'''

    




