import torch
import torch.nn as nn


from  .decode_head import BaseDecodeHead
from ..utils import ConvModule, resize

from torch import Tensor





class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """
    def __init__(self, 
                 interpolate_mode='bilinear', 
                 **kwargs
                 ):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        
        assert num_inputs == len(self.in_index)
        
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    conv_cfg=dict(conv_type = 'Conv2d'),
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            conv_cfg=dict(conv_type = 'Conv2d'),
            norm_cfg=self.norm_cfg)
        
        
    def forward(self, inputs) -> Tensor:
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        # each feature map: (N, c, h, w)
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out) # shape(n, num_classes, h, w)

        return out


class SegformerHead_single_input(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """
    def __init__(self, 
                 interpolate_mode='bilinear', 
                 resize_ratio = 4,
                 **kwargs
                 ):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        self.resize_ratio = resize_ratio
        
        num_inputs = len(self.in_channels)
        
        assert num_inputs == len(self.in_index)
        
        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    conv_cfg=dict(conv_type = 'Conv2d'),
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            conv_cfg=dict(conv_type = 'Conv2d'),
            norm_cfg=self.norm_cfg)
        
        
    def forward(self, inputs) -> Tensor:
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        # each feature map: (N, c, h, w)
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            target_size = []
            for size in inputs[0].shape[2:]:
                target_size.append(int(size * self.resize_ratio))
            target_size = tuple(target_size)
            
            outs.append(
                resize(
                    input=conv(x),
                    size=target_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out) # shape(n, num_classes, h, w)
        outputs = dict(
            logits_mask=out,
        )
        return outputs






