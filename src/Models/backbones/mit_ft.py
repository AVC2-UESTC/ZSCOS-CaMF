import torch.nn as nn
import torch
import torch.utils.checkpoint as cp
import math

from mmengine.model import (constant_init, normal_init,
                                        trunc_normal_init)

from ..utils import (build_activation_layer, build_dropout, build_norm_layer,
                     nlc_to_nchw, nchw_to_nlc)
from ..utils import PatchEmbed


from ..utils.attention import MultiheadAttention

from .mit import MixFFN, EfficientMultiheadAttention
from ..finetunes import Adapter

# implementation of 'Transformer Block'
# without 'overlap patch merging'
# add Adaptor finetuning
class TransformerEncoderLayer_Adapter(nn.Module):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
            
        ft_cfg (dict): fintune config. See src.Models.fintunes.
            Default: dict(
                     bottleneck=64, 
                     adapter_scalar='0.4', 
                     
                     act_cfg=dict(
                        act_type='ReLU', 
                        inplace=False
                     ),     
                     adapter_layernorm_option='none',
                               
                     dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
                     )
            
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 
                 act_cfg=dict(
                     act_type='GELU',
                 ),
                 
                 norm_cfg=dict(norm_type='LayerNorm', 
                     requires_grad=True,
                     ),
                 
                 ft_cfg = dict(
                     bottleneck=64, 
                     adapter_scalar=0.4, 
                     
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
                 
                 batch_first=True,
                 sr_ratio=1,
                 with_cp=False):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(embed_dims, **norm_cfg)[1]

        self.attn = EfficientMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer={'drop_type': 'DropPath', 
                           'drop_prob': drop_path_rate, 
                           'inplace': None},
            batch_first=batch_first,
            qkv_bias=qkv_bias,
            norm_cfg=norm_cfg,
            sr_ratio=sr_ratio)

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(embed_dims, **norm_cfg)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer={'drop_type': 'DropPath', 
                           'drop_prob': drop_path_rate, 
                           'inplace': None},
            act_cfg=act_cfg)
        
        self.adapter = Adapter(
            in_channels=embed_dims,
            **ft_cfg
        )

        self.with_cp = with_cp

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), hw_shape, identity=x)
            
            norm2_x = self.norm2(x)
            
            inner_out = self.ffn(norm2_x, hw_shape, identity=x) + self.adapter(norm2_x, add_residual=False)
            
            return inner_out

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x



class MixVisionTransformer_Adapter(nn.Module):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
            
        ft_cfg (dict): fintune config.
        
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg={'act_type': 'GELU'},
                 norm_cfg=dict(norm_type='LayerNorm', 
                     requires_grad=True,
                     ),
                 
                 ft_cfg=dict(
                     bottleneck=64, 
                     adapter_scalar=0.1, 
                     
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
                #  pretrained=None,
                 with_cp=False):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages
        
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = nn.ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg)
            layer = nn.ModuleList([
                TransformerEncoderLayer_Adapter(
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    ft_cfg=ft_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ]) # build N(num_layer) TransformerEncoderLayer
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(embed_dims_i, **norm_cfg)[1]
            self.layers.append(nn.ModuleList([patch_embed, layer, norm]))
            cur += num_layer
        # End for 
        
            
    
    def forward(self, x):
        '''
        
        Return: outs: 4 outputs from 4 stages
            Stage 1: shape(N, c_1, h/4, w/4)
            Stage 2: shape(N, c_2, h/8, w/8)
            Stage 3: shape(N, c_3, h/16, w/16)
            Stage 4: shape(N, c_4, h/32, w/32)
            
        '''
        outs = []

        for i, layer in enumerate(self.layers):# self.layers: [stage1, stage2, stage3, stage4]
            x, hw_shape = layer[0](x)   # Each stage: [patch_embed, layer, norm]
            for block in layer[1]: # Each layer: N(num_layer) TransformerEncoderLayer
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs
    
        
  
  
#====================builder=====================================


    
        
#=====================Config=======================================


        










