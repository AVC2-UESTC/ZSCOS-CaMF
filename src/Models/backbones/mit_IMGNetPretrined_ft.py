import torch.nn as nn
import torch
import torch.utils.checkpoint as cp
import math

from mmengine.model import (constant_init, normal_init,
                                        trunc_normal_init)
from mmengine.utils import to_2tuple

from ..utils import (build_activation_layer, build_dropout, build_norm_layer,
                     nlc_to_nchw, nchw_to_nlc)
from ..utils import PatchEmbed
from ..utils.drop import DropPath

from ..utils.attention import MultiheadAttention

from .mit_IMGNetPretrined import Mlp, Attention
from ..finetunes import Adapter


class Block_Adapter(nn.Module):

    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_cfg=dict(
                     act_type='GELU'), 
                 norm_cfg=dict(
                     norm_type='LayerNorm', 
                     requires_grad=True), 
                 
                 ft_cfg=dict(
                     bottleneck=64, 
                     adapter_scalar=0.1, 
                     
                     act_cfg=dict(
                        act_type='ReLU', 
                        layer_args=dict(inplace=False)
                     ),     
                     adapter_layernorm_option='none',
                               
                     dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
                     ),
                 
                 sr_ratio=1):
        super().__init__()
        self.norm1 = build_norm_layer(dim, **norm_cfg)[1]
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(dim, **norm_cfg)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_cfg=act_cfg, drop=drop)

        self.adapter = Adapter(
            in_channels=dim,
            **ft_cfg
        )
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W)) + self.adapter(x, add_residual=False)

        return x


class MixVisionTransformer_Adapter(nn.Module):
    def __init__(self, 
                 in_chans=3, 
                 embed_dims=64,
                 num_heads=[1, 2, 4, 8], 
                 patch_size=[7, 3, 3, 3], 
                 strides=[4, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0., 
                 norm_cfg=dict(
                     norm_type='LayerNorm', 
                     requires_grad=True),
                 
                 ft_cfg=dict(
                     bottleneck=64, 
                     adapter_scalar='0.1', 
                     
                     act_cfg=dict(
                        act_type='ReLU', 
                        layer_args=dict(inplace=False)
                     ),     
                     adapter_layernorm_option='none',
                               
                     dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
                     ),
                 
                 depths=[3, 4, 6, 3], 
                 sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths = depths
        
        self.embed_dims = [embed_dims * i for i in num_heads]

        # patch_embed
        
        self.patch_embed1 = PatchEmbed(
                 in_channels=in_chans,
                 embed_dims=self.embed_dims[0],
                 kernel_size=patch_size[0],
                 stride=strides[0],
                 padding=patch_size[0] // 2,
                )
        
        self.patch_embed2 = PatchEmbed(
            in_channels=self.embed_dims[0],
            embed_dims=self.embed_dims[1],
            kernel_size=patch_size[1],
            stride=strides[1],
            padding=patch_size[1] // 2,
        )
        
        self.patch_embed3 = PatchEmbed(
            in_channels=self.embed_dims[1],
            embed_dims=self.embed_dims[2],
            kernel_size=patch_size[2],
            stride=strides[2],
            padding=patch_size[2] // 2,
        )
        
        self.patch_embed4 = PatchEmbed(
            in_channels=self.embed_dims[2],
            embed_dims=self.embed_dims[3],
            kernel_size=patch_size[3],
            stride=strides[3],
            padding=patch_size[3] // 2,
        )
        

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block_Adapter(
            dim=self.embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[0], ft_cfg=ft_cfg)
            for i in range(depths[0])])
        self.norm1 = build_norm_layer(self.embed_dims[0], **norm_cfg)[1]

        cur += depths[0]
        self.block2 = nn.ModuleList([Block_Adapter(
            dim=self.embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[1], ft_cfg=ft_cfg)
            for i in range(depths[1])])
        self.norm2 = build_norm_layer(self.embed_dims[1], **norm_cfg)[1]

        cur += depths[1]
        self.block3 = nn.ModuleList([Block_Adapter(
            dim=self.embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[2], ft_cfg=ft_cfg)
            for i in range(depths[2])])
        self.norm3 = build_norm_layer(self.embed_dims[2], **norm_cfg)[1]

        cur += depths[2]
        self.block4 = nn.ModuleList([Block_Adapter(
            dim=self.embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_cfg=norm_cfg,
            sr_ratio=sr_ratios[3], ft_cfg=ft_cfg)
            for i in range(depths[3])])
        self.norm4 = build_norm_layer(self.embed_dims[3], **norm_cfg)[1]

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, (H, W) = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, (H, W) = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x
    
    
    
# =====================================================================================================
# ==================================== EVP ============================================================


    






























