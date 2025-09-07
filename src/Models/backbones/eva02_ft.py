import math 
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import xformers.ops as xops

from mmengine.utils import to_2tuple

from mmengine.model import (constant_init, normal_init,
                                        trunc_normal_init)

from ..utils import (build_activation_layer, build_dropout, build_norm_layer,
                     nlc_to_nchw, nchw_to_nlc)


from ..utils.drop import DropPath

from ..utils.rope import VisionRotaryEmbeddingFast

from ..finetunes import Adapter, SimpleFeaturePyramid

from .eva02 import (Mlp, SwiGLU, PatchEmbed, 
                  RelativePositionBias, DecoupledRelativePositionBias, 
                  Attention)

from .eva02 import Block, EVA_02_VisionTransformer



# =================================================================
# ========================= SFP ===================================


class EVA_02_VisionTransformer_SFP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 
                 ft_cfg=[
                     
                     dict(
                         type='neck_ft',
                         out_channels=256,
                         scale_factors=[4.0, 2.0, 1.0, 0.5],
                         norm_cfg=dict(
                            norm_type='LayerNorm2d'   
                        ),
                     )
                     
                 ]
            ):
        super().__init__()
        # self.num_classes = num_classes
        
        for ft_layer_cfg in ft_cfg:
            if ft_layer_cfg['type'] == 'backbone_ft':
                ft_layer_cfg.pop('type')
                ft_backbone_cfg = ft_layer_cfg
            elif ft_layer_cfg['type'] == 'neck_ft':
                ft_layer_cfg.pop('type')
                ft_neck_cfg = ft_layer_cfg
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
        # self.fc_norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1] if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.neck = SimpleFeaturePyramid(
            in_channels=embed_dim,
            **ft_neck_cfg
        )

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)
        self.fix_init_weight()

        # if isinstance(self.head, nn.Linear):
        #     self.head.weight.data.mul_(init_scale)
        #     self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)


    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:    
            x = blk(x, rel_pos_bias)

        x = self.norm(x)
        
        return x
        # if self.fc_norm is not None:
        #     t = x[:, 1:, :]
        #     if return_patch_tokens:
        #         return self.fc_norm(t)
        #     else:
        #         return self.fc_norm(t.mean(1))
        # else:
        #     if return_patch_tokens:
        #         return x[:, 1:]
        #     else:
        #         return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # (N, L+1, C)
        
        x = x[:, 1:, :]
        # (N, L, C)
        
        outs = self.neck(x)
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        # x = self.head(x)
        return outs

    






# ========================= End ===================================
# =================================================================




# =================================================================
# ========================== AdapterFormer ========================

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
                 init_values=None, 
                 norm_cfg=dict(norm_type='LayerNorm'),
                 window_size=None, 
                 attn_head_dim=None, 
                 use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False, 
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=None,
                 
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
                 
                ):
        super().__init__()
        self.norm1 = build_norm_layer(num_features=dim, **norm_cfg)[1]
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln,
            norm_cfg=norm_cfg,
            xattn=xattn,
            rope=rope,
            
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(num_features=dim, **norm_cfg)[1]

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim
            ) # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg
            ) 
            
        self.adapter = Adapter(
            in_channels=dim,
            **ft_cfg
        )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm: self.alpha = math.pow(2.0 * depth, 0.25)
        
        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x))) + self.adapter(x, add_residual=False)# <----------
            elif self.deepnorm:
                residual = x
                x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x + self.adapter(x, add_residual=False)# <----------
                x = self.norm2(x)
            else:
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter(x, add_residual=False) # <----------
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x))) +self.adapter(x, add_residual=False)# <----------
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) +self.adapter(x, add_residual=False)# <----------
        return x
    
    
    def get_self_attn(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            x = self.attn.get_attn_map(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        else:
            # raise an error that gamma_1 not supported
            raise NotImplementedError
        
        return x
    
    def get_attn_weights(self, x: torch.Tensor, rel_pos_bias=None, attn_mask=None) -> torch.Tensor:
        if self.gamma_1 is None:
            x = self.attn.get_attn_map(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        else:
            # raise an error that gamma_1 not supported
            raise NotImplementedError
        
        return x.softmax(dim=-1)
        
    


class EVA_02_VisionTransformer_Adapter(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 
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
            ):
        super().__init__()
        # self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block_Adapter(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                
                ft_cfg=ft_cfg,
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
        # self.fc_norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1] if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)
        self.fix_init_weight()

        # if isinstance(self.head, nn.Linear):
        #     self.head.weight.data.mul_(init_scale)
        #     self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:    
            x = blk(x, rel_pos_bias)

        x = self.norm(x)
        
        return x
        # if self.fc_norm is not None:
        #     t = x[:, 1:, :]
        #     if return_patch_tokens:
        #         return self.fc_norm(t)
        #     else:
        #         return self.fc_norm(t.mean(1))
        # else:
        #     if return_patch_tokens:
        #         return x[:, 1:]
        #     else:
        #         return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


# =============================================================
# ===================================

class EVA_02_VisionTransformer_Adapter_SFP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 
                 ft_cfg=[
                     dict(
                        type='backbone_ft',
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
                     dict(
                         type='neck_ft',
                         out_channels=256,
                         scale_factors=[4.0, 2.0, 1.0, 0.5],
                         norm_cfg=dict(
                            norm_type='LayerNorm2d'   
                        ),
                     )
                     
                 ]
            ):
        super().__init__()
        # self.num_classes = num_classes
        
        for ft_layer_cfg in ft_cfg:
            if ft_layer_cfg['type'] == 'backbone_ft':
                ft_layer_cfg.pop('type')
                ft_backbone_cfg = ft_layer_cfg
            elif ft_layer_cfg['type'] == 'neck_ft':
                ft_layer_cfg.pop('type')
                ft_neck_cfg = ft_layer_cfg
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block_Adapter(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                
                ft_cfg=ft_backbone_cfg,
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
        # self.fc_norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1] if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.neck = SimpleFeaturePyramid(
            in_channels=embed_dim,
            **ft_neck_cfg
        )

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)
        self.fix_init_weight()

        # if isinstance(self.head, nn.Linear):
        #     self.head.weight.data.mul_(init_scale)
        #     self.head.bias.data.mul_(init_scale)
        
        self.num_register_tokens = 0

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:    
            x = blk(x, rel_pos_bias)

        x = self.norm(x)
        
        return x
        # if self.fc_norm is not None:
        #     t = x[:, 1:, :]
        #     if return_patch_tokens:
        #         return self.fc_norm(t)
        #     else:
        #         return self.fc_norm(t.mean(1))
        # else:
        #     if return_patch_tokens:
        #         return x[:, 1:]
        #     else:
        #         return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # (N, L+1, C)
        
        x = x[:, 1:, :]
        # (N, L, C)
        
        outs = self.neck(x)
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        # x = self.head(x)
        return outs

    def get_last_selfattention(self, x, layer=11):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            
        for i, blk in enumerate(self.blocks):
            if i < layer:
                x = blk(x, rel_pos_bias)
            else:
                x = blk.get_self_attn(x, rel_pos_bias)
                return x
                # (blc)
                
                
    def get_attn_weights_layers(self, x:torch.Tensor, n=12):
        # x = self.prepare_tokens_with_masks(x)
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        
        
        output, i, total_block_len = [], 0, len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            attn_weights = blk.get_attn_weights(x)
            
            x = blk(x, rel_pos_bias)
            if isinstance(x, tuple):
                x = x[0]
            
            if i in blocks_to_take:
                output.append(attn_weights)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
        
        

        

# ===================================================================================
# ================================= Lora ============================================

from ..finetunes.lora import LoRA
class Attention_LoRA(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0, 
        proj_drop=0, 
        window_size=None, 
        attn_head_dim=None, 
        use_decoupled_rel_pos_bias=False, 
        deepnorm=False, 
        subln=False, 
        xattn=False, 
        rope=None,
        
        ft_cfg=dict(
            bottleneck = 24, 
                 
            adapter_scalar = '1.0',
        )
        
        ):
        super().__init__()

        # super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            # attn_head_dim must be none in LoRA
            raise ValueError("attn_head_dim must be none in LoRA")
            # head_dim = attn_head_dim
        # all_head_dim = head_dim * self.num_heads
        all_head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.deepnorm = deepnorm
        self.subln = subln
        if self.deepnorm or self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
            
            self.q_lora = LoRA(
                in_channels=dim,
                **ft_cfg
            )
            self.k_lora = LoRA(
                in_channels=dim,
                **ft_cfg
            )
            self.v_lora = LoRA(
                in_channels=dim, 
                **ft_cfg
            )
                
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
            
            self.q_lora = LoRA(
                in_channels=dim,
                **ft_cfg
            )
            self.k_lora = LoRA(
                in_channels=dim,
                **ft_cfg
            )
            self.v_lora = LoRA(
                in_channels=dim, 
                **ft_cfg
            )

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            if use_decoupled_rel_pos_bias:
                self.rel_pos_bias = DecoupledRelativePositionBias(window_size=window_size, num_heads=num_heads)
            else:
                self.window_size = window_size
                self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3    # (2*14-1) * (2*14-1) + 3
                self.relative_position_bias_table = nn.Parameter(
                    torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                # cls to token & token 2 cls & cls to cls

                # get pair-wise relative position index for each token inside the window
                coords_h = torch.arange(window_size[0])
                coords_w = torch.arange(window_size[1])
                coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
                coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
                relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
                relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
                relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
                relative_coords[:, :, 1] += window_size[1] - 1
                relative_coords[:, :, 0] *= 2 * window_size[1] - 1
                relative_position_index = \
                    torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
                relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
                relative_position_index[0, 0:] = self.num_relative_distance - 3
                relative_position_index[0:, 0] = self.num_relative_distance - 2
                relative_position_index[0, 0] = self.num_relative_distance - 1

                self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.xattn = xattn
        self.rope = rope


    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        
        if self.deepnorm or self.subln: 
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias) + self.q_lora(x)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None) + self.k_lora(x)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias) + self.v_lora(x)
            

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 
        else: 
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            
            q_lora = self.q_lora(x)
            k_lora = self.k_lora(x)
            v_lora = self.v_lora(x)
            qkv_lora = torch.cat((q_lora, k_lora, v_lora), dim=-1)
            qkv = qkv + qkv_lora
            
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]   

        if self.rope:
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(q, k, v)
            x = x.reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            if self.qk_float:
                attn = (q.float() @ k.float().transpose(-2, -1))
            else:
                attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if self.rel_pos_bias is not None:
                attn = attn + self.rel_pos_bias().type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)
            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            attn = attn.softmax(dim=-1).type_as(x)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)

        return x



class Block_LoRA(Block):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 init_values=None, 
                 norm_cfg=dict(norm_type='LayerNorm'),
                 window_size=None, 
                 attn_head_dim=None, 
                 use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False, 
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=None,
                 
                 ft_cfg=dict(
                        bottleneck = 24, 
                            
                        adapter_scalar = '1.0',
                    )
                 
                ):
        super(Block, self).__init__()
        
        self.norm1 = build_norm_layer(num_features=dim, **norm_cfg)[1]
        self.attn = Attention_LoRA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln,
            xattn=xattn,
            rope=rope,
            
            ft_cfg=ft_cfg
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(num_features=dim, **norm_cfg)[1]

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim
            ) # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg
            ) 

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm: self.alpha = math.pow(2.0 * depth, 0.25)
        
        self.postnorm = postnorm


class EVA_02_VisionTransformer_LoRA(EVA_02_VisionTransformer):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 
                 ft_cfg=dict(
                    bottleneck = 24, 
                        
                    adapter_scalar = '1.0',
                )
            ):
        super(EVA_02_VisionTransformer, self).__init__()
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block_LoRA(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                
                ft_cfg=ft_cfg
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
       
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
        self.fix_init_weight()
        
        
        

class EVA_02_VisionTransformer_LoRA_SFP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 
                 ft_cfg=[
                     dict(
                        type='backbone_ft',
                        bottleneck=12, 
                        
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
            ):
        super().__init__()
        # self.num_classes = num_classes
        
        for ft_layer_cfg in ft_cfg:
            if ft_layer_cfg['type'] == 'backbone_ft':
                ft_layer_cfg.pop('type')
                ft_backbone_cfg = ft_layer_cfg
            elif ft_layer_cfg['type'] == 'neck_ft':
                ft_layer_cfg.pop('type')
                ft_neck_cfg = ft_layer_cfg
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block_LoRA(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                
                ft_cfg=ft_backbone_cfg,
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
        # self.fc_norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1] if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.neck = SimpleFeaturePyramid(
            in_channels=embed_dim,
            **ft_neck_cfg
        )

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)
        self.fix_init_weight()

        # if isinstance(self.head, nn.Linear):
        #     self.head.weight.data.mul_(init_scale)
        #     self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:    
            x = blk(x, rel_pos_bias)

        x = self.norm(x)
        
        return x
        # if self.fc_norm is not None:
        #     t = x[:, 1:, :]
        #     if return_patch_tokens:
        #         return self.fc_norm(t)
        #     else:
        #         return self.fc_norm(t.mean(1))
        # else:
        #     if return_patch_tokens:
        #         return x[:, 1:]
        #     else:
        #         return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # (N, L+1, C)
        
        x = x[:, 1:, :]
        # (N, L, C)
        
        outs = self.neck(x)
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        # x = self.head(x)
        return outs

    def get_last_selfattention(self, x, layer=11):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            
        for i, blk in enumerate(self.blocks):
            if i < layer:
                x = blk(x, rel_pos_bias)
            else:
                x = blk.get_self_attn(x, rel_pos_bias)
                return x
                # (blc)



class Block_LoRA_Adapter(Block):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 init_values=None, 
                 norm_cfg=dict(norm_type='LayerNorm'),
                 window_size=None, 
                 attn_head_dim=None, 
                 use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False, 
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=None,
                 
                 lora_cfg=dict(
                    bottleneck = 24, 
                        
                    adapter_scalar = '1.0',
                 ),
                 
                 adapter_cfg = dict(
                    type="backbone_adapter_ft",
                    bottleneck=32,
                    adapter_scalar=1.0,
                    learnable_scalar=True,
                    act_cfg=dict(
                        act_type="ReLU",
                        layer_args=dict(
                            inplace=True
                        )
                    ),
                    adapter_layernorm_option=True,
                    dropout_layer=dict(
                        drop_type="Dropout",
                        drop_prob=0.0,
                        inplace=True
                    )
                 ),
                 
                 
                ):
        super(Block, self).__init__()
        
        self.norm1 = build_norm_layer(num_features=dim, **norm_cfg)[1]
        self.attn = Attention_LoRA(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln,
            xattn=xattn,
            rope=rope,
            
            ft_cfg=lora_cfg
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(num_features=dim, **norm_cfg)[1]

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim
            ) # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg
            ) 
            
            
        self.adapter = Adapter(
            in_channels=dim,
            **adapter_cfg
        )
            

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm: self.alpha = math.pow(2.0 * depth, 0.25)
        
        self.postnorm = postnorm
        
    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x))) + self.adapter(x, add_residual=False)# <----------
            elif self.deepnorm:
                residual = x
                x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x + self.adapter(x, add_residual=False)# <----------
                x = self.norm2(x)
            else:
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter(x, add_residual=False) # <----------
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x))) +self.adapter(x, add_residual=False)# <----------
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) +self.adapter(x, add_residual=False)# <----------
        return x


class EVA_02_VisionTransformer_LoRA_Adapter_SFP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 
                 ft_cfg=[
                     dict(
                        type='backbone_lora_ft',
                        bottleneck=12, 
                        
                     ),
                     
                     dict(
                        type="backbone_adapter_ft",
                        bottleneck=32,
                        adapter_scalar=2.0,
                        learnable_scalar=True,
                        act_cfg=dict(
                            act_type="ReLU",
                            layer_args=dict(
                                inplace=True
                            )
                        ),
                        adapter_layernorm_option=True,
                        dropout_layer=dict(
                            drop_type="Dropout",
                            drop_prob=0.0,
                            inplace=True
                        )
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
            ):
        super().__init__()
        # self.num_classes = num_classes
        
        for ft_layer_cfg in ft_cfg:
            if ft_layer_cfg['type'] == 'backbone_adapter_ft':
                ft_layer_cfg.pop('type')
                ft_backbone_adapter_cfg = ft_layer_cfg
            elif ft_layer_cfg['type'] == 'backbone_lora_ft':
                ft_layer_cfg.pop('type')
                ft_backbone_lora_cfg = ft_layer_cfg
            elif ft_layer_cfg['type'] == 'neck_ft':
                ft_layer_cfg.pop('type')
                ft_neck_cfg = ft_layer_cfg
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block_LoRA_Adapter(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                
                lora_cfg=ft_backbone_lora_cfg,
                adapter_cfg=ft_backbone_adapter_cfg,
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
        # self.fc_norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1] if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.neck = SimpleFeaturePyramid(
            in_channels=embed_dim,
            **ft_neck_cfg
        )

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)
        self.fix_init_weight()

        # if isinstance(self.head, nn.Linear):
        #     self.head.weight.data.mul_(init_scale)
        #     self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:    
            x = blk(x, rel_pos_bias)

        x = self.norm(x)
        
        return x
        # if self.fc_norm is not None:
        #     t = x[:, 1:, :]
        #     if return_patch_tokens:
        #         return self.fc_norm(t)
        #     else:
        #         return self.fc_norm(t.mean(1))
        # else:
        #     if return_patch_tokens:
        #         return x[:, 1:]
        #     else:
        #         return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # (N, L+1, C)
        
        x = x[:, 1:, :]
        # (N, L, C)
        
        outs = self.neck(x)
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        # x = self.head(x)
        return outs

    def get_last_selfattention(self, x, layer=11):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
            
        for i, blk in enumerate(self.blocks):
            if i < layer:
                x = blk(x, rel_pos_bias)
            else:
                x = blk.get_self_attn(x, rel_pos_bias)
                return x
                # (blc)
















# ================================== End ============================================
# ===================================================================================




# =================== deprecated =======================================================================================================
# ======================================================================
# ============== InvertedSwiGLU_Adapter + ViTDet =======================

from ..finetunes import InvertedSwiGLU_Adapter
class Block_InvertedSwiGLUAdapter(nn.Module):

    def __init__(self, 
                 dim, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 init_values=None, 
                 norm_cfg=dict(norm_type='LayerNorm'),
                 window_size=None, 
                 attn_head_dim=None, 
                 use_decoupled_rel_pos_bias=False,
                 depth=None,
                 postnorm=False, 
                 deepnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=None,
                 
                 ft_cfg=dict(
                     bottleneck=48, 
                     subln=True,
                     adapter_scalar='0.1', 
                     
                     act_cfg = dict(act_type='SiLU'), 
     
                     adapter_layernorm_option='none',
                               
                     ),
                 
                ):
        super().__init__()
        self.norm1 = build_norm_layer(num_features=dim, **norm_cfg)[1]
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, 
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln,
            xattn=xattn,
            rope=rope,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(num_features=dim, **norm_cfg)[1]

        mlp_hidden_dim = int(dim * mlp_ratio)
        
        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim
            ) # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_cfg=norm_cfg
            ) 
            
        self.adapter = InvertedSwiGLU_Adapter(
            in_channels=dim,
            **ft_cfg
        )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm: self.alpha = math.pow(2.0 * depth, 0.25)
        
        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x))) + self.adapter(x, add_residual=False)# <----------
            elif self.deepnorm:
                residual = x
                x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x + self.adapter(x, add_residual=False)# <----------
                x = self.norm2(x)
            else:
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x))) + self.adapter(x, add_residual=False) # <----------
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x))) +self.adapter(x, add_residual=False)# <----------
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) +self.adapter(x, add_residual=False)# <----------
        return x


class EVA_02_VisionTransformer_InvertedSwiGLUAdapter_SFP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                #  num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_cfg=dict(norm_type='LaryerNorm'), 
                 init_values=None, 
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False, 
                 use_shared_rel_pos_bias=False, 
                 use_decoupled_rel_pos_bias=False,
                 postnorm=False,
                 subln=False,
                 xattn=False,
                 swiglu=False,
                 naiveswiglu=False,
                 rope=False,
                 pt_hw_seq_len=16,
                 intp_freq=False,
                 
                 ft_cfg=[
                     dict(
                        type='backbone_ft',
                        bottleneck=48,
                        subln = True, 
                        adapter_scalar='0.1', 
                        
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
            ):
        super().__init__()
        # self.num_classes = num_classes
        
        for ft_layer_cfg in ft_cfg:
            if ft_layer_cfg['type'] == 'backbone_ft':
                ft_layer_cfg.pop('type')
                ft_backbone_cfg = ft_layer_cfg
            elif ft_layer_cfg['type'] == 'neck_ft':
                ft_layer_cfg.pop('type')
                ft_neck_cfg = ft_layer_cfg
        
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        self.use_decoupled_rel_pos_bias = use_decoupled_rel_pos_bias

        if use_decoupled_rel_pos_bias or use_rel_pos_bias:
            window_size = self.patch_embed.patch_shape
        else:
            window_size = None

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: self.rope = None

        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block_InvertedSwiGLUAdapter(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_cfg=norm_cfg,
                init_values=init_values, window_size=window_size, use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias,
                postnorm=postnorm,
                subln=subln,
                xattn=xattn,
                swiglu=swiglu,
                naiveswiglu=naiveswiglu,
                rope=self.rope,
                
                ft_cfg=ft_backbone_cfg,
            )
            for i in range(depth)])
        self.norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1]
        # self.fc_norm = build_norm_layer(num_features=embed_dim, **norm_cfg)[1] if use_mean_pooling else None
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.neck = SimpleFeaturePyramid(
            in_channels=embed_dim,
            **ft_neck_cfg
        )

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        # if isinstance(self.head, nn.Linear):
        #     trunc_normal_(self.head.weight, std=.02)
        # self.apply(self._init_weights)
        self.fix_init_weight()

        # if isinstance(self.head, nn.Linear):
        #     self.head.weight.data.mul_(init_scale)
        #     self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1:
        #     x = x.detach()
        
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:    
            x = blk(x, rel_pos_bias)

        x = self.norm(x)
        
        return x
        # if self.fc_norm is not None:
        #     t = x[:, 1:, :]
        #     if return_patch_tokens:
        #         return self.fc_norm(t)
        #     else:
        #         return self.fc_norm(t.mean(1))
        # else:
        #     if return_patch_tokens:
        #         return x[:, 1:]
        #     else:
        #         return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # (N, L+1, C)
        
        x = x[:, 1:, :]
        # (N, L, C)
        
        outs = self.neck(x)
        # List: [Tensor, ...]
        #   Tensor: (N, c, h, w). c = 256
        
        # x = self.head(x)
        return outs














