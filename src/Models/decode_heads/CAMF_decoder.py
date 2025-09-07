from typing import Type, Tuple, Union, Optional, Dict, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

from mmengine.model import kaiming_init, constant_init

from  .decode_head import BaseDecodeHead


from ..utils.attention import BaseAttention 
from ..utils import (ConvModule, resize, 
                     build_activation_layer, build_norm_layer, 
                     nlc_to_nchw, nchw_to_nlc, nlc2nchw2nlc)

from ..utils.ffn import MLP_FFN




class Spatial_MLP(MLP_FFN):
    def __init__(self, 
                 in_len: int, 
                 mlp_ratio: int = None, 
                 out_len: int = None, 
                 act_cfg: Dict = dict(act_type='GELU'), 
                 drop: float = 0., 
                 
                 norm_cfg: Dict = None, 
                 subln: bool = False):
        super().__init__(in_len, mlp_ratio * in_len, out_len, act_cfg, drop, norm_cfg, subln)

    def forward(self, x):
        '''
        Shape:
            x: (N, L, C)
        '''
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.act(x)
        
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        return x
    



    
class MLP_Block(nn.Module):
    '''
    Args:
    
    
    Shape:
    
    
    '''
    def __init__(self, 
                in_len: int,
                dim: int,
                mlp_ratio: int = 2,
                
                
                norm_cfg: Dict = dict(
                    norm_type='LayerNorm',
                )
                
                
                ) -> None:
        super().__init__()
        
        
        self.norm1 = build_norm_layer(dim, **norm_cfg)[1]
        self.spatial_mlp = Spatial_MLP(in_len=in_len, mlp_ratio=mlp_ratio, out_len=in_len, 
                                       act_cfg=dict(act_type='GELU')
                                       )
        
        self.norm2 = build_norm_layer(dim, **norm_cfg)[1]
        self.channel_mlp = MLP_FFN(in_features=dim, hidden_features=dim * mlp_ratio, out_features=dim, 
                                   act_cfg=dict(act_type='GELU')
                                   )
        
        
    def forward(self, x):
        x = x + self.spatial_mlp(self.norm1(x))
        x = x + self.channel_mlp(self.norm2(x))
        return x
    

        
class MFA(nn.Module):
    '''
    Args:
    
    
    Shape:
    
    
    '''


    def __init__(self, 
                 img_in_dims: int,
                 prompt_in_len: int,
                 vocab_size: int = 50272, #49408,  
                 channels: int = 128,
                 mlp_ratio: int = 2,
                 
                 norm_cfg: Dict = dict(
                    norm_type='LayerNorm',
                )
                 
                 ) -> None:
        super().__init__()
        
        self.text_embedder = nn.Embedding(vocab_size, channels)
        
        self.query_feat = nn.Embedding(prompt_in_len, channels)
        # self.mlp_proj_img = nn.Sequential(
        #     nn.Linear(img_in_dims, channels),
        #     nn.GELU(),
        #     nn.Linear(channels, channels)
        # )
        # self.mlp_proj_prompt = nn.Sequential(
        #     nn.Linear(prompt_in_dims, channels),
        #     nn.GELU(),
        #     nn.Linear(channels, channels)
        # )
        
        self.mlp_proj_img = MLP_FFN(in_features=img_in_dims, hidden_features=channels, out_features=channels, 
                                    act_cfg=dict(act_type='GELU')
        )
        
        # self.mlp_proj_img = nn.Linear(img_in_dims, channels)
        
        self.mlp_prompt1 = MLP_Block(in_len=prompt_in_len, dim=channels, mlp_ratio=mlp_ratio, norm_cfg=norm_cfg)
        self.down_fc = nn.Linear(prompt_in_len, prompt_in_len // 2)
        self.mlp_prompt2 = MLP_Block(in_len=prompt_in_len // 2, dim=channels, mlp_ratio=mlp_ratio, norm_cfg=norm_cfg)
        # self.mlp_proj_img = nn.Linear(img_in_dims, channels)
        # self.mlp_proj_prompt = nn.Linear(prompt_in_dims, channels)
        
        
    @staticmethod    
    def _cal_fg_align(img_embedding, prompt_embedding, L_p, H, W):
        
        threshold = 1/L_p
        similarity = torch.einsum('bpc,btc->bpt', img_embedding, prompt_embedding)
        
        # min-max normalization
        similarity = (similarity - similarity.min(dim=-1, keepdim=True)[0]) / (similarity.max(dim=-1, keepdim=True)[0] - similarity.min(dim=-1, keepdim=True)[0])
        
        # thresholding 
        similarity = torch.where(similarity > threshold, similarity, torch.zeros_like(similarity))
        
        # alignment-weighting
        v_align_weights = F.softmax(similarity, dim=-1)
        # v_align_weights = similarity / sum(similarity, dim = -1, keep_dims=True)
        
        v_grouped_l_embedding = torch.einsum('bpt,btc->bpc', v_align_weights, prompt_embedding)
        out = rearrange(v_grouped_l_embedding, 'b (h w) c -> b c h w', h=H, w=W)
        return out
    
        
    def forward(self, img_embedding_list, prompt_idx):
        '''
        Shape:
            img_embedding: list of  (N, C, H, W)
            prompt_embedding: (N, L_t, C)
            
            out: list of (N, C, H, W)
        
        '''
        
        prompt_embedding = self.text_embedder(prompt_idx)
        
        
        # (N, L, C)
        N = prompt_embedding.shape[0]
        query_embedding = self.query_feat.weight.unsqueeze(0).repeat(N, 1, 1)
        # (N, L, C)
        
        
        prompt_embedding = self.mlp_prompt1(prompt_embedding)
        prompt_embedding = prompt_embedding.permute(0, 2, 1)
        prompt_embedding = self.down_fc(prompt_embedding)
        prompt_embedding = prompt_embedding.permute(0, 2, 1)
        prompt_embedding = self.mlp_prompt2(prompt_embedding)
        
        query_embedding = self.mlp_prompt1(query_embedding)
        query_embedding = query_embedding.permute(0, 2, 1)
        query_embedding = self.down_fc(query_embedding)
        query_embedding = query_embedding.permute(0, 2, 1)
        query_embedding = self.mlp_prompt2(query_embedding)
        
        out_dict = dict()
        
        out_img_txt_list = []
        out_img_q_list = []
        for img_embedding in img_embedding_list:
            _, _, H, W = img_embedding.shape
            img_embedding = rearrange(img_embedding, 'b c h w -> b (h w) c')
            _, L_p, _ = img_embedding.shape
            img_embedding = self.mlp_proj_img(img_embedding)
            
            # threshold = 1/L_p
            # similarity = torch.einsum('bpc,btc->bpt', img_embedding, prompt_embedding)
            
            # # min-max normalization
            # similarity = (similarity - similarity.min(dim=-1, keepdim=True)[0]) / (similarity.max(dim=-1, keepdim=True)[0] - similarity.min(dim=-1, keepdim=True)[0])
            
            # # thresholding 
            # similarity = torch.where(similarity > threshold, similarity, torch.zeros_like(similarity))
            
            # # alignment-weighting
            # v_align_weights = F.softmax(similarity, dim=-1)
            
            # v_grouped_l_embedding = torch.einsum('bpt,btc->bpc', v_align_weights, prompt_embedding)
            # out = rearrange(v_grouped_l_embedding, 'b (h w) c -> b c h w', h=H, w=W)
            
            out_img_txt = self._cal_fg_align(img_embedding, prompt_embedding, L_p, H, W)
            out_img_q = self._cal_fg_align(img_embedding, query_embedding, L_p, H, W)
            
            out_img_txt_list.append(out_img_txt)
            out_img_q_list.append(out_img_q)
            
        out_dict['img_txt_embedding'] = out_img_txt_list
        out_dict['img_q_embedding'] = out_img_q_list
        
        return out_dict
        
        
    def infer_forward(self, img_embedding_list, prompt_idx='query'):
        '''
        Shape:
            img_embedding: list of  (N, C, H, W)
            prompt_embedding: (N, L_t, C)
            
            out: list of (N, C, H, W)
        
        '''
        N = img_embedding_list[0].shape[0]
        query_embedding = self.query_feat.weight.unsqueeze(0).repeat(N, 1, 1)
        # (N, L, C)
        # query_embedding = torch.zeros_like(query_embedding)
        # if prompt_idx is not None: # use mm-llm prompt
        #     query_embedding = self.text_embedder(prompt_idx)
        if prompt_idx != 'query': # use mm-llm prompt
            query_embedding = self.text_embedder(prompt_idx)
        
        # query_embedding = torch.rand_like(query_embedding)
        
        query_embedding = self.mlp_prompt1(query_embedding)
        query_embedding = query_embedding.permute(0, 2, 1)
        query_embedding = self.down_fc(query_embedding)
        query_embedding = query_embedding.permute(0, 2, 1)
        query_embedding = self.mlp_prompt2(query_embedding)
        
        out_dict = dict()
        
        out_img_q_list = []
        for img_embedding in img_embedding_list:
            _, _, H, W = img_embedding.shape
            img_embedding = rearrange(img_embedding, 'b c h w -> b (h w) c')
            _, L_p, _ = img_embedding.shape
            img_embedding = self.mlp_proj_img(img_embedding)
            
            out_img_q = self._cal_fg_align(img_embedding, query_embedding, L_p, H, W)
            
            out_img_q_list.append(out_img_q)
            
        out_dict['img_q_embedding'] = out_img_q_list
        
        return out_dict
        


        
        
        
        
class camf_decoder(BaseDecodeHead):
    '''
    Args:
    
    
    Shape:
    
    
    
    '''   
        
    def __init__(self, 
                 in_channels,
                 channels: int = 256,
                 num_classes: int = 2,
                 out_channels: int = 1,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg: dict = dict(norm_type='LayerNorm'),
                 act_cfg: dict = dict(act_type='GELU'),
                 in_index=-1,
                 align_corners=False,
                 interpolate_mode='bilinear', 
                 
                 ft_cfg = dict(
                    prompt_in_len = 32,
                    tm_channels = 256,
                 ),
                 
                 ) -> None:
        super().__init__(input_transform='multiple_select', in_channels=in_channels, channels=channels, num_classes=num_classes, out_channels=out_channels, dropout_ratio=dropout_ratio, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, in_index=in_index, align_corners=align_corners)

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
            
            
            
        # prompt_in_channels = ft_cfg.get('prompt_in_channels', None)
        prompt_in_len = ft_cfg.get('prompt_in_len', None)
        tm_channels = ft_cfg.get('tm_channels', None)
        self.token_match = MFA(img_in_dims=self.in_channels[0],prompt_in_len=prompt_in_len, channels=tm_channels)

        
        self.i_g_convs = nn.ModuleList()
        for i in range(num_inputs):
            self.i_g_convs.append(
                ConvModule(
                    in_channels=tm_channels,
                    out_channels=self.channels,
                    kernel_size=1, 
                    stride=1, 
                    conv_cfg=dict(conv_type = 'Conv2d'), 
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                )
            )
            
        # self.scalers = nn.Parameter(torch.tensor([0.1, 0.1, 1.0, 0.1]))
            

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs*2,
            out_channels=self.channels,
            kernel_size=1,
            conv_cfg=dict(conv_type = 'Conv2d'),
            norm_cfg=self.norm_cfg)
        
    #     self.init_weights()
        
    # def init_weights(self):
    #     for conv in self.i_g_convs:
    #         constant_init(conv, 0, bias=0)
        
        
    def forward(self, inputs, prompt_idx) -> Tensor:
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
            
        
        TM_outs_dict = self.token_match(img_embedding_list=inputs, 
                                   prompt_idx=prompt_idx)
        
        TM_outs = TM_outs_dict['img_txt_embedding']
        
        t_outs = []
        for idx in range(len(TM_outs)):
            x = TM_outs[idx]
            i_g_conv = self.i_g_convs[idx]
            t_outs.append(
                resize(
                    input=i_g_conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            )
        outs.extend(t_outs)
            
        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out) # shape(n, num_classes, h, w)

        return dict(logits_mask=out, 
                    logits_img_txt = TM_outs_dict['img_txt_embedding'],
                    logits_img_q = TM_outs_dict['img_q_embedding'],
                    )
        
        
    def infer_forward(self, inputs, prompt_idx="query") -> Tensor:
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
            
        
        TM_outs_dict = self.token_match.infer_forward(img_embedding_list=inputs, prompt_idx=prompt_idx)
        
        TM_outs = TM_outs_dict['img_q_embedding']
        
        t_outs = []
        for idx in range(len(TM_outs)):
            x = TM_outs[idx]
            i_g_conv = self.i_g_convs[idx]
            t_outs.append(
                resize(
                    input=i_g_conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners
                )
            )
        outs.extend(t_outs)
            
        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out) # shape(n, num_classes, h, w)

        return dict(logits_mask=out, 
                    )
        
        
        
        
        
   
   
    # def get_matched_token(self, inputs, prompt_idx=None):
    #     inputs = self._transform_inputs(inputs)
    #     outs = []
    #     for idx in range(len(inputs)):
    #         x = inputs[idx]
    #         conv = self.convs[idx]
    #         outs.append(
    #             resize(
    #                 input=conv(x),
    #                 size=inputs[0].shape[2:],
    #                 mode=self.interpolate_mode,
    #                 align_corners=self.align_corners))
            
        
    #     TM_outs_dict = self.token_match.infer_forward(img_embedding_list=inputs, prompt_idx=prompt_idx)
    #     TM_outs = TM_outs_dict['img_q_embedding']

    #     t_outs = []
    #     for idx in range(len(TM_outs)):
    #         x = TM_outs[idx]
    #         x = x.mean(dim=1, keepdim=True)
    #         t_outs.append(
    #             resize(
    #                 input=x,
    #                 size=inputs[0].shape[2:],
    #                 mode='nearest',
    #                 # align_corners=self.align_corners
                    
    #             )
    #         )
            
    #     matched_token = t_outs[2]
    #     # matched_token = matched_token.mean(dim=0)
    #     # (1, 1, H, W)
        
    #     return dict(
    #         tm_map=matched_token
    #     )
