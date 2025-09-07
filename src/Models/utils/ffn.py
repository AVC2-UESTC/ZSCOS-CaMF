import warnings
from typing import Any, Optional, Tuple, Type, Dict, List

import numpy as np
import torch
from torch import nn

from mmengine.model import (constant_init, normal_init,
                                        trunc_normal_init)

from einops import rearrange

import torch.nn.functional as F

try:
    import xformers.ops as xops
except ImportError:
    warnings.warn("xformers not installed")


from . import build_activation_layer, build_norm_layer




class MLP_FFN(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_cfg=dict(act_type='GELU'), 
                 drop=0., 
                 norm_cfg=dict(norm_type='LayerNorm'), 
                 subln=False
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(**act_cfg)

        self.ffn_ln = build_norm_layer(num_features=hidden_features, **norm_cfg)[1] if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class ConvMLP_FFN(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_cfg=dict(act_type='GELU'), 
                 drop=0., 
                 norm_cfg=dict(norm_type='LayerNorm2d'), 
                 subln=False
                 ) -> None:
        super().__init__()
        
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = build_activation_layer(**act_cfg)

        self.ffn_ln = build_norm_layer(num_features=hidden_features, **norm_cfg)[1] if subln else nn.Identity()

        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)


    def forward(self, x):
        '''
        Args:
            x: (N, C, h, w)
        
        '''
        x = self.fc1(x)
        x = self.act(x)
        
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
class GLU_FFN(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_cfg=dict(act_type='SiLU'), # GELU
                 drop=0., 
                 norm_cfg=dict(norm_type='LayerNorm'), 
                 subln=False, 
                 xglu=False,
                 
            ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = build_activation_layer(**act_cfg)
        self.ffn_ln = build_norm_layer(num_features=hidden_features, **norm_cfg)[1] if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)
        
        # if xglu:
        #     act = act_cfg['act_type']
        #     if act == 'SiLU':
        #         self.ffn = xops.SwiGLU(
        #             in_features=in_features, 
        #             hidden_features=hidden_features, 
        #             out_features=out_features,
        #         ) # hidden_features: 2/3
        #     else:
        #         raise ValueError(f'Does not support {act} when using xglu.')
        

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x
    
    
    
    
    
# ====================================== Under Construction ======================================



class Naive_Expert(nn.Module):
    def __init__(self, 
                 
                 in_features: int, 
                 out_features: int,
                 
                 ) -> None:
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        
        
    def forward(self, x):
        return self.linear(x)
        


class Naive_MoE_FFN(nn.Module):
    def __init__(self, 
                 
                 num_experts: int, 
                 in_features: int, 
                 out_features: int,
                 
                 
                 
                 ) -> None:
        super().__init__()
        
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([Naive_Expert(in_features, out_features) for _ in range(num_experts)])
        
        self.gate = nn.Sequential(
            nn.Linear(in_features, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        gate_score = self.gate(x) # (N, num_experts, C)
        
        expert_outputs = [expert(x) for expert in self.experts]
        # List[torch.Tensor] 
        # Each Tensor: (N, L, C)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        # (N, num_experts, L, C)
        
        # output = torch.bmm(gate_score.unsqueeze(1), expert_outputs).squeeze(1)
        output = torch.einsum('bnc,bnlc->blc', (gate_score, expert_outputs))
        # (N, L, C)
        
        return output
    


  
class BasicExpert(nn.Module):
    def __init__(self, 
        in_features: int, 
        hidden_features: int, 
        out_features: int, 
        act_cfg: dict

    ) -> None:
        super().__init__()
        
        self.MLP = MLP_FFN(in_features=in_features, 
                 hidden_features=hidden_features, 
                 out_features=out_features,
                 act_cfg=act_cfg,
                 norm_cfg=None
        )
        
    def forward(self, x):
        return self.MLP(x)
    
    
class BasicRouter(nn.Module):
    def __init__(self,
        in_features: int, 
        num_experts: int, 
        topk:int,

    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        
        self.gate =  nn.Linear(in_features=in_features, out_features=num_experts)



    def forward(self, x:torch.Tensor):
        # x:(N, L, C)
        router_logits = self.gate(x)
        router_prob = F.softmax(router_logits, dim=-1) # (N, L, num_experts)
        expert_weights, selected_expert_indices = torch.topk(router_prob, self.topk, dim=-1)
        # expert_weights: (N, L, topk)
        # expert_indices: (N, L, topk)
        
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(x.dtype)

        expert_mask = F.one_hot(selected_expert_indices, num_classes=self.num_experts)
        # (N, L, topk, num_experts)
        # expert_mask = expert_mask.permute(0, 3, 1, 2)
        # # (N, num_experts, L, topk)
        return router_logits, expert_weights, selected_expert_indices, expert_mask
        
        

    
class MoE_FFN(nn.Module):
    def __init__(self, 
        in_features: int, 
        hidden_features: int, 
        out_features: int, 
        num_experts: int, 
        act_cfg: dict, 
        
        topk:int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([BasicExpert(in_features=in_features, 
                                                  hidden_features=hidden_features,
                                                  out_features=out_features, 
                                                  act_cfg=act_cfg) 
                                      for _ in range(num_experts)])
        
        self.router = BasicRouter(in_features=in_features,
                                 num_experts=num_experts,
                                 topk=topk
        )
        
        
        
        
        
    def forward(self, x:torch.Tensor):
        B, L, C = x.size()
        
        router_logits, expert_weights, selected_expert_indices, expert_mask = self.router(x)
        # router_logits: (N, L, num_experts)
        # expert_weights: (N, L, topk)
        # expert_indices: (N, L, topk)
        # expert_mask: (N, L, topk, num_experts)
        
        x_bl_fl = x.view(B*L, C) # (B*L, C)
        expert_mask = rearrange(expert_mask, 'b l k n_e -> n_e k (b l)') # (num_experts, topk, B*L)
        expert_weights = rearrange(expert_weights, 'b l k -> (b l) k') # (B*L, topk)
        
        
        final_x = torch.zeros_like(x_bl_fl)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx]: (topk, B*L)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx: (topk, )
            # top_x: (l_i, )
            
            selected_x_bl_fl = x_bl_fl.unsqueeze(0)[:, top_x, :].reshape(-1, C) # (l_i, C)
            selected_x_bl_fl = expert_layer(selected_x_bl_fl) * expert_weights[top_x, idx].unsqueeze(-1)
            final_x.index_add_(0, top_x, selected_x_bl_fl)

        final_x = rearrange(final_x, '(b l) c -> b l c', b=B, l=L) # (B, L, C)
        

        return final_x, router_logits
            
            
            
class SharedExpertSparseMoE(MoE_FFN):
    def __init__(self, 
            in_features: int, 
            hidden_features: int, 
            out_features: int, 
            num_shared_experts: int, 
            num_experts: int, 
            act_cfg: dict, 
            topk: int) -> None:
        super().__init__(in_features, hidden_features, out_features, num_experts, act_cfg, topk)
        
        self.shared_experts = nn.ModuleList([BasicExpert(in_features=in_features, 
                                                  hidden_features=hidden_features,
                                                  out_features=out_features, 
                                                  act_cfg=act_cfg) 
                                      for _ in range(num_shared_experts)])
            
            
    def forward(self, x:torch.Tensor):
        B, L, C = x.size()
        
        # Shared =================================
        
        shared_experts_out = [
            expert(x) for expert in self.shared_experts
        ] 
        
        shared_experts_out = torch.stack(
            shared_experts_out, dim=0
        ).sum(dim=0)
        
        # Sparse =================================
        
        router_logits, expert_weights, selected_expert_indices, expert_mask = self.router(x)
        # router_logits: (N, L, num_experts)
        # expert_weights: (N, L, topk)
        # expert_indices: (N, L, topk)
        # expert_mask: (N, L, topk, num_experts)
        
        x_bl_fl = x.view(B*L, C) # (B*L, C)
        expert_mask = rearrange(expert_mask, 'b l k n_e -> n_e k (b l)') # (num_experts, topk, B*L)
        expert_weights = rearrange(expert_weights, 'b l k -> (b l) k') # (B*L, topk)
        
        
        final_x = torch.zeros_like(x_bl_fl)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # expert_mask[expert_idx]: (topk, B*L)
            idx, top_x = torch.where(expert_mask[expert_idx])
            # idx: (topk, )
            # top_x: (l_i, )
            
            selected_x_bl_fl = x_bl_fl.unsqueeze(0)[:, top_x, :].reshape(-1, C) # (l_i, C)
            selected_x_bl_fl = expert_layer(selected_x_bl_fl) * expert_weights[top_x, idx].unsqueeze(-1)
            final_x.index_add_(0, top_x, selected_x_bl_fl)

        final_x = rearrange(final_x, '(b l) c -> b l c', b=B, l=L) # (B, L, C)
        
        ffn_out = shared_experts_out + final_x

        return ffn_out, router_logits