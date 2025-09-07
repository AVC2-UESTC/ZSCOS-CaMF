
import math
import torch
import torch.nn as nn

from mmengine.model import kaiming_init, constant_init
from einops import rearrange

from ..utils import nchw_to_nlc, nlc_to_nchw
from ..utils import build_activation_layer
from ..utils import build_norm_layer
from ..utils import build_dropout






class LoRA(nn.Module):
    '''
    Args:
        in_channels: input channels $d$ e.g. C
        bottleneck (int): $\hat{d}$
        
        
        adapter_scalar (str): s. Could be learnable scalar i.e. 'learnable_scalar'
            Default: '1.0'
            
        act_cfg: activation function config. See src.Models.utils.
            Default: dict(
                act_type = 'ReLU', 
                inplace = False
            )
            
        adapter_layernorm_option: The position of norm layer.
            'in' for the norm before the down layer.
            'out' for the norm after the up layer.
            'none': no norm layer
            Default: 'in'
            
        norm_cfg: norm config.
            Default: dict(norm_type='LayerNorm', 
                                 requires_grad=True,
                                 eps=1e-5,
                                 affine=True)
            
        dropout_layer: dropout config
            Default: dict(drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
        
        
                                        
                                        
        
    
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    
    '''
    def __init__(self, 
                 in_channels, 
                 
                 # finetune cfg
                 bottleneck = None, 
                 
                 adapter_scalar = '1.0',
                 
                 adapter_layernorm_option="none",
                 norm_cfg = None,
                 
                 
                 ):
        super().__init__()
        self.n_embed = in_channels  
        self.down_size = bottleneck
        
        # the position of norm
        self.adapter_layernorm_option = adapter_layernorm_option
    
        
        
        if adapter_layernorm_option == 'in' or adapter_layernorm_option == 'out':
            self.adapter_layer_norm = build_norm_layer(num_features=in_channels, **norm_cfg)[1]
        else: 
            self.adapter_layer_norm = None
        
        
        
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
            
        
        
            
        
        self.down_proj = nn.Linear(self.n_embed, self.down_size)
        self.up_proj = nn.Linear(self.down_size, self.n_embed)
        
        # initalize weights
        self.init_weights()
        
        
    def init_weights(self):
        kaiming_init(self.down_proj, bias=0, distribution='uniform')
        constant_init(self.up_proj, 0, bias=0)


    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm(x)
        
        down = self.down_proj(x)
        up = self.up_proj(down)
        
        up = up * self.scale
        
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm(up)
        
        if add_residual:
            up = up + residual
        
        return up







































