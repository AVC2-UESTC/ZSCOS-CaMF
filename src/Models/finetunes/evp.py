import math
import torch
import torch.nn as nn

from mmengine.model import kaiming_init, constant_init
from einops import rearrange

from ..utils import nchw_to_nlc, nlc_to_nchw
from ..utils import build_activation_layer
from ..utils import build_norm_layer
from ..utils import build_dropout





class FourierMLP(nn.Module):
    '''
    
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    '''
    def __init__(self, in_channels) -> None:
        super().__init__()
        
        self.mask_r = nn.Linear(in_channels, in_channels, bias=False)
        



class EVPv2_Adapter(nn.Module):
    '''
    
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    '''
    def __init__(self, 
                 in_channels, 
                 
                 # finetune cfg
                 bottleneck = None, 
                 
                 adapter_scalar = '1.0',
                 
                 act_cfg = dict(act_type='GELU'), 
                 
                 dropout_layer = dict(drop_type='Dropout',
                                        drop_prob=0.0,
                                        inplace=False),
                 
                 
                 ):

        super().__init__()

        self.n_embed = in_channels  
        self.down_size = bottleneck
        
        self.fmlp = FourierMLP(self.n_embed, self.n_embed)
        
        self.down = nn.Linear(self.n_embed, self.down_size)
        
        self.pe = nn.Sequential(
            nn.Linear(self.down_size, self.down_size), 
            build_activation_layer(**act_cfg)
        )
        
        self.freq = nn.Sequential(
            nn.Linear(self.down_size, self.down_size), 
            build_activation_layer(**act_cfg)
        )
        
        self.dropout = build_dropout(**dropout_layer)

        
        self.up = nn.Linear(self.down_size, self.n_embed)
        
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
        

        # initalize weights
        self.init_weights()
        
        
    def init_weights(self):
        kaiming_init(self.down, bias=0, distribution='uniform')
        kaiming_init(self.pe, bias=0, distribution='uniform')
        kaiming_init(self.freq, bias=0, distribution='uniform')
        constant_init(self.up_proj, 0, bias=0)






class Beta_Adapter(nn.Module):
    def __init__(self, 
                 in_channels, 
                 
                 # finetune cfg
                 bottleneck = None, 
                 
                 adapter_scalar = '1.0',
                 
                 act_cfg = dict(act_type='GELU'), 
                 
                 dropout_layer = dict(drop_type='Dropout',
                                        drop_prob=0.0,
                                        inplace=False),) -> None:
        super().__init__()
        
        self.n_embed = in_channels  
        self.down_size = bottleneck
        
        self.down = nn.Linear(self.n_embed, self.down_size)
        
        self.pe = nn.Sequential(
            nn.Linear(self.down_size, self.down_size), 
            build_activation_layer(**act_cfg)
        )


        self.dropout = build_dropout(**dropout_layer)

        
        self.up = nn.Linear(self.down_size, self.n_embed)
        
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)


         # initalize weights
        self.init_weights()
        
        
    def init_weights(self):
        kaiming_init(self.down, bias=0, distribution='uniform')
        kaiming_init(self.pe, bias=0, distribution='uniform')
        constant_init(self.up, 0, bias=0)
        
        
    def forward(self, x):
        x = self.down(x)
        
        sub_residual = x 
        
        x = self.pe(x)
        x = self.dropout(x)
        
        x = sub_residual + x 
        
        x = self.up(x) * self.scale
        
        return x
















