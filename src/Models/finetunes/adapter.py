import math
import torch
import torch.nn as nn

from mmengine.model import kaiming_init, constant_init
from einops import rearrange

from ..utils import nchw_to_nlc, nlc_to_nchw
from ..utils import build_activation_layer
from ..utils import build_norm_layer
from ..utils import build_dropout



# adapter from AdaptFormer Adapting Vision Transformers for Scalable Visual Recognition
# http://arxiv.org/abs/2205.13535
# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------
class Adapter(nn.Module):
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
                 
                 adapter_scalar: float = 0.1,
                 learnable_scalar=True,
                 
                 act_cfg = dict(act_type='ReLU', 
                                layer_args=dict(inplace=False)), 
                 
                 adapter_layernorm_option="in",
                 norm_cfg = dict(norm_type='LayerNorm', 
                     requires_grad=True,
                     ),
                 
                 dropout_layer = dict(drop_type='Dropout',
                                        drop_prob=0.0,
                                        inplace=False),
                 
                 
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
        
        
        
        # if adapter_scalar == 'learnable_scalar':
        #     self.scale = nn.Parameter(torch.ones(1))
        # else:
        #     self.scale = float(adapter_scalar)
            
        if learnable_scalar == True:
            self.scale = nn.Parameter(torch.tensor(adapter_scalar))
        else:
            self.scale = adapter_scalar
            
            
        
        self.down_proj = nn.Linear(self.n_embed, self.down_size)
        self.non_linear_func = build_activation_layer(**act_cfg)
        self.up_proj = nn.Linear(self.down_size, self.n_embed)
        self.dropout = build_dropout(**dropout_layer)
        
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
        down = self.non_linear_func(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        
        up = up * self.scale
        
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm(up)
        
        if add_residual:
            up = up + residual
        
        return up
        
        
class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        '''
        Shape:
            In:
                x: (N, c, h, w)
            Out: 
                (N, c, h, w)
        
        '''
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

class Mona(nn.Module):
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
                 
                 act_cfg = dict(act_type='GELU'), 
                 
                 adapter_layernorm_option="in",
                 norm_cfg = dict(norm_type='LayerNorm'),
                 
                 dropout_layer = dict(drop_type='Dropout',
                                        drop_prob=0.1,
                                        inplace=False),
                 
                 
                 ):
        super().__init__()
        self.n_embed = in_channels  
        self.down_size = bottleneck
        
        # the position of norm
        self.adapter_layernorm_option = adapter_layernorm_option
    
        
        
        assert adapter_layernorm_option == 'in', f'adpter_layernorm_option should be "in", but got {adapter_layernorm_option}'
        self.adapter_layer_norm = build_norm_layer(num_features=in_channels, **norm_cfg)[1]
        
        
        
        
        if adapter_scalar == 'learnable_scalar':
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)
            
            
        
        self.down_proj = nn.Linear(self.n_embed, self.down_size)
        self.non_linear_func = build_activation_layer(**act_cfg)
        
        self.up_proj = nn.Linear(self.down_size, self.n_embed)
        self.dropout = build_dropout(**dropout_layer)
        
        self.adapter_conv = MonaOp(in_features=self.down_size)
        
        self.gamma = nn.Parameter(torch.ones(self.n_embed) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(self.n_embed))
        
        # initalize weights
        self.init_weights()
        
        
    def init_weights(self):
        kaiming_init(self.down_proj, bias=0, distribution='uniform')
        constant_init(self.up_proj, 0, bias=0)


    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        
        x = self.adapter_layer_norm(x) * self.gamma + x * self.gammax
        
        down = self.down_proj(x)
        N, L, C = down.shape
        h = int(math.sqrt(L))
        w = h
        down = nlc_to_nchw(down, (h, w))
        down = self.adapter_conv(down)
        down = nchw_to_nlc(down)
        
        down = self.non_linear_func(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        
        up = up * self.scale
        
        
        
        if add_residual:
            up = up + residual
        
        return up






















