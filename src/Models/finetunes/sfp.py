import math




import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import (constant_init, normal_init,
                                        trunc_normal_init)

from ..utils import build_norm_layer, ConvModule, nlc_to_nchw, nchw_to_nlc, resize


'''
From ViTDet:
    @article{li2022exploring,
        title={Exploring plain vision transformer backbones for object detection},
        author={Li, Yanghao and Mao, Hanzi and Girshick, Ross and He, Kaiming},
        journal={arXiv preprint arXiv:2203.16527},
        year={2022}
    }
'''

# wrapper
class upscaling(nn.Module):
    def __init__(self, 
                 scale_factor:int = 2, 
                 mode: str = 'nearest', 
                 align_corners: bool = False,
                 
                 
                 ) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class SimpleFeaturePyramid(nn.Module):
    '''
    Args:

    Shape:
        in:
            (N, L, C)
        out: List
            4 outputs from 4 stages
            Stage 1: shape(N, c_1, h/4 * w/4)
            Stage 2: shape(N, c_2, h/8 * w/8)
            Stage 3: shape(N, c_3, h/16 * w/16) 
            Stage 4: shape(N, c_4, h/32 * w/32)
            
            L = h/16 * w/16
    
    '''
    def __init__(self, 
                in_channels, 
                
                #ft_cfg
                out_channels, 
                scale_factors: list =[4.0, 2.0, 1.0, 0.5], 
                norm_cfg=dict(
                    norm_type='LayerNorm2d'   
                ), 
                     
            ) -> None:
        super().__init__()
        
        self.scale_factors = scale_factors
        
        
        dim = in_channels
        self.stages = []
        
        
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    build_norm_layer(num_features=dim // 2, **norm_cfg)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            
            layers.extend([
                ConvModule(
                    out_dim, 
                    out_channels, 
                    kernel_size=1, 
                    conv_cfg=dict(conv_type='Conv2d'),
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                ), 
                ConvModule(
                    out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    padding=1, 
                    conv_cfg=dict(conv_type='Conv2d'),
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                )
            ])
            
            layers = nn.Sequential(*layers)
            
            self.add_module(f"simfp_{idx+1}", layers)
            self.stages.append(layers)
            
            # self.init_weights()
        
    
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d):
    #             constant_init(m, val=0)
    
            
    def forward(self, x):
        '''
        Shape:
            in:
                (N, L, C)
            out: List
                4 outputs from 4 stages
                Stage 1: shape(N, c_1, h/4, w/4)
                Stage 2: shape(N, c_2, h/8, w/8)
                Stage 3: shape(N, c_3, h/16, w/16) 
                Stage 4: shape(N, c_4, h/32, w/32)
                
                L = h/16 * w/16
        
        '''
        N, L, C = x.shape
        h = w = int(L ** 0.5)
        x = nlc_to_nchw(x, (h, w))
        outs = []
        
        for i, layer in enumerate(self.stages):
            out = layer(x)
            outs.append(out)
            
        return outs


class SimpleFeaturePyramid_V2(nn.Module):
    '''
    Args:

    Shape:
        in:
            (N, L, C)
        out: List
            4 outputs from 4 stages
            Stage 1: shape(N, c_1, h/4 * w/4)
            Stage 2: shape(N, c_2, h/8 * w/8)
            Stage 3: shape(N, c_3, h/16 * w/16) 
            Stage 4: shape(N, c_4, h/32 * w/32)
            
            L = h/16 * w/16
    
    '''
    def __init__(self, 
                in_channels, 
                
                #ft_cfg
                out_channels, 
                scale_factors: list =[4.0, 2.0, 1.0, 0.5], 
                pyramid_factors: list = [0.1, 0.1, 1.0, 0.1], 
                
                norm_cfg=dict(
                    norm_type='LayerNorm2d'   
                ), 
                     
            ) -> None:
        super().__init__()
        
        self.scale_factors = scale_factors
        
        
        dim = in_channels
        self.stages = []
        
        
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    build_norm_layer(num_features=dim // 2, **norm_cfg)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    # nn.Conv2d(dim, dim //2, kernel_size=1),
                    # build_norm_layer(num_features=dim // 2, **norm_cfg)[1],
                    # nn.GELU(),
                    # upscaling(scale_factor=2, mode='bilinear'), 
                    # nn.Conv2d(dim // 2, dim // 4, kernel_size=3, padding=1), 
                    # upscaling(scale_factor=2, mode='bilinear'), 
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1), 
                        upscaling(scale_factor=2, mode='bilinear'),
                ]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
            
            layers.extend([
                ConvModule(
                    out_dim, 
                    out_channels, 
                    kernel_size=1, 
                    conv_cfg=dict(conv_type='Conv2d'),
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                ), 
                ConvModule(
                    out_channels, 
                    out_channels, 
                    kernel_size=3, 
                    padding=1, 
                    conv_cfg=dict(conv_type='Conv2d'),
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                )
            ])
            
            layers = nn.Sequential(*layers)
            
            self.add_module(f"simfp_{idx+1}", layers)
            self.stages.append(layers)
            
            self.pyramid_scalers = nn.Parameter(torch.tensor(pyramid_factors))
            
            
            # self.init_weights()
        
    
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d):
    #             constant_init(m, val=0)
    
            
    def forward(self, x):
        '''
        Shape:
            in:
                (N, L, C)
            out: List
                4 outputs from 4 stages
                Stage 1: shape(N, c_1, h/4, w/4)
                Stage 2: shape(N, c_2, h/8, w/8)
                Stage 3: shape(N, c_3, h/16, w/16) 
                Stage 4: shape(N, c_4, h/32, w/32)
                
                L = h/16 * w/16
        
        '''
        N, L, C = x.shape
        h = w = int(L ** 0.5)
        x = nlc_to_nchw(x, (h, w))
        outs = []
        
        for i, layer in enumerate(self.stages):
            out = layer(x) * self.pyramid_scalers[i]
            outs.append(out)
            
        return outs
