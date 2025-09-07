import inspect
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last  or channels_first (default). 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, num_channels: int, eps: float = 1e-6, data_format="channels_first") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.num_channels = (num_channels, )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    def extra_repr(self) -> str:
        return '{num_channels}, eps={eps}, '.format(**self.__dict__)


def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'


def build_norm_layer(num_features: int, # 
                                        # layer norm: normalized_shape : embed_dim
                     postfix: Union[int, str] = '',
                     
                     # norm cfg
                     norm_type: str = None, 
                     requires_grad: bool = True,
                     
                     # norm args
                     layer_args = None,
                     
                     ) -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        norm cfg: The norm layer config, which should contain:

            - norm_type (str): Layer type.
            - requires_grad (bool, optional): Whether stop gradient updates.
            
            - layer_args (dict): additional args
            
        
        num_features (int): Number of input channels.
        
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """

    batchnorms = {'BatchNorm2d': nn.BatchNorm2d, 
                  'BatchNorm1d': nn.BatchNorm1d, 
                  'BatchNorm3d': nn.BatchNorm3d, 
                  'syncBatchNorm': nn.SyncBatchNorm}
    
    instancenorms = {'InstanceNorm2d': nn.InstanceNorm2d, 
                     'InstanceNorm1d': nn.InstanceNorm1d, 
                     'InstanceNorm3d': nn.InstanceNorm3d}
    
    layernorms = {'LayerNorm': nn.LayerNorm, 
                  'LayerNorm2d': LayerNorm2d
                  }
    
    # other_norms = {'groupNorm': nn.GroupNorm}
    
    supported_norms = {**batchnorms, **instancenorms, **layernorms}
    
    norm_layer = supported_norms[norm_type] #get "type class" of norms
    
    # get name of norms
    abbr = infer_abbr(norm_layer)
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    
    #build norm layer
    if layer_args is None:
        layer = norm_layer(num_features)
    else:    
        layer = norm_layer(num_features, **layer_args)
        
    for param in layer.parameters():
        param.requires_grad = requires_grad


    
    return name, layer


# def build_norm_layer(num_features: int, # 
#                                         # layer norm: normalized_shape : embed_dim
#                      postfix: Union[int, str] = '',
                     
#                      **norm_cfg):
    
    
#     batchnorms = {'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm3d': nn.BatchNorm3d}
#     instancenorms = {'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm3d': nn.InstanceNorm3d}
#     other_norms = {'LayerNorm': nn.LayerNorm, 'groupNorm': nn.GroupNorm, 'syncBatchNorm': nn.SyncBatchNorm}
#     all_norms = {**batchnorms, **instancenorms, **other_norms}
#     b_i_norms = {**batchnorms, **instancenorms}
    
#     norm_layer = norm_cfg['norm_type']
    
#     # get name of norms
#     abbr = infer_abbr(norm_layer)
#     assert isinstance(postfix, (int, str))
#     name = abbr + str(postfix)
    
#     #build norm layer
    
#     layer = norm_layer()
    
    
    
    


#==========================debug==================================

def debug():
    
    norm_cfg = dict(norm_type='BatchNorm2d', 
                     requires_grad=False,
                     )
    
    name, layer = build_norm_layer(num_features=64, postfix=1, **norm_cfg)
    print(name, layer)
    
    
    norm_cfg = dict(norm_type='LayerNorm', 
                     requires_grad=True,
                     )
    
    name, layer = build_norm_layer(
        num_features=64, 
        **norm_cfg
    )
    
    print(name, layer)
    
    norm_cfg = dict(norm_type='LayerNorm2d', 
                     requires_grad=True,
                     layer_args=dict(eps=1e-6)
                     )
    
    name, layer = build_norm_layer(
        num_features=64,
        **norm_cfg
    )
    
    print(name, layer)
    
if __name__ == "__main__":
    debug()
    






