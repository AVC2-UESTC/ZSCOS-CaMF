import inspect
from typing import Dict, Union

import torch
import torch.nn as nn


def build_padding_layer(padding_type: str,
                        padding: Union[int, tuple]) -> nn.Module:
    """Build padding layer.

    Args:
        The padding layer config, which should contain:
            - padding_type (str): Layer type. 
            - padding: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    
    
    supported_padding_type = {'ZeroPad2d': nn.ZeroPad2d, 
                              'ReflectionPad2d': nn.ReflectionPad2d, 
                              'ReplicationPad2d': nn.ReplicationPad2d}
    
    layer = supported_padding_type[padding_type](padding)
    
    return layer























