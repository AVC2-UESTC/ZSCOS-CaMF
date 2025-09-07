import torch
import torch.nn as nn

import torch.nn.functional as F

class Clamp(nn.Module):
    """Clamp activation layer.

    This activation function is to clamp the feature map value within
    :math:`[min, max]`. More details can be found in ``torch.clamp()``.

    Args:
        min (Number | optional): Lower-bound of the range to be clamped to.
            Default to -1.
        max (Number | optional): Upper-bound of the range to be clamped to.
            Default to 1.
    """

    def __init__(self, min: float = -1., max: float = 1.):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Clamped tensor.
        """
        return torch.clamp(x, min=self.min, max=self.max)



def build_activation_layer(
                        #act_cfg   
                           act_type: str = None, 
                           # layer args
                           layer_args = None, 
                           
                           ) -> nn.Module:
    """Build activation layer.
    
    Now supported Layer type: 'GELU', 'ReLU', 'SiLU'

    Args:
        
        act_type (str): Layer type.
        
        # layer args: Args needed to instantiate an activation layer.
             
            
            # ReLU    
            inplace: bool
                Default: False
                
            #LeakyReLU
            negative_slope (float): Default: 1e-2
            inplace: Default: False
           
    Returns:
        nn.Module: Created activation layer.
    """
    
    spported_act = {
        'GELU': nn.GELU,
        'ReLU': nn.ReLU,
        'SiLU': nn.SiLU,
        'LeakyReLU': nn.LeakyReLU,
        'PReLU': nn.PReLU,
        'ReLU6': nn.ReLU6,
        'RReLU': nn.RReLU,
        'ELU': nn.ELU,
        'Sigmoid': nn.Sigmoid,
    }
    
    if layer_args is None:
        layer = spported_act[act_type]()
    else:
        layer = spported_act[act_type](**layer_args)

    return layer



#============================debug==============================

def debug():
    
    layer = build_activation_layer(act_type='GELU')
    layer2 = build_activation_layer(act_type='ReLU', 
                                    layer_args=dict(
                                        inplace=True
                                    ))
    
    
    
    print(layer)
    print(layer2)

    
    
if __name__ == '__main__':
    debug()














