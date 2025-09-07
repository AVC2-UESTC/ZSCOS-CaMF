

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .t2t_vit import T2t_vit_t_14
from .Transformer import Transformer
from .Transformer import token_Transformer
from .Decoder import Decoder


from ...builder import build_loss

from ...utils import resize

class ImageDepthNet(nn.Module):
    def __init__(self, img_size):
        super(ImageDepthNet, self).__init__()

        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=False, args=None)

        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder(embed_dim=384, token_dim=64, depth=2, img_size=img_size)
        
        self.img_size = img_size
        
        self.out_channels = 1
        self.align_corners = False
        self.threshold = None
        
        

    def forward(self, inputs):
        image_Input = inputs['image']
        image_Input = self.preprocessor(image_Input)

        B, _, _, _ = image_Input.shape
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)

        # VST Convertor
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]

        outputs = self.decoder(saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens, rgb_fea_1_8, rgb_fea_1_4)
        mask = outputs[0][-1]
        return mask
    
    def preprocessor(self, image):
        return resize(image, (self.img_size, self.img_size))

    def postprocessor(self, inputs, ori_size):
        return resize(inputs, (ori_size, ori_size))


    def logits(self, inputs) -> Tensor:
        h, w = inputs['image'].shape[2:]
        results = self.forward(inputs)
        seg_map = results['masks']
        seg_map = self.postprocessor(seg_map, h)
        return seg_map
    
        
        # raw: without sigmoid
    

    def predict(self, inputs,
                return_logits: bool = False
                ) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tensor): 
            

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        img_size = inputs['image'].shape[2:]
        
        results = self.forward(inputs)
        seg_logits = results['masks']
        logits_prob = torch.sigmoid(seg_logits) # for metric computing
        seg_logits = resize(
            input=seg_logits,
            size=img_size,
            mode='bilinear',
            align_corners=self.align_corners)
        # (N, out_channels, H, W)
        if self.out_channels == 1:
            seg_probs = torch.sigmoid(seg_logits)
            if self.threshold is not None:
                seg_map = (seg_probs > self.threshold).float()
            else:
                seg_map = seg_probs
            # (N, 1, H, W)
            seg_map = seg_map.squeeze(dim=1)
            #(N, H, W)
        else:
            seg_probs = F.softmax(seg_logits, dim=1)
            seg_map = torch.argmax(seg_probs, dim=1)
            # (N, H, W)
        
        if return_logits:
            return seg_map, logits_prob
        else:
            return seg_map













class VST_Segmentation_Config_depre():
    def __init__(self, img_size) -> None:
        
                
        
        self.img_size = img_size
        
        
    @property
    def model(self):
        return ImageDepthNet(**self.__dict__)


def VST_Segmentation_fgseg_cfg():
    args = VST_Segmentation_Config_depre(img_size=224)
    return args    

