import random
from typing import List, Union, Tuple, Optional, Sequence

import numpy as np


import torch
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

import mmengine, mmcv

from mmcv.transforms.utils import cache_randomness


# from ...Models.restorers import ResUNet_Config
# class Camouflageator(object):
#     def __init__(self, 
                 
#                  trained_weights: str,
#                  prob: float = 0.3,
                 
#                  ) -> None:
#         self.prob = prob
#         camouflageator_cfg_inst = ResUNet_Config(
#             pretrained_weights=trained_weights, 
#             restorer_cfg=dict(
#                 channel=3, 
#                 out_channel=3,
#                 filters=[64, 128, 256, 512],
#             ), 
#             loss_decode=dict(
#                 loss_type='FG_FidelityLoss', 
#                 reduction="mean",
#                 loss_weight=1.0,
#                 loss_name="loss_fidel"
        
#             )
#         )
#         self.camouflageator, _ = camouflageator_cfg_inst.model
        
#     def __call__(self, image, target=None):
#         if random.random() < self.prob:
#             self.camouflageator.eval()
#             with torch.no_grad():
#                 image = image.unsqueeze(0)
#                 inputs = dict(image=image)
#                 _, preds = self.camouflageator.predict(inputs, return_logits=True)
#                 pred_image = preds['pred_image']
#                 image = pred_image.squeeze(0)
#         return image, target




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    '''
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
    
    Shape:
        in: (h, w, c) or (h, w)
        out: (c, h, w) or (1, h, w)
    '''
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = F.to_tensor(target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


# class Resize(object):
#     def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
#         self.size = size  # [h, w]
#         self.resize_mask = resize_mask

#     def __call__(self, image, target=None):
#         image = F.resize(image, self.size)
#         if self.resize_mask is True:
#             target = F.resize(target, self.size)

#         return image, target
    
    
class Resize(object):
    '''
    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.
    
    
    '''
    def __init__(self, 
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 
                 keep_ratio: bool = False,
                 resize_mask: bool = True, 
                 ):
        
        if isinstance(scale, int):
                self.scale = (scale, scale)
        else:
            self.scale = scale
            
        self.keep_ratio = keep_ratio

        self.resize_mask = resize_mask
        
    def _resize_img(self, image):
        if self.keep_ratio:
            h, w = image.shape[1:]
            new_size= mmcv.rescale_size(
                (w, h), self.scale, return_scale=False
            )
            nw, nh = new_size
            rescaled_img = F.resize(image, (nh, nw))
        else:
            rescaled_img = F.resize(image, self.scale)
            
        return rescaled_img
            

    def __call__(self, image, target=None):
        image = self._resize_img(image)
        if self.resize_mask is True and target is not None:
            target = self._resize_img(target)
        return image, target

    
class RandomResize(object):
    '''
    Args:
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            e.g. (1024, 512) or
                (
                    (512, 256), #(min_d1, min_d2)
                    (2048, 1024)  #(max_d1, max_d2)
                )
            Defaults to None.
            
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
        e.g. (0.5, 2.0)
            Defaults to None.
    '''
    def __init__(self, scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]], 
                 ratio_range: Optional[Tuple[float, float]], 
                 keep_ratio: bool = True, 
                 resize_mask: bool = True                
            ):
        self.scale = scale
        self.ratio_range = ratio_range
        
        self.keep_ratio = keep_ratio
        self.resize_mask = resize_mask
        
    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> tuple:
        """Private function to randomly sample a scale from a list of tuples.

        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert mmengine.is_list_of(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale
    
    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float,
                                                              float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.

        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
            
        Shape:
            in: Tensor (c, h, w)
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale
    
    @cache_randomness
    def _random_scale(self):
        '''
        
        Returns:
            scale (Tuple): The targeted scale of the image to be resized.
        
        '''
        if mmengine.is_tuple_of(self.scale, int):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(
                self.scale,  # type: ignore
                self.ratio_range)
        elif mmengine.is_seq_of(self.scale, tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError('Do not support sampling function '
                                      f'for "{self.scale}"')
        
        return scale
    
    def _resize_img(self, image, scale):
        '''
        Shape:
            in: Tensor (c, h, w)
        '''
        if self.keep_ratio:
            h, w = image.shape[1:]
            new_size= mmcv.rescale_size(
                (w, h), scale, return_scale=False
            )
            nw, nh = new_size
            rescaled_img = F.resize(image, (nh, nw))
        else:
            rescaled_img = F.resize(image, scale)
        return rescaled_img
            

    def __call__(self, image, target=None):
        scale = self._random_scale()
        image = self._resize_img(image, scale)
        # image = F.resize(image, scale)
        
        if self.resize_mask is True:
            target = self._resize_img(target, scale)
            # target = F.resize(target, scale)
        return image, target
    

class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            # ow, oh = img.size
            oh, ow = img.shape[-2:]
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target=None):
        image = self.pad_if_smaller(image)
        
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        if target is not None:
            target = self.pad_if_smaller(target)
            target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size: int):
        self.size = size
        
    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            # ow, oh = img.size
            oh, ow = img.shape[-2:]
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target=None):
        image = self.pad_if_smaller(image)
        
        image = F.center_crop(image, self.size)
        if target is not None:
            target = self.pad_if_smaller(target)
            target = F.center_crop(target, self.size)
        return image, target


# ==================debug==============================

def debug():
    test_transforms = RandomResize(
        scale=(500, 352), 
        ratio_range=(0.5, 2.0),
        keep_ratio=True, 
        resize_mask=True
    )

    input = torch.randn(3, 380, 450)
    label = torch.randn(1, 380, 450)
    
    output = test_transforms(input, label)
    print(output[0].shape, output[1].shape)
    
def resize_debug():
    test_transforms = Resize(
        scale=(2048, 512), 
        keep_ratio=True,
        resize_mask=True
    )

    input = torch.randn(3, 1024, 640)
    label = torch.randn(1, 1024, 640)

    output = test_transforms(input, label)
    print(output[0].shape, output[1].shape)


if __name__ == '__main__':
    resize_debug()

