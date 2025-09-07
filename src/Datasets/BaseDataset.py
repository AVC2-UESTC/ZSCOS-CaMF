import os
import warnings

from typing import Dict, Union, Sequence

import mmcv
import torch

from torch.utils.data import Dataset
from torchvision import transforms


from .transform_builder import build_transform


class Foreground_Dataset(Dataset):
    '''
    Custom dataset for SOD. An example of file structure
    is as followed.

    .. code-block:: none

        ├── dataset
        │   ├── my_dataset
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val

    The img/gt_semantic_seg pair of BaseSegDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    
    
    Args:
        mode: 'train', 'val'

    '''
    def __init__(self, dataset_root, 
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 mode: str = 'train', 
                 transforms=None,
                 truncate_ratio: float = None,
                 ):
        
        assert os.path.exists(dataset_root), f"path '{dataset_root}' does not exist."
       
        # choose mode and load dataset path
        if mode == 'train':
            self.images_path = os.path.join(dataset_root, 'images', 'training')
            self.seg_maps_path = os.path.join(dataset_root, 'annotations', 'training')
        elif mode == 'val':
            self.images_path = os.path.join(dataset_root, 'images', 'validation')
            self.seg_maps_path = os.path.join(dataset_root, 'annotations', 'validation')
        else:
            raise ValueError(f'Invalid mode {mode}')
        
        assert os.path.exists(self.images_path), f"path '{self.images_path}' does not exist."
        assert os.path.exists(self.seg_maps_path), f"path '{self.seg_maps_path}' does not exist."
        
        
        image_names = [p for p in os.listdir(self.images_path) if p.endswith(img_suffix)]
        seg_map_names = [p for p in os.listdir(self.seg_maps_path) if p.endswith(seg_map_suffix)]
        
        assert len(image_names) > 0, f"not find any images in {self.images_path}."
        
        # check images and mask
        
        self._check_img_segmap(image_names=image_names, seg_map_names=seg_map_names,
                               img_suffix=img_suffix, seg_map_suffix=seg_map_suffix)
        
        #each file path loading
        self.image_file_paths = [os.path.join(self.images_path, n) for n in image_names]
        self.mask_file_paths = [os.path.join(self.seg_maps_path, n) for n in seg_map_names]
                
        
        self.transform = transforms
        
        if truncate_ratio is not None:
            assert 0 < truncate_ratio < 1, f"truncate_ratio should be in (0, 1). But truncate_ratio={truncate_ratio}"
            step = int(1/truncate_ratio)
        
            self.image_file_paths = self.image_file_paths[0::step]
            self.mask_file_paths = self.mask_file_paths[0::step]
            

    def _check_img_segmap(self, image_names, seg_map_names, img_suffix, seg_map_suffix):
        re_seg_map_names = []
        for p in image_names:
            seg_map_name = p.replace(img_suffix, seg_map_suffix)
            assert seg_map_name in seg_map_names, f"{p} has no corresponding mask."
            re_seg_map_names.append(seg_map_name)
        seg_map_names = re_seg_map_names
        

    def __len__(self):
        return len(self.image_file_paths)

    
    def __getitem__(self, idx):
        image = mmcv.imread(self.image_file_paths[idx])
        image = mmcv.bgr2rgb(image)
        seg_map = mmcv.imread(self.mask_file_paths[idx], flag='grayscale')
        # seg_map = mmcv.bgr2gray(seg_map)
        # ndarray
        # seg_map:(h, w)
        # image: (h, w, c)
        
        #create Tensor 
        
        #image, seg_map 
        if self.transform:
            image, seg_map = self.transform(image, seg_map)
            #image: (c, h, w)
            #segmap: (1, h, w)
            
        return image, seg_map


class Joint_Foreground_Dataset(Dataset):
    '''
    Custom dataset for SOD. An example of file structure
    is as followed.

    .. code-block:: none

        ├── dataset
        │   ├── my_dataset1
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{img_suffix}
        │   │   │   │   ├── yyy{img_suffix}
        │   │   │   │   ├── zzz{img_suffix}
        │   │   │   ├── val
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   │   ├── zzz{seg_map_suffix}
        │   │   │   ├── val
        |   |── my_dataset2

    The img/gt_semantic_seg pair of BaseSegDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{img_suffix}`` and ``xxx{seg_map_suffix}`` (extension is also included
    in the suffix). If split is given, then ``xxx`` is specified in txt file.
    Otherwise, all files in ``img_dir/``and ``ann_dir`` will be loaded.
    
    
    Args:
        mode: 'train', 'val'

    '''
    def __init__(self, dataset_root: Union[str, Sequence[str]], 
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 mode: str = 'train', 
                 transforms=None,
                 truncate_ratio: float = None,
                 ):
        if isinstance(dataset_root, str):
            assert os.path.exists(dataset_root), f"path '{dataset_root}' does not exist."
            dataset_root = [dataset_root, ]
        elif isinstance(dataset_root, (list, tuple)):
            for root in dataset_root:
                assert os.path.exists(root), f"path '{root}' does not exist."
        else:
            raise TypeError(f'"dataset_root" must be a string or a sequence of strings, '
                            f'but got {type(dataset_root)}')
       
        # choose mode and load dataset path
        self.images_path_dict = dict()
        self.seg_maps_path_dict = dict()
        
        self.dataset_name_path_dict = {os.path.basename(root): root for root in dataset_root}
        # {'dataset1 name': dataset1 path, ...}
        # [os.path.basename(root) for root in dataset_root]
       
        
        # choose mode and load dataset path
        if mode == 'train':
            for dataset_name, dataset_path in self.dataset_name_path_dict.items():
                self.images_path_dict[dataset_name] = os.path.join(dataset_path, 'images', 'training')
                self.seg_maps_path_dict[dataset_name] = os.path.join(dataset_path, 'annotations', 'training')
                
        elif mode == 'val':
            for dataset_name, dataset_path in self.dataset_name_path_dict.items():
                self.images_path_dict[dataset_name] = os.path.join(dataset_path, 'images', 'validation')
                self.seg_maps_path_dict[dataset_name] = os.path.join(dataset_path, 'annotations', 'validation')
                
        else:
            raise ValueError(f'"mode" must be "train" or "val", but got {mode}')
            
        for path in self.images_path_dict.values():
            assert os.path.exists(path), f"path '{path}' does not exist."
        for path in self.seg_maps_path_dict.values():
            assert os.path.exists(path), f"path '{path}' does not exist."
        
            
        image_names = dict() # {dataset_name1: [img_names], ...}
        image_names_for_check = []
        for dataset_name, path in self.images_path_dict.items():
            names = [p for p in os.listdir(path) if p.endswith(img_suffix)]
            image_names[dataset_name] = names
            image_names_for_check.extend(names)
            
            
        seg_map_names = dict()
        seg_map_names_for_check = []
        for dataset_name, path in self.seg_maps_path_dict.items():
            names = [p for p in os.listdir(path) if p.endswith(seg_map_suffix)]
            seg_map_names[dataset_name] = names
            seg_map_names_for_check.extend(names)
        
        
        assert len(image_names_for_check) > 0, f"not find any images in {self.images_path_dict}."
        
        self._check_img_segmap(image_names=image_names_for_check, seg_map_names=seg_map_names_for_check,
                               img_suffix=img_suffix, seg_map_suffix=seg_map_suffix)
        
        #each file path loading
        # self.image_file_paths = [os.path.join(self.images_path, n) for n in image_names]
        # self.mask_file_paths = [os.path.join(self.seg_maps_path, n) for n in seg_map_names]
        self.image_file_paths = []
        for dataset_name, names in image_names.items():
            self.image_file_paths.extend([os.path.join(self.images_path_dict[dataset_name], n) for n in names])
            
        self.mask_file_paths = []
        for dataset_name, names in seg_map_names.items():
            self.mask_file_paths.extend([os.path.join(self.seg_maps_path_dict[dataset_name], n) for n in names])
                
        
        self.transform = transforms
        
        if truncate_ratio is not None:
            assert 0 < truncate_ratio < 1, f"truncate_ratio should be in (0, 1). But truncate_ratio={truncate_ratio}"
            step = int(1/truncate_ratio)
        
            self.image_file_paths = self.image_file_paths[0::step]
            self.mask_file_paths = self.mask_file_paths[0::step]
            

    def _check_img_segmap(self, image_names, seg_map_names, img_suffix, seg_map_suffix):
        re_seg_map_names = []
        for p in image_names:
            seg_map_name = p.replace(img_suffix, seg_map_suffix)
            assert seg_map_name in seg_map_names, f"{p} has no corresponding mask."
            re_seg_map_names.append(seg_map_name)
        seg_map_names = re_seg_map_names
        

    def __len__(self):
        return len(self.image_file_paths)

    
    def __getitem__(self, idx):
        image = mmcv.imread(self.image_file_paths[idx])
        image = mmcv.bgr2rgb(image)
        seg_map = mmcv.imread(self.mask_file_paths[idx], flag='grayscale')
        # seg_map = mmcv.bgr2gray(seg_map)
        # ndarray
        # seg_map:(h, w)
        # image: (h, w, c)
        
        #create Tensor 
        
        #image, seg_map 
        if self.transform:
            image, seg_map = self.transform(image, seg_map)
            #image: (c, h, w)
            #segmap: (1, h, w)
            
        out_dict = dict(
            image=image, # Tensor
            seg_map=seg_map, # Tensor
        )
            
        return out_dict
    
    @staticmethod
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        seg_maps = [item['seg_map'] for item in batch]
        
        
            
        images_tensor = torch.stack(images)
        seg_maps_tensor = torch.stack(seg_maps)
        
        inputs = dict(
            image=images_tensor
        )
        
        labels = dict(
            label_mask=seg_maps_tensor
        )
        
        
        return inputs, labels



class Joint_Foreground_Dataset_Config():
    '''
    
    '''

    def __init__(self,
                 data_root, 
                 
                 transform_cfg_name: str,
                 transform_cfg_args: Dict,
                 
                 
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 truncate_ratio = None) -> None:
        self.data_root = data_root
        
        transform_cfg_instance = build_transform(transform_cfg_name, transform_cfg_args)
        self.train_pipeline = transform_cfg_instance.get_train_pipeline_compose
        # compose object
        self.test_pipeline = transform_cfg_instance.get_validate_pipeline_compose
        
        
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        
        self.truncate_ratio = truncate_ratio
        

    @property
    def dataset_train(self):
        return Joint_Foreground_Dataset(
            dataset_root=self.data_root, 
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
                                               
            mode='train', 
            transforms=self.train_pipeline,
            truncate_ratio=self.truncate_ratio
        )
    
    @property
    def dataset_val(self):
        return Joint_Foreground_Dataset(
            dataset_root=self.data_root, 
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
            
            mode='val', 
            transforms=self.test_pipeline,
            truncate_ratio=self.truncate_ratio
        ) 



#=====================Config=======================================
class Foreground_Dataset_Config():
    '''
    
    '''
    def __init__(self,
                 data_root, 
                 transform_cfg_name: str,
                 transform_cfg_args: Dict, 
                 truncate_ratio: float = None) -> None:
        self.data_root = data_root
        self.truncate_ratio = truncate_ratio
        
        
        transform_cfg_instance = build_transform(transform_cfg_name, transform_cfg_args)
        self.train_pipeline = transform_cfg_instance.get_train_pipeline_compose
        # compose object
        self.test_pipeline = transform_cfg_instance.get_validate_pipeline_compose
        

    @property
    def dataset_train(self):
        return Foreground_Dataset(dataset_root=self.data_root, 
                                  mode='train', 
                                  transforms=self.train_pipeline,
                                  truncate_ratio=self.truncate_ratio)
    
    @property
    def dataset_val(self):
        return Foreground_Dataset(dataset_root=self.data_root, 
                                  mode='val', 
                                  transforms=self.test_pipeline,
                                  truncate_ratio=self.truncate_ratio)

        


# from .transforms import fgseg_transforms

# from .transforms import fgseg 

# def build_transform(transform_prefix: str, transform_pipepline: dict, **kwargs):
#     '''
#     Return:
#         Compose Object
#     '''
    
#     supported_transforms = [item for item in dir(fgseg) if not (item.startswith("__") and item.endswith("__"))]
    
#     transform_instance_list = []

#     if transform_prefix == '':
#         compose_method_name = "Compose"
#         for transform_type, transform_params in transform_pipepline.items():

#             assert transform_type in supported_transforms, f'{transform_type} is not supported'
            
#             transform_class = getattr(fgseg, transform_type)
#             if transform_params is None:
#                 transform_instance = transform_class()
#             else:
#                 transform_instance = transform_class(**transform_params)
#             transform_instance_list.append(transform_instance)
        
#     else: 
#         compose_method_name = transform_prefix + "_" + "Compose"
        
#         for transform_type, transform_params in transform_pipepline.items():
#             transform_type = transform_prefix + "_" + transform_type
#             assert transform_type in supported_transforms, f'{transform_type} is not supported'
            
#             transform_class = getattr(fgseg, transform_type)
#             if transform_params is None:
#                 transform_instance = transform_class()
#             else:
#                 transform_instance = transform_class(**transform_params)
#             transform_instance_list.append(transform_instance)
        
    
#     transform_compose = getattr(fgseg, compose_method_name)(transform_instance_list)
#     # print(transform_instance_list)
#     return transform_compose

       










    

#===========================Debug==================================

def debug1():
    from . import transforms
    
    prefix = 'fgseg'
    r = hasattr(transforms, prefix)
    
    print(r)


    
    
    

if __name__ == "__main__":
    debug1()

















