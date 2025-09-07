import os
import warnings
import json

from typing import Dict

from typing import Sequence, Union

import mmcv
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from .transform_builder import build_transform


class Foreground_Prompt_Dataset(Dataset):
    '''
    Custom dataset for SOD. An example of file structure
    is as followed.

    .. code-block:: none

        ├── dataset
        │   ├── my_dataset
        │   │   ├── prompt.pt
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
                 prompt_suffix='.pt',
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
            self.prompts_path = os.path.join(dataset_root, 'train_latent_prompt'+prompt_suffix)
        elif mode == 'val':
            self.images_path = os.path.join(dataset_root, 'images', 'validation')
            self.seg_maps_path = os.path.join(dataset_root, 'annotations', 'validation')
            self.prompts_path = os.path.join(dataset_root, 'val_latent_prompt'+prompt_suffix)
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
                
        # load prompts
        self.prompts = torch.load(self.prompts_path) # dict('{img_name}': Tensor(1, prompt_length, c))
        # padding
        # max_length = max([p.size(1) for p in self.prompts.values()])
        # self.prompts = {k: torch.nn.functional.pad(p, (0, max_length - p.size(0)), mode='constant', value=0) for k, p in self.prompts.items()}
        padding_length = 30
        self.prompts = {k: torch.nn.functional.pad(p, (0, 0, 0, padding_length - p.size(1)), mode='constant', value=0) for k, p in self.prompts.items()}
        
        
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
        
        # prompts
        image_name = os.path.basename(self.image_file_paths[idx])
        prompt = self.prompts[image_name].float()
        padding_prompt = None
        
        #create Tensor 
        
        #image, seg_map 
        if self.transform:
            image, seg_map = self.transform(image, seg_map)
            #image: (c, h, w)
            #segmap: (1, h, w)
            
            
        out_dict = dict(
            prompt=prompt, # Tensor (1, 30)
            image=image, # Tensor
            seg_map=seg_map, # Tensor
        )
            
        return out_dict
    
    
    @staticmethod
    def collate_fn(batch):
        prompts = [item['prompt'].squeeze(0) for item in batch]
        images = [item['image'] for item in batch]
        seg_maps = [item['seg_map'] for item in batch]
        prompts_tensor = torch.stack(prompts)
        images_tensor = torch.stack(images)
        seg_maps_tensor = torch.stack(seg_maps)
        
        inputs = dict(
            prompt=prompts_tensor,
            image=images_tensor
        )
        label = dict(
            seg_maps=seg_maps_tensor
        )
        
        
        return inputs, label
    
    
    
    
# ===============================================================
# =================== Config ====================================
    
class Foreground_Prompt_Dataset_Config():
    '''
    
    '''

    def __init__(self,
                 data_root, 
                 train_pipeline, 
                 test_pipeline, 
                 truncate_ratio = None) -> None:
        self.data_root = data_root
        
        self.train_pipeline = train_pipeline
        self.test_pipeline = test_pipeline
        
        self.truncate_ratio = truncate_ratio
        

    @property
    def dataset_train(self):
        return Foreground_Prompt_Dataset(dataset_root=self.data_root, 
                                  mode='train', 
                                  transforms=self.train_pipeline,
                                  truncate_ratio=self.truncate_ratio)
    
    @property
    def dataset_val(self):
        return Foreground_Prompt_Dataset(dataset_root=self.data_root, 
                                  mode='val', 
                                  transforms=self.test_pipeline,
                                  truncate_ratio=self.truncate_ratio)  
    
    
    
    
    
    
    
    
    

class Joint_Foreground_Prompt_Dataset(Dataset):
    '''
    Custom dataset for SOD. An example of file structure
    is as followed.

    .. code-block:: none

        ├── dataset
        │   ├── my_dataset
        │   │   ├── prompt.pt
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
        
        
        class_suffix: If None, then no class labels will be used.

    '''
    def __init__(self, dataset_root: Union[str, Sequence[str]], 
                 prompt_suffix='.pt',
                 class_suffix='.json',
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
       
       
        # self.images_path = []
        # self.seg_maps_path = []
        # self.prompts_path = []
        # self.class_path = []
        
        self.images_path = dict()
        self.seg_maps_path = dict()
        self.prompts_path = dict()
        
        if class_suffix is not None:
            self.class_path = dict()
        else:
            self.class_path = None
        # {dataset name: path, ...}
        
        self.dataset_name_path_dict = {os.path.basename(root): root for root in dataset_root}
        # {'dataset1 name': dataset1 path, ...}
        # [os.path.basename(root) for root in dataset_root]
       
        # choose mode and load dataset path
        if mode == 'train':
            for dataset_name, dataset_path in self.dataset_name_path_dict.items():
                self.images_path[dataset_name] = os.path.join(dataset_path, 'images', 'training')
                self.seg_maps_path[dataset_name] = os.path.join(dataset_path, 'annotations', 'training')
                self.prompts_path[dataset_name] = os.path.join(dataset_path, 'train_latent_prompt'+prompt_suffix)
                
                if self.class_path is not None:
                    self.class_path[dataset_name] = os.path.join(dataset_path, 'train_class_results'+class_suffix)
                
        elif mode == 'val':
            for dataset_name, dataset_path in self.dataset_name_path_dict.items():
                self.images_path[dataset_name] = os.path.join(dataset_path, 'images', 'validation')
                self.seg_maps_path[dataset_name] = os.path.join(dataset_path, 'annotations', 'validation')
                self.prompts_path[dataset_name] = os.path.join(dataset_path, 'val_latent_prompt'+prompt_suffix)
                if self.class_path is not None:
                    self.class_path[dataset_name] = os.path.join(dataset_path, 'val_class_results'+class_suffix)
                
        else:
            raise ValueError(f'"mode" must be "train" or "val", but got {mode}')
            
        for path in self.images_path.values():
            assert os.path.exists(path), f"path '{path}' does not exist."
        for path in self.seg_maps_path.values():
            assert os.path.exists(path), f"path '{path}' does not exist."
        # assert os.path.exists(self.images_path), f"path '{self.images_path}' does not exist."
        # assert os.path.exists(self.seg_maps_path), f"path '{self.seg_maps_path}' does not exist."
        
        
        # image_names = [p for p in os.listdir(self.images_path) if p.endswith(img_suffix)]
        # seg_map_names = [p for p in os.listdir(self.seg_maps_path) if p.endswith(seg_map_suffix)]
        
        image_names = dict() # {dataset_name1: [img_names], ...}
        image_names_for_check = []
        for dataset_name, path in self.images_path.items():
            names = [p for p in os.listdir(path) if p.endswith(img_suffix)]
            image_names[dataset_name] = names
            image_names_for_check.extend(names)
            
            
        seg_map_names = dict()
        seg_map_names_for_check = []
        for dataset_name, path in self.seg_maps_path.items():
            names = [p for p in os.listdir(path) if p.endswith(seg_map_suffix)]
            seg_map_names[dataset_name] = names
            seg_map_names_for_check.extend(names)
        
        
        assert len(image_names_for_check) > 0, f"not find any images in {self.images_path}."
        
        # check images and mask
        self._check_img_segmap(image_names=image_names_for_check, seg_map_names=seg_map_names_for_check,
                               img_suffix=img_suffix, seg_map_suffix=seg_map_suffix)
        
        #each file path loading
        # self.image_file_paths = [os.path.join(self.images_path, n) for n in image_names]
        # self.mask_file_paths = [os.path.join(self.seg_maps_path, n) for n in seg_map_names]
        self.image_file_paths = []
        for dataset_name, names in image_names.items():
            self.image_file_paths.extend([os.path.join(self.images_path[dataset_name], n) for n in names])
            
        self.mask_file_paths = []
        for dataset_name, names in seg_map_names.items():
            self.mask_file_paths.extend([os.path.join(self.seg_maps_path[dataset_name], n) for n in names])
                
                
        # load prompts
        self.prompts = dict()
        for dadtaset_name, one_prompts_path in self.prompts_path.items():
            self.prompts = self.prompts | torch.load(one_prompts_path)
            # self.prompts = {**self.prompts, **torch.load(one_prompts_path)}
        # self.prompts = torch.load(self.prompts_path) # dict('{img_name}': Tensor(1, prompt_length, c))
        # padding
        # max_length = max([p.size(1) for p in self.prompts.values()])
        # self.prompts = {k: torch.nn.functional.pad(p, (0, max_length - p.size(0)), mode='constant', value=0) for k, p in self.prompts.items()}
        padding_length = 32
        # self.prompts = {k: torch.nn.functional.pad(p, (0, 0, 0, padding_length - p.size(1)), mode='constant', value=0) for k, p in self.prompts.items()}
        self.prompts = {k: torch.nn.functional.pad(p, (0, padding_length - p.size(1)), mode='constant', value=0) for k, p in self.prompts.items()}
        
        # load class indices
        self.class_labels = dict()
        if self.class_path is not None:
            for dataset_name, one_class_labels_path in self.class_path.items():
                with open(one_class_labels_path, 'r') as f:
                    self.class_labels = self.class_labels | json.load(f)
                    # self.class_labels = {**self.class_labels, **json.load(f)}
        # {'imgname.surrfix': index, ...}. index: str
        self.class_labels = {k: v for k, v in self.class_labels.items() if k in image_names_for_check}
        class_map = dict()
        i = 0
        for name, class_label in self.class_labels.items():
            if class_label not in class_map.keys():
                class_map[class_label] = i
                i += 1
        self.class_label_indices = dict()
        for k, v in self.class_labels.items():
            self.class_label_indices[k] = class_map[v]
            
        
        
        
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
        
        # prompts
        image_name = os.path.basename(self.image_file_paths[idx])
        prompt = self.prompts[image_name].long()
        # padding_prompt = None
        
        # class indices
        if self.class_path is not None:
            class_label = torch.tensor(self.class_label_indices[image_name], dtype=torch.long)#.unsqueeze(0)
        else:
            class_label = None
        
        #create Tensor 
        
        #image, seg_map 
        if self.transform:
            image, seg_map = self.transform(image, seg_map)
            #image: (c, h, w)
            #segmap: (1, h, w)
            
            
        out_dict = dict(
            prompt=prompt, # Tensor (1, 30, 2048)
            class_label=class_label, # Tensor(1, 1) (num_masks, 1)
            image=image, # Tensor
            seg_map=seg_map, # Tensor
        )
            
        return out_dict
    
    
    @staticmethod
    def collate_fn(batch):
        prompts = [item['prompt'].squeeze(0) for item in batch]
        class_labels = [item['class_label'] for item in batch if item['class_label'] is not None]
        images = [item['image'] for item in batch]
        seg_maps = [item['seg_map'] for item in batch]
        
        prompts_tensor = torch.stack(prompts)
        if class_labels == []:
            class_labels_tensor = None
        else:
            class_labels_tensor = torch.stack(class_labels)
            
        images_tensor = torch.stack(images)
        seg_maps_tensor = torch.stack(seg_maps)
        
        inputs = dict(
            prompt=prompts_tensor,
            image=images_tensor
        )
        
        if class_labels_tensor is None:
            labels = dict(
                label_mask=seg_maps_tensor
            )
        else:
            labels = dict(
                class_labels=class_labels_tensor,
                label_mask=seg_maps_tensor
            )
        
        return inputs, labels
    
    
    
    
# ===============================================================
# =================== Config ====================================
    
class Joint_Foreground_Prompt_Dataset_Config():
    '''
    
    '''

    def __init__(self,
                 data_root, 
                 
                 transform_cfg_name: str,
                 transform_cfg_args: Dict,
                 
                 prompt_suffix='.pt',
                 class_suffix='.json',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 truncate_ratio = None) -> None:
        self.data_root = data_root
        
        transform_cfg_instance = build_transform(transform_cfg_name, transform_cfg_args)
        self.train_pipeline = transform_cfg_instance.get_train_pipeline_compose
        # compose object
        self.test_pipeline = transform_cfg_instance.get_validate_pipeline_compose
        
        
        self.prompt_suffix = prompt_suffix
        self.class_suffix = class_suffix
        self.img_suffix = img_suffix
        self.seg_map_suffix = seg_map_suffix
        
        self.truncate_ratio = truncate_ratio
        

    @property
    def dataset_train(self):
        return Joint_Foreground_Prompt_Dataset(
            dataset_root=self.data_root, 
            prompt_suffix=self.prompt_suffix,
            class_suffix=self.class_suffix,
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
                                               
            mode='train', 
            transforms=self.train_pipeline,
            truncate_ratio=self.truncate_ratio
        )
    
    @property
    def dataset_val(self):
        return Joint_Foreground_Prompt_Dataset(
            dataset_root=self.data_root, 
            prompt_suffix=self.prompt_suffix,
            class_suffix=self.class_suffix,
            img_suffix=self.img_suffix,
            seg_map_suffix=self.seg_map_suffix,
            
            mode='val', 
            transforms=self.test_pipeline,
            truncate_ratio=self.truncate_ratio
        ) 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    