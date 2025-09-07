
from typing import List, Union, Tuple, Optional, Sequence


from torchvision import transforms as torch_T 




# # from . import classification
# class Classification_Transform_Config():
#     def __init__(self, 
#                  train_pipeline, 
#                  validate_pipeline,
                 
#                  ) -> None:
#         self.train_pipeline = train_pipeline
#         self.validate_pipeline = validate_pipeline
        
    
#     def _compose_transforms(self, mode=None, compose_func='default'):
#         transform_instance_list =[]
        
#         if mode == 'train':
#             pipeline = self.train_pipeline
#         elif mode == 'validate':
#             pipeline = self.validate_pipeline
#         else:
#             raise ValueError('mode must be "train" or "validate"')
        
#         for transform_dict in pipeline:
#             transform_class = getattr(classification, transform_dict['transform_type'])
#             transform_dict.pop('transform_type')
            
#             # whether a dict is empty
#             if not transform_dict:
#                 transform_inst = transform_class()
#             else:
#                 transform_inst = transform_class(**transform_dict)
#             transform_instance_list.append(transform_inst)
            
            
#         if compose_func is None:
#             transform_compose = torch_T.Compose(transform_instance_list)
#         elif compose_func == 'default':
#             transform_compose = getattr(classification, 'Compose')(transform_instance_list)
#         else:
#             raise ValueError('compose_func must be "default" or None')
        
#         return transform_compose
    
    
#     @property
#     def get_train_pipeline_compose(self):
#         return self._compose_transforms(mode='train')
    
#     @property
#     def get_validate_pipeline_compose(self):
#         return self._compose_transforms(mode='validate')
    


from . import fgseg

class Foreground_Segmentation_Transform_Config():
    def __init__(self, 
                 train_pipeline, 
                 validate_pipeline,
                 
                 ) -> None:
        self.train_pipeline = train_pipeline
        self.validate_pipeline = validate_pipeline
        
    
    def _compose_transforms(self, mode=None, compose_func='default'):
        transform_instance_list =[]
        
        if mode == 'train':
            pipeline = self.train_pipeline
        elif mode == 'validate':
            pipeline = self.validate_pipeline
        else:
            raise ValueError('mode must be "train" or "validate"')
        
        for transform_dict in pipeline:
            transform_class = getattr(fgseg, transform_dict['transform_type'])
            transform_dict.pop('transform_type')
            
            # whether a dict is empty
            if not transform_dict:
                transform_inst = transform_class()
            else:
                transform_inst = transform_class(**transform_dict)
            transform_instance_list.append(transform_inst)
            
            
        if compose_func is None:
            transform_compose = torch_T.Compose(transform_instance_list)
        elif compose_func == 'default':
            transform_compose = getattr(fgseg, 'Compose')(transform_instance_list)
        else:
            raise ValueError('compose_func must be "default" or None')
        
        return transform_compose
    
    
    @property
    def get_train_pipeline_compose(self):
        return self._compose_transforms(mode='train')
    
    @property
    def get_validate_pipeline_compose(self):
        return self._compose_transforms(mode='validate')
    
    
    

from . import semseg

class Semantic_Segmentation_Transform_Config():
    def __init__(self, 
                 train_pipeline, 
                 validate_pipeline,
                 
                 ) -> None:
        self.train_pipeline = train_pipeline
        self.validate_pipeline = validate_pipeline
        
    
    def _compose_transforms(self, mode=None, compose_func='default'):
        transform_instance_list =[]
        
        if mode == 'train':
            pipeline = self.train_pipeline
        elif mode == 'validate':
            pipeline = self.validate_pipeline
        else:
            raise ValueError('mode must be "train" or "validate"')
        
        for transform_dict in pipeline:
            transform_class = getattr(semseg, transform_dict['transform_type'])
            transform_dict.pop('transform_type')
            
            # whether a dict is empty
            if not transform_dict:
                transform_inst = transform_class()
            else:
                transform_inst = transform_class(**transform_dict)
            transform_instance_list.append(transform_inst)
            
            
        if compose_func is None:
            transform_compose = torch_T.Compose(transform_instance_list)
        elif compose_func == 'default':
            transform_compose = getattr(semseg, 'Compose')(transform_instance_list)
        else:
            raise ValueError('compose_func must be "default" or None')
        
        return transform_compose
    
    
    @property
    def get_train_pipeline_compose(self):
        return self._compose_transforms(mode='train')
    
    @property
    def get_validate_pipeline_compose(self):
        return self._compose_transforms(mode='validate')
    
    
# from . import restore_tr

# class Restoration_Transform_Config():
#     def __init__(self, 
#                  train_pipeline: Sequence[dict], 
#                  validate_pipeline: Sequence[dict],
                 
#                  ) -> None:
#         self.train_pipeline = train_pipeline
#         self.validate_pipeline = validate_pipeline
        
    
    
    
#     def _compose_transforms(self, mode=None, compose_func='default'):
#         transform_instance_list =[]
        
#         if mode == 'train':
#             pipeline = self.train_pipeline
#         elif mode == 'validate':
#             pipeline = self.validate_pipeline
#         else:
#             raise ValueError('mode must be "train" or "validate"')
        
#         for transform_dict in pipeline:
#             transform_class = getattr(restore_tr, transform_dict['transform_type'])
#             transform_dict.pop('transform_type')
            
#             # whether a dict is empty
#             if not transform_dict:
#                 transform_inst = transform_class()
#             else:
#                 transform_inst = transform_class(**transform_dict)
#             transform_instance_list.append(transform_inst)
            
            
#         if compose_func is None:
#             transform_compose = torch_T.Compose(transform_instance_list)
#         elif compose_func == 'default':
#             transform_compose = getattr(restore_tr, 'Compose')(transform_instance_list)
#         else:
#             raise ValueError('compose_func must be "default" or None')
        
#         return transform_compose
            
    
    
#     @property
#     def get_train_pipeline_compose(self):
#         return self._compose_transforms(mode='train')
    
#     @property
#     def get_validate_pipeline_compose(self):
#         return self._compose_transforms(mode='validate')
    
    
    
    
# from . import objdet
# class ObjectDetection_Transform_Config():
#     def __init__(self, 
#                  train_pipeline: Sequence[dict], 
#                  validate_pipeline: Sequence[dict],
                 
#                  ) -> None:
#         self.train_pipeline = train_pipeline
#         self.validate_pipeline = validate_pipeline
        
#     def _compose_transforms(self, mode=None, compose_func='default'):
#         transform_instance_list =[]
        
#         if mode == 'train':
#             pipeline = self.train_pipeline
#         elif mode == 'validate':
#             pipeline = self.validate_pipeline
#         else:
#             raise ValueError('mode must be "train" or "validate"')
        
#         for transform_dict in pipeline:
#             transform_class = getattr(objdet, transform_dict['transform_type'])
#             transform_dict.pop('transform_type')
            
#             # whether a dict is empty
#             if not transform_dict:
#                 transform_inst = transform_class()
#             else:
#                 transform_inst = transform_class(**transform_dict)
#             transform_instance_list.append(transform_inst)
            
            
#         if compose_func is None:
#             transform_compose = torch_T.Compose(transform_instance_list)
#         elif compose_func == 'default':
#             transform_compose = getattr(objdet, 'Compose')(transform_instance_list)
#         else:
#             raise ValueError('compose_func must be "default" or None')
        
#         return transform_compose
    
#     @property
#     def get_train_pipeline_compose(self):
#         return self._compose_transforms(mode='train')
    
#     @property
#     def get_validate_pipeline_compose(self):
#         return self._compose_transforms(mode='validate')
    

# from . import ctr_pred
# class ContourPrediction_Transform_Config():
#     def __init__(self, 
#                  train_pipeline: Sequence[dict], 
#                  validate_pipeline: Sequence[dict],
                 
#                  ) -> None:
#         self.train_pipeline = train_pipeline
#         self.validate_pipeline = validate_pipeline
        
#     def _compose_transforms(self, mode=None, compose_func='default'):
#         transform_instance_list =[]
        
#         if mode == 'train':
#             pipeline = self.train_pipeline
#         elif mode == 'validate':
#             pipeline = self.validate_pipeline
#         else:
#             raise ValueError('mode must be "train" or "validate"')
        
#         for transform_dict in pipeline:
#             transform_class = getattr(ctr_pred, transform_dict['transform_type'])
#             transform_dict.pop('transform_type')
            
#             # whether a dict is empty
#             if not transform_dict:
#                 transform_inst = transform_class()
#             else:
#                 transform_inst = transform_class(**transform_dict)
#             transform_instance_list.append(transform_inst)
            
            
#         if compose_func is None:
#             transform_compose = torch_T.Compose(transform_instance_list)
#         elif compose_func == 'default':
#             transform_compose = getattr(ctr_pred, 'Compose')(transform_instance_list)
#         else:
#             raise ValueError('compose_func must be "default" or None')
        
#         return transform_compose
    
#     @property
#     def get_train_pipeline_compose(self):
#         return self._compose_transforms(mode='train')
    
#     @property
#     def get_validate_pipeline_compose(self):
#         return self._compose_transforms(mode='validate')
    
    
# =============== under construction ========================
    
class BaseTransform_Config():
    def __init__(self, 
                 train_pipeline, 
                 validate_pipeline,
                 
                 ) -> None:
        self.train_pipeline = train_pipeline
        self.validate_pipeline = validate_pipeline
    
    
    
    