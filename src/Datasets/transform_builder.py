
from . import transforms


def build_transform(transform_cfg_name: str, transform_cfg_args: dict):
    supported_transforms = [item for item in dir(transforms) if not (item.startswith("__") and item.endswith("__"))]
    assert transform_cfg_name in supported_transforms, f'Unsupported transform: {transform_cfg_name}, supported transform configs are: {supported_transforms}. Please add your own transform config class in "src/Datasets/transforms/__init__.py".'
    
    assert transform_cfg_args is not None, 'transform_cfg_args cannot be None.'
    
    transform_cfg_class = getattr(transforms, transform_cfg_name)
    transform_cfg_instance = transform_cfg_class(**transform_cfg_args)
    
    return transform_cfg_instance

 





