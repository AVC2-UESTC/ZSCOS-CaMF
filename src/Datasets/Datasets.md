## BaseDataset.py

### Foreground_Dataset_Config

**Format in Config file**

```json
{
    "Dataset_cfg": {
        "dataset_cfg_name": "Foreground_Dataset_Config",
        "dataset_cfg_args": {
            "data_root": "{The path of the specific dataset}",
            "transform_prefix": "{The prefix of transform functions}",
            "train_pipeline": "{The Transform functions for training}",
            "test_pipeline": "{The Transform functions for validation}",
            "truncate_ratio": "{If specified, the dataset will be truncated at the given ratio. Usually used in debugging}"
        }
    }
}


```


Here is an example of Foreground_Dataset_Config
```json
{
    "Dataset_cfg": {
        "dataset_cfg_name": "Foreground_Dataset_Config", 
        "dataset_cfg_args": {
            "data_root": "./dataset/CAMO",
            "transform_prefix": "LT",

            "train_pipeline": {
                "ToTensor": null,
                "RandomResize": {
                    "scale": [512, 352],
                    "ratio_range": [0.5, 2.0],
                    "keep_ratio": true, 
                    "resize_mask": true
                }, 

                "RandomCrop": {
                    "size": 352
                },
                "RandomHorizontalFlip": {
                    "prob": 0.5
                },
                "Normalize": {
                    "mean": [0.485, 0.456, 0.406], 
                    "std": [0.229, 0.224, 0.225]
                }
            },

            "test_pipeline": {
                "ToTensor": null,

                "Resize": {
                    "scale": [512, 352],
                    "keep_ratio": true,
                    "resize_mask": true
                },
                "CenterCrop": {
                    "size": 352
                },
                "Normalize": {
                    "mean": [0.485, 0.456, 0.406], 
                    "std": [0.229, 0.224, 0.225]
                }
            },
            "truncate_ratio": null
        }
    }
}

```



## local_transform.py


You could see more about transform functions in [local_transforms.py](./local_transforms.py)





























