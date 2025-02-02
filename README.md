## Installation 

It is reconmended to install the newest version of Pytorch and MMCV.

### Pytorch

The code requires `python>=3.9`, as well as `pytorch>=2.0.0`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

### MMCV 

Please install MMCV following the instructions [here](https://github.com/open-mmlab/mmcv/tree/master).

### xFormers

Please install xFormers following the instructions [here](https://github.com/facebookresearch/xformers/tree/main).


### Other Dependencies

Please install the following dependencies:

```
pip install -r requirements.txt
```

### Best practice

- Create a conda virtual environment and activate it:

```powershell
conda create -n env_name python=3.11 -y
conda activate env_name
```

- Install PyTorch and torchvision; We recommend using the PyTorch>=2.3.1 with CUDA>=12.1.

```powershell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```

- Install MMCV 

```powershell
pip install -U openmim
mim install mmcv
```

- Install xFormers

```powershell 
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

- Install other dependencies

```powershell
pip3 install -r requirements.txt
```