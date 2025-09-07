<h1 align="center">[TPAMI] Towards Real Zero-Shot Camouflaged Object Segmentation without Camouflaged Annotations</h1>

<div align="center">
  <hr>
  Cheng Lei, &nbsp;
  Jie Fan, &nbsp;
  Xinran Li, &nbsp;
  Tianzhu Xiang, &nbsp;
  Ao Li, &nbsp;
  Ce Zhu, &nbsp;
  Le Zhang, &nbsp;
  <br>
    University of Electronic Science and Technology of China; &nbsp;
    Space42, UAE &nbsp;

  <h4>
    <a href="https://arxiv.org/abs/2410.16953">Paper</a> &nbsp; 
  </h4>
</div>

<blockquote>
<b>Abstract:</b> <i> Camouflaged Object Segmentation (COS) faces significant challenges due to the scarcity of annotated data, where meticulous pixel-level annotation is both labor-intensive and costly, primarily due to the intricate object-background boundaries. Addressing the core question, "Can COS be effectively achieved in a zero-shot manner without manual annotations for any camouflaged object?", we propose an affirmative solution. We examine the learned attention patterns for camouflaged objects and introduce a robust zero-shot COS framework. Our findings reveal that while transformer models for salient object segmentation (SOS) prioritize global features in their attention mechanisms, camouflaged object segmentation exhibits both global and local attention biases. Based on these findings, we design a framework that adapts with the inherent local pattern bias of COS while incorporating global attention patterns and a broad semantic feature space derived from SOS. This enables efficient zero-shot transfer for COS. Specifically, We incorporate a Masked Image Modeling (MIM) based image encoder optimized for Parameter-Efficient Fine-Tuning (PEFT), a Multimodal Large Language Model (M-LLM), and a Multi-scale Fine-grained Alignment (MFA) mechanism. The MIM encoder captures essential local features, while the PEFT module learns global and semantic representations from SOS datasets. To further enhance semantic granularity, we leverage the M-LLM to generate caption embeddings conditioned on visual cues, which are meticulously aligned with multi-scale visual features via MFA. This alignment enables precise interpretation of complex semantic contexts. Moreover, we introduce a learnable codebook to represent the M-LLM during inference, significantly reducing computational demands while maintaining performance. Our framework demonstrates its versatility and efficacy through rigorous experimentation, achieving state-of-the-art performance in zero-shot COS with $F_{\beta}^w$ scores of 72.9\% on CAMO and 71.7\% on COD10K. By removing the M-LLM during inference, we achieve an inference speed comparable to that of traditional end-to-end models, reaching 18.1 FPS. Additionally, our method excels in polyp segmentation, and underwater scene segmentation, outperforming challenging baselines in both zero-shot and supervised settings, thereby implying its potentiality in various segmentation tasks. The source code will be made available at \url{https://github.com/AVC2-UESTC/ZSCOS-CaMF}.</i>
</blockquote>

<!-- <p align="center">
  <img width="1000" src="figs/framework.png">
</p> -->

---


## Install

For setup, refer to the [Quick Start](#quick-start) guide for a fast setup, or follow the detailed instructions below for a step-by-step configuration.

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

---

## Model Weights

### Pretrained Weights

You can download the pretrained weights `eva02_L_pt_m38m_p14to16.pt` from [EVA02](https://github.com/baaivision/EVA/tree/master/EVA-02) or [here](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_m38m_p14to16.pt).

Run the following command to convert the PyTorch weights to the format used in this repository.

```sh
python convert_pt_weights.py 
```

For training, put the converted weights in the `model_weights` folder.



### Fine-tuned Weights

Preparing...

For testing, put the pretrained weights and fine-tuned weights in the `model_weights` folder.



---

## Dataset

The following datasets are used in this paper:
- [DUTS](https://saliencydetection.net/duts/#orgf319326)
- [COD10K](https://github.com/DengPingFan/SINet/)
- [CAMO](https://drive.google.com/drive/folders/1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6)

Preparing...

---

## Quick Start

### Environment Setup

Make sure cuda 11.8 is installed in your virtual environment. Linux is recommmended.

Install pytorch

```sh
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

Install xformers

```sh
pip install xformers==0.0.22 --index-url https://download.pytorch.org/whl/cu118

# test installation (optional)
python -m xformers.info
```

Install mmcv

```sh
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html
```

Other dependencies

```sh
pip install -r requirements.txt
```

### Prepare Dataset

We follow the [ADE20K](https://github.com/CSAILVision/semantic-segmentation-pytorch) dataset format. Organize your dataset files as follows:

```
./datasets/dataset_name/

├── images/
│   ├── training/       # Put training images here
│   └── validation/     # Put validation images here
└── annotations/
    ├── training/       # Put training segmentation maps here 
    └── validation/     # Put validation segmentation maps here 
```

### Test

Put the model weights into the `model_weights` folder, and run the following command to test the model. 

```sh
python test.py

```

### Train

Preparing


### Debug

If you want to debug the code, ckeck `train_debug.py` and `test_debug.py`.





---

## Citation

If you find the code helpful in your research or work, please cite the following paper:

```
@article{lei2024towards,
  title={Towards Real Zero-Shot Camouflaged Object Segmentation without Camouflaged Annotations},
  author={Lei, Cheng and Fan, Jie and Li, Xinran and Xiang, Tianzhu and Li, Ao and Zhu, Ce and Zhang, Le},
  journal={arXiv preprint arXiv:2410.16953},
  year={2024}
}
```


---

## Acknowledgement

This project is based on [MMCV](https://github.com/open-mmlab/mmcv), [timm](https://github.com/huggingface/pytorch-image-models), [EVA02](https://github.com/baaivision/EVA/tree/master/EVA-02), [MAM](https://github.com/jxhe/unify-parameter-efficient-tuning), and [EVP](https://github.com/NiFangBaAGe/Explicit-Visual-Prompt). We thank the authors for their valuable contributions.
