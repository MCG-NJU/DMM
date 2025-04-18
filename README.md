# DMM: Building a Versatile Image Generation Model via Distillation-Based Model Merging

<div style="text-align: center;">
  <a href="https://arxiv.org/abs/2504.12364"><img src="https://img.shields.io/badge/arXiv-2504.12364-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/MCG-NJU/DMM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue"></a>
  <a href="https://huggingface.co/spaces/MCG-NJU/DMM"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online_Demo-green" alt="arXiv"></a>  
</div>


Official repository of our paper [Building a Versatile Image Generation Model via Distillation-Based Model Merging](https://arxiv.org/abs/2504.12364).


## Introduction

We propose a score distillation based model merging paradigm **DMM**, compressing multiple models into a single versatile T2I model.

![](assets/method.jpg)

## Checkpoints

HuggingFace🤗: https://huggingface.co/MCG-NJU/DMM.

## Usage

Install required packages with:

```bash
pip install -r requirements.txt
```

and initialize an Accelerate environment with:

```bash
accelerate config
```

An example of a training launch is in `train.sh`:

```bash
sh train.sh
```

An example of inference script is in `inference.py`:

```bash
python inference.py
```
## Visualization

![](assets/visualization.jpg)



## TODO
- [x] Pre-training code.
- [x] Model weight release.
- [ ] Incremental training code.
- [x] Inference code with Diffusers.
- [ ] Journeydb dataset code.
- [x] Online demo.
- [ ] ComfyUI plugins.
