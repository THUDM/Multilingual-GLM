# Multilingual-GLM
This repository contains the code of MGLM, the multilingual variant of GLM, a general language model trained with autoregressive blank infilling objective.

The backbone structure of this model is based on [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://aclanthology.org/2022.acl-long.26/) (Du et al., ACL 2022) 


Code is mainly based on [GLM](https://github.com/THUDM/GLM). Part of the code is also based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [PET](https://github.com/timoschick/pet).

## Pretrained Models

[Pretrained Checkpoint](https://static.aminer.cn/misc/MultiGLM/1B/pretrained.pt) (13.9 GB)

[Finetuned Summarizer Checkpoint](https://static.aminer.cn/misc/MultiGLM/1B/summarizer_zh.pt) (2.0 GB)

[DeepSpeed Configuration File](https://static.aminer.cn/misc/MultiGLM/1B/model_blocklm_multilingual_large.sh)

[Tokenizer](https://static.aminer.cn/misc/MultiGLM/1B/mglm250k/mglm250k-uni.model) 

## Test Results

### Tasks in XTREME Benchmark
#### [XNLI](https://aclanthology.org/D18-1269/)

#### [PAWS-X](https://aclanthology.org/D19-1382/)

#### [XQuAD](https://github.com/deepmind/xquad)

#### [MLQA](https://aclanthology.org/2020.acl-main.653/)

#### [TyDiQA](https://aclanthology.org/2020.tacl-1.30/)

### Neural Cross Lingual Summarization

#### [NCLS](https://aclanthology.org/D19-1302/)

## Get Started

### Manual Installation
Please first install PyTorch 
`pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html  --no-cache-dir`
and [apex](https://github.com/NVIDIA/apex).

Then install other dependencies
`pip3 install -r requirements.txt`

## Usage

## Pretrain

## MT5 Reproduction 
The code for reproducing experiments in MT5 `finetune_mt5.py`. We use a tool called [wandb](https://wandb.ai/site) to track our experiments. After signing up for a new account, use `wandb login --relogin` to login. You can also use `wandb offline` to turn off wandb synchronizing your experiment online.

If you only want to use one GPU to train, simply type `python3 finetune_mt5.py scisummnet simple` to train on the [scisummnet dataset](https://cs.stanford.edu/~myasu/projects/scisumm_net/). Our distributed training is automated with [Accelerate](https://huggingface.co/docs/accelerate/index). `accelerate config` sets up the configuration. `accelerate test` runs a sanity check. `accelerate launch finetune_mt5.py scisummnet simple` runs the training on the scisummnet dataset.

## Citation 
Citation for the GLM paper： 
```
@inproceedings{du-etal-2022-glm,
    title = "{GLM}: General Language Model Pretraining with Autoregressive Blank Infilling",
    author = "Du, Zhengxiao  and
      Qian, Yujie  and
      Liu, Xiao  and
      Ding, Ming  and
      Qiu, Jiezhong  and
      Yang, Zhilin  and
      Tang, Jie",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.26",
    doi = "10.18653/v1/2022.acl-long.26",
    pages = "320--335",
    abstract = "There have been various types of pretraining architectures including autoencoding models (e.g., BERT), autoregressive models (e.g., GPT), and encoder-decoder models (e.g., T5). However, none of the pretraining frameworks performs the best for all tasks of three main categories including natural language understanding (NLU), unconditional generation, and conditional generation. We propose a General Language Model (GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding 2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over BERT and T5 on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of blanks. On a wide range of tasks across NLU, conditional and unconditional generation, GLM outperforms BERT, T5, and GPT given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25{\mbox{$\times$}} parameters of BERT Large , demonstrating its generalizability to different downstream tasks.",
}
```

Citation for the Multilingual GLM paper to be released
