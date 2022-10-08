# Multilingual-GLM
This repository contains the code of MGLM: a multilingual variant of GLM, a general language model trained with an autoregressive blank infilling objective. 

You may want to check out our [interactive demo](https://models.aminer.cn/mglm-1b/demo/) based on MGLM that generates a brief Chinese/English summary for your article in any commonly used language.

The backbone structure of this model is based on [GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://aclanthology.org/2022.acl-long.26/) (Du et al., ACL 2022) 

Code is mainly based on [THUDM/GLM](https://github.com/THUDM/GLM). Part of the code is also based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [PET](https://github.com/timoschick/pet).

### Parameters
Here we provide a comparison between the sizes of different multilingual language models. 
| Model     | Parameters |
|  ----  | ---- | 
| [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) | 180M | 
| [XLM-R](https://arxiv.org/abs/1911.02116) | 550M |
| [MT5-Large](https://arxiv.org/abs/2010.11934) | 1.2B                   |
| GLM-Large | 1B                 |

## Pretrained Models

Here are the download links to

[Our Pretrained Checkpoint](https://static.aminer.cn/misc/MultiGLM/1B/pretrained.pt)

[Model Configuration File](https://static.aminer.cn/misc/MultiGLM/1B/model_blocklm_multilingual_large.sh)

[Multilingual Tokenizer](https://static.aminer.cn/misc/MultiGLM/1B/mglm250k/mglm250k-uni.model) 

which can be used directly by the original GLM code. 

Here in this repo, the mGLM configuration file and multilingual tokenizer are already included.


## Test Results

### Tasks in XTREME Benchmark
|  Model | XNLI | PAWS-X | XQuAD | MLQA | TyDiQA |
|  ----  | ---- | ---- | ---- | ---- | ---- |
| GLM-Large | 75.6 | 85.2 | 83.6/71.9 | 67.52/54.34 |69.6/55.6 |
| [MT5-Large](https://github.com/google-research/multilingual-t5) | 81.1 | 88.9 | 77.8/61.5 | 71.2/51.7 | 69.9/52.2 |


### Neural Cross Lingual Summarization

The following table contains our test results for the [NCLS](https://aclanthology.org/D19-1302/) English to Chinese(EN2ZHSUM) dataset

Metric is Rouge-1/Rouge-2/Rouge-L

|  Model | NCLS English to Chinese|
|  ----  | ---- | 
| GLM-Large | 50.27/30.94/38.44 | 
| MT5-Large(Reproduced) | 42.31/22.40/31.33 |

## Get Started
<!--
### Docker Image
We prepare two docker images based on CUDA 10.2 and CUDA 11.2. You can pull the pre-built images from Docker Hub and run with docker v19.03+
  ```shell
    docker run --gpus all --rm -it --ipc=host zxdu20/glm-cuda102
  ```
  or replace `glm-cuda102` with `glm-cuda112`.

  You can also modify the image according to your requirements in [docker/cuda102.dockerfile](docker/cuda102.dockerfile) and build the image yourself
  ```shell
    docker build -f cuda102.dockerfile . -t glm-cuda102
  ```
-->
### Manual Installation
Please first install PyTorch 
`pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html  --no-cache-dir`
and [apex](https://github.com/NVIDIA/apex).

Then install other dependencies
`pip3 install -r requirements.txt`


## Usage

### XTREME

- Download the [XTREME](https://sites.research.google/xtreme/) data and check the experiment setup in 
  [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh). Note that `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` 
  need to be changed to your local path. You may also change the `batch-size` and `nproc_per_node` according to your 
  available hardware.

- For Classification tasks, we use the script `scripts/ds_finetune_superglue.sh`.Run the following script to train on the XNLI dataset.
```shell
  bash scripts/ds_finetune_superglue.sh \
     config_tasks/model_blocklm_multilingual_large.sh \
     config_tasks/task_xnli.sh
```

- For QA tasks, we use the script `scripts/ds_finetune_seq2seq.sh`. Run the following script to train on the MLQA dataset.
```shell
  bash scripts/ds_finetune_seq2seq.sh  \
    config_tasks/model_blocklm_multilingual_large.sh  \
    config_tasks/seq_mlqa.sh
```
### Cross-lingual Summary
- Download the [NCLS dataset](https://github.com/ZNLP/NCLS-Corpora)
- For Summarization tasks, we use the script `scripts/ds_finetune_summary.sh`. Run the following to train on NCLS English to Chinese. 
```shell
  bash scripts/ds_finetune_summary.sh  \
    config_tasks/model_blocklm_multilingual_large.sh  \
    config_tasks/seq_ncls.sh
```

### Blank Filling(Interactive)
- Change `CHECKPOINT_PATH` in  `scripts/generate_block.sh` to your local path and run the following script.
```shell
  bash scripts/generate_block.sh  \
    config_tasks/model_blocklm_multilingual_large.sh
```

### Model Parallelism
If your encounter the `CUDA out of memory` error, which means you GPU memory is limited, you can try the model parallelism to divide the parameters into multiple GPUs. Take the two-way model parallelism as an example. First run `change_mp.py` to divide the checkpoint:
```shell
  python3 change_mp.py path_to_the_checkpoint 2
```
Then update the checkpoint path in the model config file (such as [config_tasks/model_blocklm_multilingual_large.sh](config_tasks/model_blocklm_multilingual_large.sh)) and change `MP_SIZE` in the script (such as [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh)) to `2`.

## Pretrain
Run the following script to pre-train the MGLM-Large model
```shell
  bash scripts/ds_pretrain_nvidia.sh config/ds_multi_blockta_large.sh
```

The script [scripts/ds_pretrain_nvidia.sh](scripts/ds_pretrain_nvidia.sh) launches the training program with DeepSpeed. You should change `NUM_WORKERS` and `NUM_GPUS_PER_WORKER` to the number of workers and the number of gpus per worker. Also change `HOST_FILE_PATH` to the path to an OpenMPI-style hostfile. More details about DeepSpeed launcher can be found [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node).

The file [config/ds_multi_blockta_large.sh](config/ds_multi_blockta_large.sh) defines the hyperparameters for pretraining. Most of the arguments are fairly self-explanatory. Specifically, `--train-data` can be multiple keywords defined in `NAMED_CORPORA` in [data_utils/corpora.py](data_utils/corpora.py). The hyperparameters of the optimizer are defined in the corresponding json file under `config`. The semantics of the json file can be found [here](https://www.deepspeed.ai/docs/config-json).

## MT5 Reproduction 
The code for reproducing experiments in MT5 is at `mt5/finetune_mt5.py`. We use a tool called [wandb](https://wandb.ai/site) to track our experiments. After signing up for a new account, use `wandb login --relogin` to login. You can also use `wandb offline` to turn off wandb synchronizing your experiment online.

If you only want to use one GPU to train, use
```shell
  cd mt5
  python3 finetune_mt5.py scisummnet simple
``` 
to train on the [scisummnet dataset](https://cs.stanford.edu/~myasu/projects/scisumm_net/). 

Our distributed training is automated with [Accelerate](https://huggingface.co/docs/accelerate/index). `accelerate config` sets up the configuration for distributed training. `accelerate test` runs a sanity check.
```shell
  cd mt5
  accelerate launch finetune_mt5.py scisummnet simple
``` 
runs the training on the scisummnet dataset.

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
