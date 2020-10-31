# Lifelong Language Knowledge Distillation 🙋‍♂️ → 🏫 → 👨‍🎓

Code for the paper "[Lifelong Language Knowledge Distillation](https://arxiv.org/abs/2010.02123)"  
In The 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020)  
by [Yung-Sung Chuang](https://voidism.github.io/home/), [Shang-Yu Su](https://www.shangyusu.com/), [Yun-Nung Chen](https://www.csie.ntu.edu.tw/~yvchen/index.html)  

Our code is based on the released code from [LAnguage-MOdeling-for-Lifelong-Language-Learning](https://github.com/jojotenya/LAMOL). Most of the settings are identical to theirs.

## 📚 Dataset

| Task | Dataset (Original Data Link) |
| ---- | ------- |
| Summarization | [CNN/DM](https://cs.nyu.edu/~kcho/DMQA/) |
| Goal-Oriented Dialogue | [WOZ](https://github.com/nmrksic/neural-belief-tracker/tree/master/data/woz) |
| Semantic Parsing | [WikiSQL](https://github.com/salesforce/WikiSQL) |
| Natural Language Generation | [E2ENLG](https://github.com/tuetschek/e2e-dataset) |
| Natural Language Generation | [RNNLG](https://github.com/shawnwun/RNNLG) |
| Text Classification | [AGNews, Yelp, Amazon, DBPedia, Yahoo](http://goo.gl/JyCnZq) |

We use the released data from LAMOL's authors [here](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view?usp=sharing), except for E2ENLG and RNNLG datasets.

We also release our processed data in [here](https://www.dropbox.com/s/t51qq9lzz0gtg7m/l2kd_data.zip).

## 💻 Dependencies (same as LAMOL)
- Ubuntu >= 16.04
- This code only supports the following GPUs:
  - NVIDIA Geforce RTX 2080TI 
  - NVIDIA TESLA V100
- python3
- cuda 10.1
- python packages are listed in `requirements.txt`

## 🔧 Setup (same as LAMOL)
1. Create the following two directories in wherever you want. (you can name the directories arbitrarily):
    - `data directory`: Where the dataset will be load by the model.
    - `model directory`: The place for the model to dump its outputs.
2. Download the dataset: Download [here](https://www.dropbox.com/s/t51qq9lzz0gtg7m/l2kd_data.zip) and decompress it. After decompression, move all the files in the decompressed directory into `data directory`.
3. Make a copy of `env.example` and save it as `env`. In `env`, set the value of DATA_DIR as `data directory` and set the value of  MODEL_ROOT_DIR as `model directory`.

## 👨‍🏫 Training and Testing (same as LAMOL)

`train.sh` and `test.sh` are the entrance for training and testing. Main options for them include:

| Options        | Description   |
| -------------  | ------------- |
| seq_train_type | The mode to deal with a sequence of tasks. Mode include: lll\|finetune\|multitask\|mas\|ewc\|gem. "lll" is the default value corresponding our proposed method. The others are the methods for comparing with our proposal. |
| tasks          | A sequence of tasks we want to train by seq_train_type. Leave a space between tasks after the `--tasks` tag. Tasks are the keys in TASK_DICT variable in `settings.py` |
| model_name     | The language model we want to use. The default is `gpt2`. Options include gpt2\|openai-gpt, |
| gen_lm_sample_percentage | This tag only works with `--seq_train_type lll`. The percentage of the size of the dataset will be generated as pseudo samples for our proposed method. |
| lm_lambda      | Lambda value for the loss function. |
| max_n_epochs   | Maximum epoch value for all tasks. |
| min_batch_size | Minimum batch size for all tasks. |
| min_n_steps    | Minimum step for optimizing the model for all tasks. |
| n_train_epochs | Epochs for training for all tasks. |
| n_gpu          | Number of gpu to be used. |
| reg_lambda     | Lambda value for mas and ewc. |
| top_k_lm       | Top k sampling for the language model. |
| top_k_qa       | Top k sampling for the qa model. |
| train_batch_size | Batch size for all tasks. The default is 0. Once the value equals to 0, The batch size will be decided dynamically based on the memory usage of the gpu. |

### 🚨 New Arguments

| Options        | Description   |
| -------------  | ------------- |
| distil | Use `--distil` to conduct Word-KD (the teacher model under `models/gpt2/lll/[TASK]_0.2/` is needed if `[TASK]` is in your LLL tasks.) |
| seq_distil | Use `--seq_distil` to conduct Seq-KD (distilled data need to be put in `data/[TASK]_to_squad-distil-v2.0.json`, which can be found in Supplementary Materials.) |

### Examples

See examples in `run_seqsoftkd-WCS.sh/run_seqsoftkd-NLG.sh/run_seqsoftkd-TC.sh`, which conduct Seq-KD(soft) on all the experiments in our paper.

In the examples, both `--seq_distil` and `--distil` are add to the arguments.

If you want to conduct Word-KD, skip `--seq_distil` in the arguments.

If you want to conduct Seq-KD, skip `--distil` in the arguments.


#### Outputs:

_We add $SEED suffix to the model dir_  
If assigning multitask to `--seq_train_type` tag, the model will be dumped in `$MODEL_ROOT_DIR / model_name / seq_train_type /TASK1_TASK2_...` directory. Otherwise, it will be in `$MODEL_ROOT_DIR / model_name / seq_train_type / TASK1_TASK2_... / TASK1`, `$MODEL_ROOT_DIR / model_name / seq_train_type / TASK1_TASK2_... / TASK2`, ... directories. 

## 📝 Acknowledgements:
- We adapted the open source code of [LAMOL](https://github.com/jojotenya/LAMOL) provided by Cheng-Hao Ho and Fan-Keng Sun.
- We use the language model offered by [transformers](https://github.com/huggingface/transformers), a state-of-the-art natural language processing models library by Thomas Wolf et al.
- The implementation of MAS follows [MAS-Memory-Aware-Synapses](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses), the Memory Aware Synapses method implementation code by Aljundi R. et al.
- The implementation of GEM follows [GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory), the Gradient Episodic Memory method implementation code by Lopez-Paz, David et al.
- The implementation of fp16 (`fp16.py`, `fp16util.py`) is from [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), the ongoing research training transformer language models at scale by NVIDIA.
- Data format conversion refer to [decaNLP](https://github.com/salesforce/decaNLP), the Natural Language Decathlon: Multitask Learning as Question Answering implementation code by Bryan McCann et al.

## 📕 Citation

```
@article{chuang2020lifelong,
  title={Lifelong Language Knowledge Distillation},
  author={Chuang, Yung-Sung and Su, Shang-Yu and Chen, Yun-Nung},
  journal={arXiv preprint arXiv:2010.02123},
  year={2020}
}

@inproceedings{sun2019lamol,
  title={LAMOL: LAnguage MOdeling for Lifelong Language Learning},
  author={Sun, Fan-Keng and Ho, Cheng-Hao and Lee, Hung-Yi},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```
