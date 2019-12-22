
# Athena

*Athena* is an open-source implementation of end-to-end Automatic Speech Recognition (ASR) engine. Currently this project supports training and decoding of Connectionist Temporal Classification (CTC) based model, transformer-basesd encoder-decoder model and Hybrid CTC/attention based model, and MPC based unsupervised pretraning.

Our vision is to empower both industrial application and academic research on end-to-end models for speech recognition. To make ASR accessible to everyone, we're also releasing some example implementation based on some opensource dataset, like HKSUT, Librispeech

All of our models are implemented in Tensorflow>=2.0.0.


## Table of Contents

- [Athena](#athena)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
    - [Create Manifest](#create-manifest)
  - [Training](#training)
    - [Setting the Configuration File](#setting-the-configuration-file)
    - [Train a Model](#train-a-model)
  - [Results](#results)
  - [Directory Structure](#directory-structure)

## Key Features

- Hybrid CTC/Transformer based end-to-end ASR
- Speech-Transformer
- MPC based unsupervised pretraining

## Installation

This project has only been tested on Python 3. We recommend creating a virtual environment and installing the python requirements there.

```bash
git clone https://github.com/didichuxing/athena.git
pip install -r requirements.txt
python setup.py bdist_wheel sdist
python -m pip install --ignore-installed dist/athena-0.1.0*.whl
source ./tools/env.sh
```

## Data Preparation

### Create Manifest

Athena accepts a textual manifest file as data set interface, which describes speech data set in csv format. In such file, each line contains necessary meta data (e.g. key, audio path, transcription) of a speech audio. For custom data, such manifest file needs to be prepared first. An example is shown as follows:

```csv
wav_filename	wav_length_ms	transcript
/dataset/train-clean-100-wav/374-180298-0000.wav	465004	chapter sixteen i might have told you of the beginning of this liaison in a few lines but i wanted you to see every step by which we came i to agree to whatever marguerite wished
/dataset/train-clean-100-wav/374-180298-0001.wav	514764	marguerite to be unable to live apart from me it was the day after the evening when she came to see me that i sent her manon lescaut from that time seeing that i could not change my mistress's life i changed my own
/dataset/train-clean-100-wav/374-180298-0002.wav	425484	i wished above all not to leave myself time to think over the position i had accepted for in spite of myself it was a great distress to me thus my life generally so calm
/dataset/train-clean-100-wav/374-180298-0003.wav	356044	assumed all at once an appearance of noise and disorder never believe however disinterested the love of a kept woman may be that it will cost one nothing
```

## Training

### Setting the Configuration File

All of our training/ inference configurations are written in config.json. Below is an example configuration file with comments to help you understand.

```json
{
  "batch_size":32,
  "num_epochs":20,
  "sorta_epoch":1,
  "ckpt":"examples/asr/hkust/ckpts/transformer",

  "solver_gpu":[0],
  "solver_config":{
    "clip_norm":100,
    "log_interval":10,
    "enable_tf_function":true
  },

  "model":"speech_transformer",
  "num_classes": null,
  "pretrained_model": null,
  "model_config":{
    "return_encoder_output":false,
    "num_filters":512,
    "d_model":512,
    "num_heads":8,
    "num_encoder_layers":12,
    "num_decoder_layers":6,
    "dff":1280,
    "rate":0.1,
    "label_smoothing_rate":0.0
  },

  "optimizer":"warmup_adam",
  "optimizer_config":{
    "d_model":512,
    "warmup_steps":8000,
    "k":0.5
  },

  "dataset_builder": "speech_recognition_dataset",
  "dataset_config":{
    "audio_config":{
      "type":"Fbank",
      "filterbank_channel_count":40,
      "local_cmvn":false
      },
    "cmvn_file":"examples/asr/hkust/data/cmvn",
    "vocab_file":"examples/asr/hkust/data/vocab",
    "input_length_range":[10, 5000]
  },
  "train_csv":"/tmp-data/dataset/opensource/hkust/train.csv",
  "dev_csv":"/tmp-data/dataset/opensource/hkust/dev.csv",
  "test_csv":"/tmp-data/dataset/opensource/hkust/dev.csv"
}
```

### Train a Model

With all the above preparation done, training becomes straight-forward. `athena/main.py` is the entry point of the training module. Just run `python athena/main.py <your_config_in_json_file>`

Please install Horovod and MPI at first, if you want to train model using multi-gpu. See the [Horovod page](https://github.com/horovod/horovod) for more instructions.

To run on a machine with 4 GPUs with Athena:
`$ horovodrun -np 4 -H localhost:4 python athena/horovod_main.py <your_config_in_json_file>`

To run on 4 machines with 4 GPUs each with Athena:
`$ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python athena/horovod_main.py <your_config_in_json_file>`

## Results

Language  | Model Name | Training Data | Hours of Speech | WER/%
:-----------: | :------------: | :----------: |  -------: | -------:
English  | Transformer | [LibriSpeech Dataset](http://www.openslr.org/12/) | 960 h |
Mandarin | Transformer | HKUST Dataset | 151 h |

## Directory Structure

Below is the basic directory structure for Athena

```bash
|-- Athena
|   |-- data  # - root directory for input-related operations
|   |   |-- datasets  # custom datasets for ASR and pretraining
|   |-- layers  # some layers
|   |-- models  # some models
|   |-- tools # contains various tools, e.g. decoding tools
|   |-- transform # custom featureizer based on C++
|   |   |-- feats
|   |   |   |-- ops # c++ code on tensorflow ops
|   |-- utils # utils, e.g. checkpoit, learning_rate, metric, etc
|-- docs  # docs
|-- examples  # example scripts for ASR, TTS, etc
|   |-- asr  # each subdirectory contains a data preparation scripts and a run script for the task
|       |-- aishell
|       |-- hkust
|       |-- librispeech
|       |-- switchboard_fisher
|-- tools  # need to source env.sh before training
```
