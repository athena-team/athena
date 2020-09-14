# What is Athena?

*Athena* is an open-source implementation of end-to-end speech processing engine. Our vision is to empower both industrial application and academic research on end-to-end models for speech processing. To make speech processing available to everyone, we're also releasing example implementation and recipe on some opensource dataset for various tasks (ASR, TTS, Voice Conversion, Speaker Recognition, etc).

All of our models are implemented in Tensorflow>=2.0.0. For ease of use, we provide Kaldi-free pythonic feature extractor with [Athena_transform](https://github.com/athena-team/athena-transform).

## Key Features

- Hybrid CTC/Transformer based end-to-end ASR
- Speech-Transformer
- Unsupervised pre-training
- Multi-GPU training on one machine or across multiple machines with Horovod
- End-to-end Tacotron2 based TTS with support for multi-speaker and GST
- Transformer based TTS and FastSpeech
- WFST creation and WFST-based decoding
- Deployment with Tensorflow C++

## Directory Structure

Below is the basic directory structure for Athena

```bash
|-- Athena
|   |-- data  # - root directory for input-related operations
|   |   |-- datasets  # custom datasets for ASR, TTS and pre-training
|   |-- layers  # some layers
|   |-- models  # some models
|   |-- tools # contains various tools, e.g. decoding tools
|   |-- transform # custom featureizer based on C++
|   |   |-- feats
|   |   |   |-- ops # c++ code on tensorflow ops
|   |-- utils # utils, e.g. checkpoit, learning_rate, metric, etc
|-- deploy  # deployment with Tensorflow C++
|   |-- include
|   |-- src
|-- docker
|-- docs  # docs
|-- examples  # example scripts for ASR, TTS, etc
|   |-- asr  # each subdirectory contains a data preparation scripts and a run script for the task
|   |   |-- aishell
|   |   |-- hkust
|   |   |-- librispeech
|   |   |-- switchboard
|   |   |-- timit
|   |-- translate # examples for translate
|   |   |-- spa-eng-example
|   |-- tts # examples for tts
|   |   |-- data_baker
|   |   |-- libritts
|   |   |-- ljspeech
|-- tools  # need to source env.sh before training
```
