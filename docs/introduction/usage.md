# Usage

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
  "num_data_threads": 1,
  "trainset_config":{
    "data_csv": "examples/asr/hkust/data/train.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/hkust/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/hkust/data/vocab"},
    "input_length_range":[10, 8000]
  },
  "devset_config":{
    "data_csv": "examples/asr/hkust/data/dev.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/hkust/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/hkust/data/vocab"},
    "input_length_range":[10, 8000]
  },
  "testset_config":{
    "data_csv": "examples/asr/hkust/data/dev.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/hkust/data/cmvn",
    "text_config": {"type":"vocab", "model":"examples/asr/hkust/data/vocab"}
  }
}
```

### Train a Model

With all the above preparation done, training becomes straight-forward. `athena/main.py` is the entry point of the training module. Just run:
```
$ python athena/main.py <your_config_in_json_file>
````

Please install Horovod and MPI at first, if you want to train model using multi-gpu. See the [Horovod page](https://github.com/horovod/horovod) for more instructions.

To run on a machine with 4 GPUs with Athena:
```
$ horovodrun -np 4 -H localhost:4 python athena/horovod_main.py <your_config_in_json_file>
```

To run on 4 machines with 4 GPUs each with Athena:
```
$ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python athena/horovod_main.py <your_config_in_json_file>
```

## Deployment
After training, you can deploy the model on servers using the TensorFlow C++ API. Below are some steps to achieve this functionality with an ASR model. 

1. Install all dependencies, including TensorFlow, Protobuf, absl, Eigen3 and kenlm (optional).
2. Freeze the model to pb format with `athena/deploy_main.py`.
3. Compile the C++ codes.
4. Load the model and do argmax decoding in C++ codes, see `deploy/src/argmax.cpp` for the entry point.

After compiling, an executable file will be generated and you can run the executable file:
```
$ ./argmax
```

Detailed implementation is described [here](../tutorials/deployment.md).
