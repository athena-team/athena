# Athena 101 - Tutorial with TIMIT
We will use ASR task TIMIT as an example to walk you through the whole training process. The recipe for this tutorial can be found at ```examples/asr/timit/run_101.sh```.

## Prepare the data
The data for TIMIT can be found [here](https://www.kaggle.com/mfekadu/darpa-timit-acousticphonetic-continuous-speech) or [here](https://ndownloader.figshare.com/files/10256148). First, we need to download the data and place it at ```examples/asr/timit/data/TIMIT```.
Then we will run the following scripts, which will do some data precessing and generate data csv for train, dev and test set of TIMIT.
```
mkdir -p examples/asr/timit/data
python examples/asr/timit/local/prepare_data.py examples/asr/timit/data/TIMIT examples/asr/timit/data
```
Below is an example csv we generated, it contains the absolute path of input audio, its length, its transcript and its speaker
```csv
wav_filename	wav_length_ms	transcript	speaker
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SI1456.WAV	3065	sil dh iy z eh er er vcl g ae sh vcl b ah vcl b ax sh epi m ey cl k hh ay l ix f ah ng cl sh epi en el th er m el vcl b eh r ix er z sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX286.WAV	3283	sil ih n eh v r ih m ey vcl jh er cl k l ow v er l iy f cl t r ae f ix cl k s ah m cl t ay m z vcl g eh cl s vcl b ae cl t ah cl p sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX196.WAV	1740	sil hh aw vcl d uw ao r sh cl ch er zh epi m ey cl p er l vcl d z sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX106.WAV	2214	sil eh hh y uw vcl jh cl t ae cl p ix sh cl t r ix hh ah ng ix n er hh ah l w ey sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX16.WAV	1926	sil ey r ow l el v w ay er l ey n ih er dh ax w ao l sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SI2086.WAV	2745	sil ae vcl b s el uw sh en f ao r hh ix z l ay hh sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SX376.WAV	2464	sil w ih m ix n m ey n eh v er vcl b ix cl k ah ng cl k ax m cl p l iy cl l iy cl k w el cl t ax m eh n sil	MCLM0
/workspace/athena/examples/asr/timit/data/wav/TRAIN/MCLM0-SI826.WAV	3596	sil k ao sh en cl k en cl t ih n y uw s ix vcl m ih n ax sh cl t r ey sh en ix z epi n aa vcl r eh cl k m eh n d ix f ax l ae cl t ey dx ng cl k aw z sil	MCLM0
```

## Setting the Configuration File

All of our training/ inference configurations are written in config.json. Below is an example configuration file with comments to help you understand.

```jsonc
{
  "batch_size":16,
  "num_epochs":20,
  "sorta_epoch":1,  # keep batches sorted for sorta_epoch, this helps with the convergence of models
  "ckpt":"examples/asr/timit/ckpts/mtl_transformer_ctc_sp/",
  "summary_dir":"examples/asr/timit/ckpts/mtl_transformer_ctc_sp/event",

  "solver_gpu":[0],
  "solver_config":{
    "clip_norm":100,  # clip gradients into a norm of 100
    "log_interval":10,  # print logs for log_interval steps
    "enable_tf_function":true  # enable tf_function to make training faster
  },

  "model":"mtl_transformer_ctc",  # the type of model this training uses, it's a multi-task transformer based model
  "num_classes": null,
  "pretrained_model": null,
  "model_config":{
    "model":"speech_transformer",
    "model_config":{
      "return_encoder_output":true,  # whether to return encoder only or encoder + decoder
      "num_filters":256,  # dimension of cnn filter
      "d_model":256,  # dimension of transformer
      "num_heads":8,  # heads of transformer
      "num_encoder_layers":9,
      "num_decoder_layers":3,
      "dff":1024,  # dimension of feed forward layer
      "rate":0.2,  # dropout rate for transformer
      "label_smoothing_rate":0.0,  # label smoothing rate for output logits
      "schedual_sampling_rate":1.0  # scheduled sampling rate for decoder
    },
    "mtl_weight":0.5 
  },

  "inference_config":{
    "decoder_type":"beam_search_decoder",  # use beam search instead of argmax
    "beam_size":10,
    "ctc_weight":0.0,  # weight for ctc joint decoding
    "model_avg_num":10  # averaging checkpoints gives better results than using single checkpoint with best loss/ metrics
  },

  "optimizer":"warmup_adam",
  "optimizer_config":{  # configs for warmup optimizer
    "d_model":256,
    "warmup_steps":4000,
    "k":1
  },


  "dataset_builder": "speech_recognition_dataset",
  "num_data_threads": 1,
  "trainset_config":{
    "data_csv": "examples/asr/timit/data/train.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},  # config for feature extraction
    "cmvn_file":"examples/asr/timit/data/cmvn",  # mean and variance of FBank
    "text_config": {"type":"eng_vocab", "model":"examples/asr/timit/data/vocab"},  # vocab list
    "speed_permutation": [0.9, 1.0, 1.1],  # use speed perturbation to increase data diversitty
    "input_length_range":[10, 8000]  # range of audio input length
  },
  "devset_config":{
    "data_csv": "examples/asr/timit/data/dev.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/timit/data/cmvn",
    "text_config": {"type":"eng_vocab", "model":"examples/asr/timit/data/vocab"},
    "input_length_range":[10, 8000]
  },
  "testset_config":{
    "data_csv": "examples/asr/timit/data/test.csv",
    "audio_config":{"type":"Fbank", "filterbank_channel_count":40},
    "cmvn_file":"examples/asr/timit/data/cmvn",
    "text_config": {"type":"eng_vocab", "model":"examples/asr/timit/data/vocab"}
  }
}
```

To get state-of-the-art models, we usually need to train for more epochs and use ctc joint decoding with language model. These are omitted for to make this tutorial easier to understand.

## Data normalization
Data normalization is important for the convergence of neural network models. With the generated csv file, we will compute the cmvn file like this 
```
python athena/cmvn_main.py examples/asr/$dataset_name/configs/mpc.json examples/asr/$dataset_name/data/all.csv
```
The generated cmvn files will be found at ```examples/asr/timit/data/cmvn```.

## Train a Model

With all the above preparation done, training becomes straight-forward. `athena/main.py` is the entry point of the training module. Just run:
```
$ python athena/main.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```

Please install Horovod and MPI at first, if you want to train model using multi-gpu. See the [Horovod page](https://github.com/horovod/horovod) for more instructions.

To run on a machine with 4 GPUs with Athena:
```
$ horovodrun -np 4 -H localhost:4 python athena/horovod_main.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```

To run on 4 machines with 4 GPUs each with Athena:
```
$ horovodrun -np 16 -H server1:4,server2:4,server3:4,server4:4 python athena/horovod_main.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```

## Evaluate a model
All of our inference related scripts are merged into inference.py. `athena/inference.py` is the entry point of inference. Just run:
```
python athena/inference.py examples/asr/timit/configs/mtl_transformer_sp_101.json
```
A file named `inference.log` will be generated, which contains the log of decoding. `inference.log` is very important to get correct scoring results, and it will be overwrited if you run `athena/inference.py` multiple times.

## Scoring
For scoring, you will need to install [sclite](https://github.com/usnistgov/SCTK) first. The results of scoring can be found in `score/score_map/inference.log.result.map.sys`. The last few lines will look like this 
```
|================================================================|
| Sum/Avg|  192   7215 | 84.4   11.4    4.3    3.2   18.8   99.5 |
|================================================================|
|  Mean  |  1.0   37.6 | 84.7   11.4    3.9    3.3   18.6   99.5 |
|  S.D.  |  0.0   11.7 |  7.7    6.3    4.2    3.6    9.0    7.2 |
| Median |  1.0   36.0 | 85.0   10.8    2.9    2.8   17.5  100.0 |
|----------------------------------------------------------------|
```
The line with ```Sum/Avg``` is usually what you should be looking for if you just want an overall PER result. In this case, 11.4 is the substitution error, 4.3 is the deletion error, 3.2 is the insertion error and 18.8 is the total PER.
