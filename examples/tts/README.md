# Speech Synthesis

Currently supported TTS tasks are LJSpeech and Chinese Standard Mandarin Speech Copus(data_baker). Supported models are shown in the table below:
(Note:HiFiGAN is trained based on [TensorflowTTS](https://github.com/TensorSpeech/TensorFlowTTS))

Traing Data | Acoustic Model | Vocoder|  Audio Demo
:---------: |:-------------: | :-------------:| :------------:
data_baker  |Tacotron2       | GL             |  [audio_demo](data_baker/audio_demo/)
data_baker  |Transformer_tts | GL             |  [audio_demo](data_baker/audio_demo/)
data_baker  |Fastspeech      | GL             |  [audio_demo](data_baker/audio_demo/)
data_baker  |Fastspeech2     | GL             |  [audio_demo](data_baker/audio_demo/)
data_baker  |Fastspeech2     | HiFiGAN        |  [audio_demo](data_baker/audio_demo/)
ljspeech    |Tacotron2       | GL             |  [audio_demo](ljspeech/audio_demo/)

To perform the full procedure of TTS experiments, simply run:
```bash
source tools/env.sh
bash examples/tts/$task_name/run.sh
```

## Core Stages:
Take Tacotron2 as an example. 

####Note: You should install ```unrar``` at first.
```shell
apt-get update
apt-get install unrar
```
### (1) Data preparation

You can run `examples/tts/$task_name/run.sh`, it will download the corresponding dataset and store it in `examples/tts/$task_name/data`. The script `examples/tts/$task_name/local/prepare_data.py` would generate the desired csv file decripting the dataset



### (2) Data normalization
With the generated csv file, we should compute the cmvn file firstly like this 
```
$ python athena/cmvn_main.py examples/tts/$task_name/configs/t2.json examples/tts/$task_name/data/train.csv
```
We can also directly run the training command like this 
```
$ python athena/main.py examples/tts/$task_name/configs/t2.json
```
It will automatically compute the cmvn file before training if the cmvn file does not exist.
### (3) Training
You can train a Tacotron2 model using json file `examples/tts/$task_name/configs/t2.json`

### (4) Synthesis
Currently, we provide a simple synthesis process using GriffinLim vocoder. To test your training performance, run
```
$ python athena/inference.py examples/tts/$task_name/t2.json
```

### (5) SavedModel
You can also save your ckpt as a TensorFlow SavedModel:
```
$ python athena/freeze.py examples/tts/$task_name/t2.json your_saved_model_dir
```

