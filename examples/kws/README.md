# 1. prepare data
prepare the video and audio data and extract feature for training procedure
### 1.1 create data dir to contain misp audio and video feature data
```bash
mkdir -p examples/misp/kws/xtxt/data
```
### 1.2 prepare audio feature dir for audio model; we assume that all the audio have been in the feature format.
```bash
# the dir contains feats.scp and labels.scp
mkdir -p examples/misp/kws/xtxt/train
mkdir -p examples/misp/kws/xtxt/dev
```
### 1.3 prepare audio+video feature dir for audio-video model
```bash
# the dir contains feats.scp , labels.scp and video.scp
mkdir -p examples/misp/kws/xtxt/train_av
mkdir -p examples/misp/kws/xtxt/dev_av

```


# 2. train model
4 kinds of models are offered now:
* conformer/transformer using only audio or video data
* fine tune model for conformer/transformer using focal-loss and label_smoothing
* audio-visual transformer model using both audio and video data by 2 kinds of fusion operation
* Majority Vote by all models

### 2.1 train audio transformer/conformer
1. run the following commands to start training audio transformer/conformer
```python
python athena_wakeup/main.py examples/misp/kws/xtxt/configs/kws_audio_conformer.json
python athena_wakeup/main.py examples/misp/kws/xtxt/configs/kws_audio_transformer.json
```
2. if you have multiple GPUs , you can train models parallel using the following commands
```python
python athena_wakeup/horovod_main.py examples/misp/kws/xtxt/configs/kws_audio_conformer.json
python athena_wakeup/horovod_main.py examples/misp/kws/xtxt/configs/kws_audio_transformer.json
```
3. the model will be stored in ``examples/misp/kws/xtxt/ckpts/kws_audio_conformer`` and ``examples/misp/kws/xtxt/ckpts/kws_audio_transformer``

### 2.2 fine-tune audio transformer using focal-loss
1. focal-loss wii be used to fine tune model to get improvements
```python
python athena_wakeup/main.py examples/misp/kws/xtxt/configs/kws_audio_transformer_finuetune_ft.json
```
### 2.3 train audio-video transformer
1. train model using multi-moda data and the model will be stored in ``examples/misp/kws/xtxt/ckpts/kws_av_transformer``
```python
python athena_wakeup/main.py examples/misp/kws/xtxt/configs/kws_av_transformer.json
```


# 3. test model

### 3.1 test audio transformer/conformer
1. test the trained model and the FRR and FAR will be shown
```python
python athena_wakeup/test_main.py examples/misp/kws/xtxt/configs/kws_audio_conformer.json
python athena_wakeup/test_main.py examples/misp/kws/xtxt/configs/kws_audio_transformer.json
```

### 3.2 test audio-video transformer
1. test the trained model
```python
python athena_wakeup/test_main_av.py examples/misp/kws/xtxt/configs/kws_av_transformer.json
```

# 4. model vote
1. As you have got audio transformer and audio-video transformer, you can use mode vote to get better results

# 5. About MISP Challenge 2021
### 5.1 MISP Challenge 2021 webset:https://mispchallenge.github.io/index.html
### 5.2 Our final score is 0.091 and ranked 3rd among all the 17 teams
### 5.3 the Paper ["AUDIO-VISUAL WAKE WORD SPOTTING SYSTEM FOR MISP CHALLENGE 2021"](https://arxiv.org/abs/2204.08686) have been accepted by ICASSP 2022
