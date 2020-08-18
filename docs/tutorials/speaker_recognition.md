# Speaker Recognition

Speaker Recognition (SR) can be categorized as two tasks: speaker identification (closed-set) and speaker verification (open set). The former classifies a speaker to a specific identity, while the latter determines whether a pair of utterances belongs to the same person. [[1]](#1).

Currently *Athena* supports closed-set speaker recognition, i.e. speaker identification. The baseline system is built on ResNet-34 [[2]](#2). An example scipt for Voxceleb1 dataset and detailed results are ready at examples/speaker_identification/voxceleb. An example script for speaker verification on the Voxceleb dataset will be released soon.

To run experiments:
```bash
source tools/env.sh
bash examples/speaker_identification/voxceleb/run.sh
```

## Core Stages:

### (1) Data preparation
For data preparation, please run the script `examples/speaker_identification/voxceleb/local/prepare_data.py`. It downloads the voxceleb datasets, process the data (e.g. convert aac files to wav files, make speaker labels), and generates csv files. 

### (2) Data normalization
With the generated csv files, we should firstly compute the cmvn file for data normalization:
```
$ python athena/cmvn_main.py \
        examples/speaker_identification/voxceleb/configs/speaker_resnet.json \
        examples/speaker_identification/voxceleb/data/vox1_iden_train.csv
```
Or, it is okay to start training directly, the cmvn computation step will be done in the `athena/main.py`:
```
$ python athena/main.py examples/speaker_identification/voxceleb/configs/speaker_resnet.json
```
It will automatically compute the cmvn file before training if the cmvn file does not exist.

### (3) Model training
You can train the model using json file:
```
$ python athena/main.py examples/speaker_identification/voxceleb/configs/speaker_resnet.json
```

### (4) Inference
To test your model's performance, run:
```
$ python athena/inference_main.py examples/speaker_identification/voxceleb/configs/speaker_resnet.json
```

#### References
<a id="1">[1]</a> Exploring the Encoding Layer and Loss Function in End-to-End Speaker and Language Recognition System<br>
<a id="2">[2]</a> Deep Residual Learning for Image Recognition<br>