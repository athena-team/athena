# Voice Activity Detection
To run an VAD example, execute the following commands from your Athena root directory:
```bash
source env.sh
bash examples/vad/$dataset_name/run.sh
```
## Core Stages:

### (1) Data download
First, the coorsponding dataset should be downloaded and stored in `examples/vad/$dataset_name/data/resource`. In google_dataset_v2 example, both speech commands in Google Speech Commands Dataset V2 dataset and noises in openslr-28 database are used.

### (2) Data preparation
By using `examples/vad/$dataset_name/local/prepare_data.sh`, the dataset would be divided into three parts separately as training set, validation set and testing set. After
 that, the features of all of them will be extracted and stored in kaldiio format.

### (3) Model training
Two kinds of VAD models, DNN and MarbleNet, are supported in Athena. You can using json file `examples/vad/$dataset_name/configs/vad_dnn.json` and `examples/vad/$dataset_name/vad_marbleNet.json` to performe the training of them respectively. Both of DNN and MarbleNet are frame-level classifiers. The prediction resulsts is generated with overlapping input segments, whose length is specified by `left_context` and `right_context` in json files.

### (4) Inferencing
Currently, you can performe frame-level inferencing. To do it, run
```
$ python athena/inference.py examples/vad/google_dataset_v2/configs/vad_$model.json
```

Task | Model Name |      Training Data      | Input Segment | Frame Error Rate 
:-----------: | :------: | :------------: | :-----: | :----------:
VAD  | DNN | Google Speech Commands Dataset V2 | 0.21s | 8.49% 
VAD  | MarbleNet | Google Speech Commands Dataset V2 | 0.63s | 2.50%
