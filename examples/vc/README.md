# Voice Conversion

Currently supported VC task is vcc2018 Speech Copus.

vcc2018 Speech Copus download from:
https://datashare.is.ed.ac.uk/handle/10283/3061?show=full

According to the processing method in StarGAN's original paper, the corpus of four speakers(SF1,SF2,SM1,SM2)
were selected for training and conversion.

To perform the full procedure of VC experiments, simply run:
```bash
source tools/env.sh
bash examples/vc/$task_name/run.sh
```

## Core Stages:

### (1) Data preparation
You can run examples/vc/$task_name/run.sh, it will download the corresponding dataset and store it in examples/vc/$task_name/data.
The script examples/vc/$task_name/local/prepare_data.py would generate the desired csv file.

### (2) Data normalization
With the generated csv file, we should compute the cmvn file firstly like this 
```
$ python athena/cmvn_main.py examples/vc/$task_name/configs/stargan_voice_conversion.json examples/vc/$task_name/data/train.csv
```
We can also directly run the training command like this 
```
$ python athena/main.py examples/vc/$task_name/configs/stargan_voice_conversion.json
```
It will automatically compute the cmvn file before training if the cmvn file does not exist.

### (3) Acoustic model training
You can train a StarGAN model using json file examples/vc/$task_name/configs/stargan_voice_conversion.json

### (4) Conversion
We provide a simple synthesis process using World vocoder. To test your training performance, run
```
$ python athena/synthesize_main.py examples/vc/$task_name/stargan_voice_conversion.json
```
If you want better synthetic voice quality，just try parallel WaveGAN vocoder , which is reference from https://github.com/bigpon/vcc20_baseline_cyclevae


