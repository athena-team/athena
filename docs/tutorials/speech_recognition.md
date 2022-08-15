# Speech Recognition
Please refer to [the results table](https://github.com/athena-team/athena#7-results) for supported tasks/examples. To run an ASR example, execute the following commands from your Athena root directory:
```bash
source env.sh
bash examples/asr/$dataset_name/run.sh
```

## Core Stages:

### (1) Data preparation
Before you run `examples/asr/$dataset_name/run.sh`, you should download the coorsponding dataset and store it in `examples/asr/$dataset_name/data`. The script `examples/asr/$dataset_name/local/prepare_data.py` would generate the desired csv file decripting the dataset

### (2) Data normalization
With the generated csv file, we should compute the cmvn file firstly like this
```
$ python athena/cmvn_main.py examples/asr/$dataset_name/configs/mpc.json examples/asr/$dataset_name/data/all.csv
```

### (3) Unsupervised pretraining
You can perform the unsupervised pretraining using the json file `examples/asr/$dataset_name/mpc.json` or just skip this

### (4) Acoustic model training
 You can train a transformer model using json file `examples/asr/$dataset_name/configs/transformer.json` or train a mtl_transformer_ctc model using json file `examples/asr/$dataset_name/configs/mtl_transformer.json`

### (5) Language model training
 You can train a rnnlm model using the transcripts with the json file `examples/asr/$dataset_name/rnnlm.json`, of course, you should firstly prepare the csv file for it

### (6) Decoding
Currently, we provide a simple but not so effective way for decoding mtl_transformer_ctc model. To use it, run
```
$ python athena/decode_main.py examples/asr/$dataset_name/configs/mtl_transformer.json
```

For more detailed results with MPC, please refer to README section of each dataset. Download link of MPC checkpoints can also be found there.
