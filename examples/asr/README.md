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
$ python athena/inference.py examples/asr/$dataset_name/configs/$model_name_deocde.json
```
### (7) Computing score with sclite
`bash examples/asr/aishell/local/run_score.sh inference.log score_aishell examples/asr/aishell/data/vocab`

Language |  Task  | Model Name | Training Data | Hours of Speech | Error Rate
:-----------: | :------:  | :------------: | :----------: |  -------: | -------:
English  | ASR | Transformer | [LibriSpeech Dataset](http://www.openslr.org/12/) | 960 h | 3.1% (WER)
English  | ASR | Transformer | [GigaSpeech Dataset] | 10000 h | 11.7% (WER)
Mandarin | ASR | Transformer | HKUST Dataset | 151 h | 21.64 (CER)
Mandarin | ASR | Conformer | HKUST Dataset | 151 h | 21.33% (CER)
Mandarin | ASR | Transformer | [AISHELL Dataset](http://www.openslr.org/33/) | 178 h | 5.13% (CER)
Mandarin | ASR | Conformer | [AISHELL Dataset](http://www.openslr.org/33/) | 178 h | 4.95% (CER)
Mandarin | ASR | Conformer | MISP2021 Challenge Task2 | 120h | 49% (CER)
Mandarin | AV-ASR | Conformer-AV | MISP2021 Challenge Task2 | 120h | 61% (CER)