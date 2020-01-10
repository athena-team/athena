
# Examples for HKUST

```bash
source env.sh
bash examples/asr/hkust/run.sh
```

1) Before you run `examples/asr/hkust/run.sh`, you should download the hkust dataset and store it in `/nfs/project/datasets/opensource_data/hkust`. The script `examples/asr/hkust/local/prepare_data.py` would generate the desired csv file decripting the dataset

2) With the generated csv file, we should compute the cmvn file firstly like this `python athena/cmvn_main.py examples/asr/hkust/mpc.json examples/asr/hkust/data/all.csv`

3) You can perform the unsupervised pretraining using the json file `examples/asr/hkust/mpc.json` or just skip this

4) You can train a transformer model using json file `examples/asr/hkust/transformer.json` or train a mtl_transformer_ctc model using json file `examples/asr/hkust/mtl_transformer.json`

5) You can train a rnnlm model using the transcripts with the json file `examples/asr/hkust/rnnlm.json`, of course, you should firstly prepare the csv file for it

6) Currently, we prepare a simple but not effective way for decoding mtl_transformer_ctc model

By checking the script `examples/asr/hkust/run.sh`, you find out all. Currently, the WER of HKUST dataset is: 

# Examples for Librispeech

TODO: need test
