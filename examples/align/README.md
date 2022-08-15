# Speech Alignment
Before runing an Align example, you should refer to the [ASR recipe](https://github.com/athena-team/athena/blob/master/examples/asr/README.md) to get the trained ASR model first:
```bash
source env.sh
bash examples/align/$dataset_name/run.sh
```

## Core Stages(stage1-3 is ASR stages, stage4 is Align stage):

### (1) Data preparation
Before you run `examples/asr/$dataset_name/run.sh`, you should download the coorsponding dataset and store it in `examples/asr/$dataset_name/data`. The script `examples/asr/$dataset_name/local/prepare_data.py` would generate the desired csv file decripting the dataset

### (2) Data normalization
With the generated csv file, we should compute the cmvn file firstly like this
```
$ python athena/cmvn_main.py examples/asr/$dataset_name/configs/mpc.json examples/asr/$dataset_name/data/all.csv
```
### (3) Acoustic model training
 You can train a transformer model using json file `examples/asr/$dataset_name/configs/transformer.json` or train a mtl_transformer_ctc model using json file `examples/asr/$dataset_name/configs/mtl_transformer.json`

### (4) CTC alignment
After trained a transformer model, we provide the CTC alignment method for aligning the speech and label on frame level. To use it, run
```
$ python athena/alignment.py examples/align/$dataset_name/configs/mtl_transformer.align.json
```
### (5) Alignment result
The CTC alignment result of one utterance is shown below, we can see the output of ctc alignment is with time delayed
<div align="left"><img src="ctc_alignment_demo.png" width="550"/></div>
