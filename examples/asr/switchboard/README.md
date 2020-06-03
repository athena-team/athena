
Below we describe experimental details and results involved to get results.

#### Experimental Details

The training config we used is mtl_transformer_sp.json. The batch size was set to 16 beacause that switchboard consists of many long utterances. We got following results by training using two machines, each with four GPUs. In all experiments, per-speaker cmvn was used.

#### Results
Model                                              |   WER 
|:-| :- |
Transformer (Ours without pre-training)            |   8.6%
Transformer (Ours pre-trained on Librispeech 960h) |   8.2%
