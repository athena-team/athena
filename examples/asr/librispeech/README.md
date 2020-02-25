Result update:

config: librispeech/configs/mtl_transformer_sp.json

Comparsion with published baseline
test-clean

Model|WER
[Transformer (ESPNET)](https://arxiv.org/abs/1909.06317)| 2.5 
Transformer (Ours without pre-training) | 4.4 
Transformer (Ours with self pre-training) | 



Model|WER | Num parameters
[TDNN-hybrid](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf)|23.7 | -
[LSTM enc + LSTM dec](https://arxiv.org/abs/1806.06342)|29.4 | -
[Transformer (ESPNET)](https://arxiv.org/abs/1909.06317)|23.5 | 28.38M
Transformer (Ours without pre-training) | 23.3 | 57.88M
Transformer (Ours with self pre-training) | 22.98 | 57.88M
