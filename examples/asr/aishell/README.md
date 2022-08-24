Below we describe the major stages involved to get published results on AISHELL

#### Pre-training[Optional]

For pre-training, we used Masked Predictive Coding objective (https://arxiv.org/pdf/1910.09932.pdf). Similar to the implementation in the paper, 15% of the frames in each sequence are chosen to be masked.
L1 loss is computed between masked input FBANK features and encoder output at corresponding position.
If used, set pretrain option as true,you can use other pre-training methods to replace MPC.

#### Multi-task training

The network structure we used is Transformer and the loss function we used is CTC-Attention multi-task loss.

Parameters for Transformer are: d_model=256, num_heads=4, num_encoder_layers=12, num_decoder_layers=6, dff=2048

For training, we train MTL model for 300 epochs.
Parameters for the optimizer are: d_model=512, warmup_steps=25000, warmup_k=1.0

For decoding we averaged 10 checkpoints that performed best on dev set and conducted CTC-Attention joint decoding with beam size 10, CTC weight 0.5 and LM weight 0.3.

#### Comparsion with published baseline
Model|WER | Num parameters | LM 
-|:-:| :-: | :-: 
[TDNN-hybrid](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf)|7.43 | |
[VGG + BLSTM + CTC-CRF](https://arxiv.org/pdf/1911.08747.pdf)|6.34 | |
[Transformer (ESPNET)](https://arxiv.org/abs/1909.06317)|6.7 |  |
[Transformer (Ours without pre-training)](configs/mtl_transformer_sp_fbank80.json) | 5.22 | |[None](configs/mtl_transformer_sp_fbank80_decode.json)
[Transformer (Ours without pre-training)](configs/mtl_transformer_sp_fbank80.json) | 5.13 | |[transformer](configs/mtl_transformer_sp_fbank80_decode_transformer_lm.json):0.5
[Conformer (Ours without pre-training)](configs/mtl_conformer_sp_fbank80.json) | 5.04 | |[None](configs/mtl_conformer_sp_fbank80_decode.json)
[Conformer (Ours without pre-training)](configs/mtl_conformer_sp_fbank80.json) | 4.95 | |[transformer](configs/mtl_conformer_sp_fbank80_decode_transformer_lm.json):0.2