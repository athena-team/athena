Below we describe in detail the stages involved to get published results on AISHELL

#### Pre-training

For pre-training, we used Masked Predictive Coding objective (https://arxiv.org/pdf/1910.09932.pdf). Similar to the implementation in the paper, 15% of the frames in each sequence are chosen to be masked.
 L1 loss is computed between masked input FBANK features and encoder output at corresponding position.

#### Fine-tuning

The network structure we used is Transformer and the loss function we used is CTC-Attention multi-task loss.
Parameters for Transformer are: d_model=512, num_heads=8, num_encoder_layers=12, num_decoder_layers=6, dff=1280
For training, we restore from pre-trained MPC model and train MTL model for 50 epochs.
Parameters for the optimizer are: d_model=512, warmup_steps=25000, warmup_k=1.0
For decoding we averaged 10 checkpoints that performed best on dev set and conducted CTC-Attention joint decoding with beam size 10, CTC weight 0.5 and LM weight 0.3.

#### Comparsion with published baseline
Model|WER | Num parameters
-|:-:| :-: |
[TDNN-hybrid](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf)|7.43 | -
[VGG + BLSTM + CTC-CRF](https://arxiv.org/pdf/1911.08747.pdf)|6.34 | -
[Transformer (ESPNET)](https://arxiv.org/abs/1909.06317)|6.7 | 28.38M
Transformer (Ours without pre-training) |  | 57.88M
Transformer (Ours with self pre-training) |  | 57.88M
