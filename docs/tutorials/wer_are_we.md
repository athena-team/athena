
# WER are we?

WER are we? An attempt at tracking states-of-the-art results and their corresponding codes on speech recognition.  *Feel free to correct!*
(Inspired by [wer_are_we](https://github.com/syhw/wer_are_we))

## HKUST

(Possibly trained on more data than HKUST.)

| CER Test     | Paper          | Published      | Notes     | Codes   |
| :----------- | :------------- | :------------- | :-------: | :-----: |
| 21.2% | [Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932) | October 2019 | Transformer-CTC MTL + RNN-LM + speed perturbation + MPC Pre-training on 10,000 hours unlabeled speech | [athena-team/Athena](https://github.com/athena-team/athena/tree/master/examples/asr/hkust) |
| 22.75% | [Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932) | October 2019 | Transformer-CTC MTL + RNN-LM + speed perturbation + MPC Self data Pre-training | [athena-team/Athena](https://github.com/athena-team/athena/tree/master/examples/asr/hkust) |
| 23.09% | [CIF: Continuous Integrate-and-Fire for End-to-End Speech Recognition](https://arxiv.org/abs/1905.11235) | February 2020 | CIF + SAN-based models (AM + LM) + speed perturbation + SpecAugment | None |
| 23.5% | [A Comparative Study on Transformer vs RNN in Speech Applications](https://arxiv.org/abs/1909.06317) | September 2019 | Transformer-CTC MTL + RNN-LM + speed perturbation | [espnet/espnet](https://github.com/espnet/espnet/blob/master/egs/hkust/asr1) |
| 23.67% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | 2016 | TDNN/HMM, lattice-free MMI + speed perturbation| [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/hkust/s5) |
| 24.12% | [Self-Attention Aligner: A Latency-Control End-to-End Model for ASR Using Self-Attention Network and Chunk-Hopping](https://arxiv.org/pdf/1902.06450.pdf) | February 2019 | SAA Model + SAN-LM (joint training) + speed perturbation | None |
| 27.67% | [Extending Recurrent Neural Aligner for Streaming End-to-End Speech Recognition in Mandarin](https://arxiv.org/abs/1806.06342) | Feberary 2019 | Extended-RNA + RNN-LM (joint training) | None |
| 28.0%  | [Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM](https://arxiv.org/pdf/1706.02737.pdf) | June 2017 | CTC-Attention MTL + joint decoding ( one-pass) + VGG Net + RNN-LM (seperate) + speed perturbation | [espnet/espnet](https://github.com/espnet/espnet/blob/master/egs/hkust/asr1) |
| 29.9% | [Joint CTC/attention decoding for end-to-end speech recognition](https://www.aclweb.org/anthology/P17-1048.pdf)| 2017 | CTC-Attention MTL-large + joint decoding (one pass) + speed perturbation | [espnet/espnet](https://github.com/espnet/espnet/blob/master/egs/hkust/asr1) |

## AISHELL-1

| CER Dev | CER Test | Paper  | Published | Notes  | Codes
| :------ | :------- | :----- | :-------- | :-----:| :---:
| None | 6.6% | [Improving Transformer-based Speech Recognition Using Unsupervised Pre-training](https://arxiv.org/abs/1910.09932) | October 2019 | Transformer-CTC MTL + RNN-LM + speed perturbation + MPC Self data Pre-training | [athena-team/Athena](https://github.com/athena-team/athena/tree/master/examples/asr/aishell) |
| None | 6.34% | [CAT: CRF-Based ASR Toolkit](https://arxiv.org/pdf/1911.08747.pdf)| November 2019 | VGG + BLSTM + CTC-CRF + 3-gram LM + speed perturbation| [thu-spmi/CAT](https://github.com/thu-spmi/CAT/tree/master/egs/aishell) |
| 6.0% | 6.7% | [A Comparative Study on Transformer vs RNN in Speech Applications](https://arxiv.org/abs/1909.06317) | September 2019 | Transformer-CTC MTL + RNN-LM + speed perturbation | [espnet/espnet](https://github.com/espnet/espnet/blob/master/egs/aishell/asr1) |
| None | 7.43% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | 2016 | TDNN/HMM, lattice-free MMI + speed perturbation| [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/aishell/s5) |

## THCHS-30

| CER Word Task<br>0db white / car / cafeteria | PER Phone Task<br>0db white / car / cafeteria | Paper | Published | Notes | Codes
| :---------------------------------------- | :----------------------------------------- | :---- | :-------- | :---: | :---: 
| 75.01% / 32.13% / 56.37% | 46.95% / 15.96% / 32.56% | [THCHS-30: A Free Chinese Speech Corpus](https://arxiv.org/pdf/1512.01882.pdf) | December 2015 | DNN + DAE-based noise cancellation | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/thchs30/s5) |
| 65.87% / 25.07% / 51.92% | 39.80% / 11.48% / 30.55%| None | None | DNN + DAE-based noise cancellation | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/thchs30/s5) |

## LibriSpeech

(Possibly trained on more data than LibriSpeech.)

| WER test-clean | WER test-other | Paper          | Published | Notes   | Codes |
| :------------- | :------------- | :------------- | :-------- | :-----: | :----:
| 5.83% | 12.69% | *Humans* [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | December 2015 | *Humans* | None |
| 2.0%  | 4.1%   | [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460) | November 2019 | Conv+Transformer AM (10k word pieces) with ConvLM decoding and Transformer rescoring + 60k hours unlabeled | [facebookresearch/wav2letter](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/sota/2019/librispeech)
| 2.3% | 4.9%  | [Transformer-based Acoustic Modeling for Hybrid Speech Recognition](https://arxiv.org/abs/1910.09799)  | October 2019 | Transformer AM (chenones) + 4-gram LM + Neural LM rescore (data augmentation:Speed perturbation and SpecAugment) | None |
| 2.3%  | 5.0%   | [RWTH ASR Systems for LibriSpeech: Hybrid vs Attention](https://arxiv.org/abs/1905.03072) | September 2019, Interspeech | HMM-DNN + lattice-based sMBR + LSTM LM + Transformer LM rescoring (no data augmentation) | [rwth-i6/returnn](https://github.com/rwth-i6/returnn)<br>[rwth-i6/returnn-experiments](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-librispeech-system) |
| 2.3%  | 5.2%   | [End-to-end ASR: from Supervised to Semi-Supervised Learning with Modern Architectures](https://arxiv.org/abs/1911.08460) | November 2019 | Conv+Transformer AM (10k word pieces) with ConvLM decoding and Transformer rescoring | [facebookresearch/wav2letter](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/sota/2019/librispeech) |
| 2.2%  | 5.8%   | [State-of-the-Art Speech Recognition Using Multi-Stream Self-Attention With Dilated 1D Convolutions](https://arxiv.org/abs/1910.00716) | October 2019 | Multi-stream self-attention in hybrid ASR + 4-gram LM + Neural LM rescore (no data augmentation) | [s-omranpour/<br>ConvolutionalSpeechRecognition](https://github.com/s-omranpour/ConvolutionalSpeechRecognition) <br> (not official) |
| 2.5%  | 5.8%   | [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) | April 2019 | Listen Attend Spell | [DemisEom/SpecAugment](https://github.com/DemisEom/SpecAugment) <br> (not official) |
| 3.2%  | 7.6%  | [From Senones to Chenones: Tied Context-Dependent Graphemes for Hybrid Speech Recognition](https://arxiv.org/abs/1910.01493) | October 2019 | LC-BLSTM AM (chenones) + 4-gram LM (data augmentation:Speed perturbation and SpecAugment) | None | 
| 3.19% | 7.64% | [The CAPIO 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1801.00059) | April 2018 | TDNN + TDNN-LSTM + CNN-bLSTM + Dense TDNN-LSTM across two kinds of trees + N-gram LM + Neural LM rescore | None | 
| 2.44%  |  8.29%   | [Improved Vocal Tract Length Perturbation for a State-of-the-Art End-to-End Speech Recognition System](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/3227.html) | September 2019, Interspeech | encoder-attention-decoder + Transformer LM | None |
| 3.80% | 8.76%  | [Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks](http://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf) | Interspeech, Sept 2018 | 17-layer TDNN-F + iVectors | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/local/chain/tuning/run_tdnn_1d.sh)
| 2.8%  | 9.3%   | [RWTH ASR Systems for LibriSpeech: Hybrid vs Attention](https://arxiv.org/abs/1905.03072) | September 2019, Interspeech | encoder-attention-decoder + BPE + Transformer LM (no data augmentation) | [rwth-i6/returnn](https://github.com/rwth-i6/returnn)<br>[rwth-i6/returnn-experiments](https://github.com/rwth-i6/returnn-experiments/tree/master/2019-librispeech-system) |
| 3.26% | 10.47% | [Fully Convolutional Speech Recognition](https://arxiv.org/abs/1812.06864) | December 2018 | End-to-end CNN on the waveform + conv LM | None |
| 3.82% | 12.76% | [Improved training of end-to-end attention models for speech recognition](https://www-i6.informatik.rwth-aachen.de/publications/download/1068/Zeyer--2018.pdf) | Interspeech, Sept 2018 | encoder-attention-decoder end-to-end model | [rwth-i6/returnn](https://github.com/rwth-i6/returnn)<br>[rwth-i6/returnn-experiments](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention) |
| 4.28% | | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-TDNN trained with MMI + data augmentation (speed) + iVectors + 3 regularizations | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/librispeech/s5) | 
| 4.83% | | [A time delay neural network architecture for efficient modeling of long temporal contexts](http://speak.clsp.jhu.edu/uploads/publications/papers/1048_pdf.pdf) | 2015 | HMM-TDNN + iVectors | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/librispeech/s5) | 
| 5.15% | 12.73% | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://proceedings.mlr.press/v48/amodei16.pdf) | December 2015 | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 100M parameters trained on 11940h | [PaddlePaddle/DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)
| 5.51% | 13.97% | [LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | 2015 | HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/librispeech/s5) |
| 4.8%  | 14.5% | [Letter-Based Speech Recognition with Gated ConvNets](https://arxiv.org/abs/1712.09444) | December 2017 | (Gated) ConvNet for AM going to letters + 4-gram LM | None
| 8.01% | 22.49% | same, [Kaldi](http://kaldi-asr.org/) | 2015 | HMM-(SAT)GMM | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/librispeech/s5) |
| | 12.51% | [Audio Augmentation for Speech Recognition](http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) | 2015 | TDNN + pNorm + speed up/down speech | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/librispeech/s5) |

## WSJ

(Possibly trained on more data than WSJ.)

| WER eval'92    | WER eval'93    | Paper          | Published | Notes   | Codes | 
| :------------- | :------------- | :------------- | :-------- | :-----: | :---: |
| 5.03% | 8.08% | *Humans* [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://arxiv.org/abs/1512.02595v1) | December 2015 | *Humans* | None |
| 2.9%  | None  | [End-to-end Speech Recognition Using Lattice-Free MMI](https://pdfs.semanticscholar.org/dcae/b29ad3307e2bdab2218416c81cb0c4e548b2.pdf) | September 2018 | HMM-DNN LF-MMI trained (biphone) | None |
| 3.10% | None  | [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](http://proceedings.mlr.press/v48/amodei16.pdf) | December 2015 | 9-layer model w/ 2 layers of 2D-invariant convolution & 7 recurrent layers, w/ 100M parameters | [PaddlePaddle/DeepSpeech](https://github.com/PaddlePaddle/DeepSpeech)
| 3.47% | None  | [Deep Recurrent Neural Networks for Acoustic Modelling](http://arxiv.org/pdf/1504.01482v1.pdf) | April 2015 | TC-DNN-BLSTM-DNN | None |
| 3.5%  | 6.8%  | [Fully Convolutional Speech Recognition](https://arxiv.org/abs/1812.06864) | December 2018 | End-to-end CNN on the waveform + conv LM | None |
| 3.63% | 5.66% | [LibriSpeech: an ASR Corpus Based on Public Domain Audio Books](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf) | 2015 | test-set on open vocabulary (i.e. harder), model = HMM-DNN + pNorm[*](http://www.danielpovey.com/files/2014_icassp_dnn.pdf) | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/wsj/s5)
| 4.1%  | None  | [End-to-end Speech Recognition Using Lattice-Free MMI](https://pdfs.semanticscholar.org/dcae/b29ad3307e2bdab2218416c81cb0c4e548b2.pdf) | September 2018 | HMM-DNN E2E LF-MMI trained (word n-gram) | None |
| 5.6%  | None  | [Convolutional Neural Networks-based Continuous Speech Recognition using Raw Speech Signal](http://infoscience.epfl.ch/record/203464/files/Palaz_Idiap-RR-18-2014.pdf) | 2014 | CNN over RAW speech (wav) | None |
| 5.7%  | 8.7%  | [End-to-end Speech Recognition from the Raw Waveform](https://arxiv.org/abs/1806.07098) | June 2018 | End-to-end CNN on the waveform | None |

## TIMIT

(So far, all results trained on TIMIT and tested on the core test set.)

| PER     | Paper  | Published | Notes   | Codes | 
| :------ | :----- | :-------- | :-----: | :---: |
| 13.8%   | [The Pytorch-Kaldi Speech Recognition Toolkit](https://arxiv.org/abs/1811.07453) | February 2019 | MLP+Li-GRU+MLP on MFCC+FBANK+fMLLR | [mravanelli/pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi/tree/master/cfg/TIMIT_baselines) |
| 14.9%   | [Light Gated Recurrent Units for Speech Recognition](https://arxiv.org/abs/1803.10225) | March 2018 | Removing the reset gate in GRU, using ReLU activation instead of tanh and batch normalization | [mravanelli/pytorch-kaldi](https://github.com/mravanelli/pytorch-kaldi/tree/master/cfg/TIMIT_baselines) |
| 16.5%   | [Phone recognition with hierarchical convolutional deep maxout networks](https://link.springer.com/content/pdf/10.1186%2Fs13636-015-0068-3.pdf) | September 2015 | Hierarchical maxout CNN + Dropout | None |
| 16.5%   | [A Regularization Post Layer: An Additional Way how to Make Deep Neural Networks Robust](https://www.researchgate.net/profile/Jan_Vanek/publication/320038040_A_Regularization_Post_Layer_An_Additional_Way_How_to_Make_Deep_Neural_Networks_Robust/links/59f97fffaca272607e2f804a/A-Regularization-Post-Layer-An-Additional-Way-How-to-Make-Deep-Neural-Networks-Robust.pdf) | 2017 | DBN with last layer regularization |  None |
| 16.7%   | [Combining Time- and Frequency-Domain Convolution in Convolutional Neural Network-Based Phone Recognition](http://www.inf.u-szeged.hu/~tothl/pubs/ICASSP2014.pdf) | 2014 | CNN in time and frequency + dropout, 17.6% w/o dropout | None |
| 16.8%   | [An investigation into instantaneous frequency estimation methods for improved speech recognition features](https://ieeexplore.ieee.org/abstract/document/8308665) | November 2017 | DNN-HMM with MFCC + IFCC features | None |
| 17.3%   | [Segmental Recurrent Neural Networks for End-to-end Speech Recognition](https://arxiv.org/abs/1603.00223) | March 2016 | RNN-CRF on 24(x3) MFSC | None |
| 17.6%   | [Attention-Based Models for Speech Recognition](http://arxiv.org/abs/1506.07503) | June 2015 | Bi-RNN + Attention | [Alexander-H-Liu/End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)<br>(not official) |
| 17.7%   | [Speech Recognition with Deep Recurrent Neural Networks](http://arxiv.org/abs/1303.5778v1) | March 2013 | Bi-LSTM + skip connections w/ RNN transducer (18.4% with CTC only) | [1ytic/warp-rnnt](https://github.com/1ytic/warp-rnnt)<br>(not official) |
| 18.0%   | [Learning Filterbanks from Raw Speech for Phone Recognition](https://arxiv.org/abs/1711.01161) | October 2017 | Complex ConvNets on raw speech w/ mel-fbanks init | [facebookresearch/wav2letter](https://github.com/facebookresearch/wav2letter/tree/master/recipes/models/learnable_frontend)<br>[facebookresearch/tdfbanks](https://github.com/facebookresearch/tdfbanks) |
| 18.8%   | [Wavenet: A Generative Model For Raw Audio](https://arxiv.org/pdf/1609.03499.pdf) | September 2016 | Wavenet architecture with mean pooling layer after residual block + few non-causal conv layers | [ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)<br>(not official) |
| 23%     | [Deep Belief Networks for Phone Recognition](http://www.cs.toronto.edu/~asamir/papers/NIPS09.pdf) | 2009 | (first, modern) HMM-DBN | None |

## Hub5'00 Evaluation (Switchboard / CallHome)

(Possibly trained on more data than SWB, but test set = full Hub5'00.)

| WER (SWB) | WER (CH) | Paper          | Published | Notes   | Codes |
| :-------  | :------- | :------------- | :-------- | :-----: | :---: |
| 5.0% | 9.1%  | [The CAPIO 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1801.00059) | December 2017 | 2 Dense LSTMs + 3 CNN-bLSTMs across 3 phonesets from [previous Capio paper](https://pdfs.semanticscholar.org/d0ec/cd60d800308cd6e59810769b92b40961c09a.pdf) & AM adaptation using parameter averaging (5.6% SWB / 10.5% CH single systems) | None |
| 5.1% | 9.9%  | [Language Modeling with Highway LSTM](https://arxiv.org/abs/1709.06436) | September 2017 | HW-LSTM LM trained with Switchboard+Fisher+Gigaword+Broadcast News+Conversations, AM from [previous IBM paper](https://arxiv.org/abs/1703.02136)| None |
| 5.1% | None  | [The Microsoft 2017 Conversational Speech Recognition System](https://arxiv.org/abs/1708.06073) | August 2017 | ~2016 system + character-based dialog session aware (turns of speech) LSTM LM | None |
| 5.3% | 10.1% | [Deep Learning-based Telephony Speech Recognition in the Wild](https://pdfs.semanticscholar.org/d0ec/cd60d800308cd6e59810769b92b40961c09a.pdf) | August 2017 | Ensemble of 3 CNN-bLSTM (5.7% SWB / 11.3% CH single systems) | None |
| 5.5% | 10.3% | [English Conversational Telephone Speech Recognition by Humans and Machines](https://arxiv.org/abs/1703.02136) | March 2017 | ResNet + BiLSTMs acoustic model, with 40d FMLLR + i-Vector inputs, trained on SWB+Fisher+CH, n-gram + model-M + LSTM + Strided (à trous) convs-based LM trained on Switchboard+Fisher+Gigaword+Broadcast | None |
| 6.3% | 11.9% | [The Microsoft 2016 Conversational Speech Recognition System](http://arxiv.org/pdf/1609.03528v1.pdf) | September 2016 | VGG/Resnet/LACE/BiLSTM acoustic model trained on SWB+Fisher+CH, N-gram + RNNLM language model trained on Switchboard+Fisher+Gigaword+Broadcast | None |
| 6.6% | 12.2% | [The IBM 2016 English Conversational Telephone Speech Recognition System](http://arxiv.org/pdf/1604.08242v2.pdf) | June 2016 | RNN + VGG + LSTM acoustic model trained on SWB+Fisher+CH, N-gram + "model M" + NNLM language model | None |
| 6.8% | 14.1% | [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/abs/1904.08779) | April 2019 | Listen Attend Spell | [DemisEom/SpecAugment](https://github.com/DemisEom/SpecAugment) <br> (not official) |
| 8.5% | 13% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-BLSTM trained with MMI + data augmentation (speed) + iVectors + 3 regularizations + Fisher | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/fisher_swbd/s5) |
| 9.2% | 13.3% | [Purely sequence-trained neural networks for ASR based on lattice-free MMI](http://www.danielpovey.com/files/2016_interspeech_mmi.pdf) | September 2016 | HMM-TDNN trained with MMI + data augmentation (speed) + iVectors + 3 regularizations + Fisher (10% / 15.1% respectively trained on SWBD only) | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/fisher_swbd/s5) |
| 12.6% | 16% | [Deep Speech: Scaling up end-to-end speech recognition](https://arxiv.org/abs/1412.5567) | December 2014 | CNN + Bi-RNN + CTC (speech to letters), 25.9% WER if trained _only_ on SWB | [mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech)<br>(not official) |
| 11% | 17.1% | [A time delay neural network architecture for efficient modeling of long temporal contexts](http://speak.clsp.jhu.edu/uploads/publications/papers/1048_pdf.pdf) | 2015 | HMM-TDNN + iVectors | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/swbd/s5c) |
| 12.6% | 18.4% | [Sequence-discriminative training of deep neural networks](http://www.danielpovey.com/files/2013_interspeech_dnn.pdf) | 2013 | HMM-DNN +sMBR | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/swbd/s5c) |
| 12.9% | 19.3% | [Audio Augmentation for Speech Recognition](http://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) | 2015 | HMM-TDNN + pNorm + speed up/down speech | [kaldi-asr/kaldi](https://github.com/kaldi-asr/kaldi/blob/master/egs/swbd/s5c) |
| 15% | 19.1% | [Building DNN Acoustic Models for Large Vocabulary Speech Recognition](http://arxiv.org/abs/1406.7806v2) | June 2014  | DNN + Dropout | None |
| 10.4% | None | [Joint Training of Convolutional and Non-Convolutional Neural Networks](http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202014/papers/p5609-soltau.pdf) | 2014 | CNN on MFSC/fbanks + 1 non-conv layer for FMLLR/I-Vectors concatenated in a DNN | None |
| 11.5% | None | [Deep Convolutional Neural Networks for LVCSR](http://www.cs.toronto.edu/~asamir/papers/icassp13_cnn.pdf) | 2013 | CNN | None |
| 12.2% | None| [Very Deep Multilingual Convolutional Neural Networks for LVCSR](http://arxiv.org/pdf/1509.08967v1.pdf) | September 2015 | Deep CNN (10 conv, 4 FC layers), multi-scale feature maps | None |
| 11.8% | 25.7% | [Improved training of end-to-end attention models for speech recognition](https://www-i6.informatik.rwth-aachen.de/publications/download/1068/Zeyer--2018.pdf) | Interspeech, Sept 2018 | encoder-attention-decoder end-to-end model, trained on 300h SWB | [rwth-i6/returnn](https://github.com/rwth-i6/returnn)<br>[rwth-i6/returnn-experiments](https://github.com/rwth-i6/returnn-experiments/tree/master/2018-asr-attention) |


## Lexicon

* WER: word error rate
* PER: phone error rate
* LM: language model
* HMM: hidden markov model
* GMM: Gaussian mixture model
* DNN: deep neural network
* CNN: convolutional neural network
* DBN: deep belief network (RBM-based DNN)
* TDNN-F: a factored form of time delay neural networks (TDNN)
* RNN: recurrent neural network
* LSTM: long short-term memory
* CTC: connectionist temporal classification
* MMI: maximum mutual information (MMI),
* MPE: minimum phone error
* sMBR: state-level minimum Bayes risk
* SAT: speaker adaptive training
* MLLR: maximum likelihood linear regression
* LDA: (in this context) linear discriminant analysis
* MFCC: [Mel frequency cepstral coefficients](http://snippyhollow.github.io/blog/2014/09/25/classical-speech-recognition-features-in-one-picture/)
* FB/FBANKS/MFSC: [Mel frequency spectral coefficients](http://snippyhollow.github.io/blog/2014/09/25/classical-speech-recognition-features-in-one-picture/)
* IFCC: Instantaneous frequency cosine coefficients (https://github.com/siplabiith/IFCC-Feature-Extraction)
* VGG: very deep convolutional neural networks from Visual Graphics Group, VGG is an architecture of 2 {3x3 convolutions} followed by 1 pooling, repeated
