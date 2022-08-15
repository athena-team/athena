
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
