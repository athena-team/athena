# Copyright (C) ATHENA AUTHORS
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" module """
#Speech dataset
from .data import SpeechDatasetBuilder
from .data import LanguageDatasetBuilder

#ASR dataset
from .data import SpeechRecognitionDatasetBuilder
from .data import SpeechRecognitionDatasetKaldiIOBuilder
from .data import SpeechRecognitionDatasetBatchBinsBuilder
from .data import SpeechRecognitionDatasetBatchBinsKaldiIOBuilder

#AV-ASR dataset
from .data import AudioVedioRecognitionDatasetBuilder
from .data import AudioVedioRecognitionDatasetBatchBinsBuilder

#MPC dataset
from .data import MpcSpeechDatasetBuilder
from .data import MpcSpeechDatasetKaldiIOBuilder

#TTS dataset
from .data import SpeechSynthesisDatasetBuilder
from .data import SpeechFastspeech2DatasetBuilder
from .data import FeatureNormalizer
from .data import FS2FeatureNormalizer
from .data import VoiceActivityDetectionDatasetKaldiIOBuilder

#KWS dataset
from .data import SpeechWakeupFramewiseDatasetKaldiIOBuilder
from .data import SpeechWakeupDatasetKaldiIOBuilder
from .data import SpeechWakeupDatasetKaldiIOBuilderAVCE

#Text processing
from .data.text_featurizer import TextFeaturizer
from .data.text_featurizer import TextTokenizer

# layers
from .layers.functional import make_positional_encoding
from .layers.functional import collapse4d
from .layers.functional import gelu
from .layers.commons import PositionalEncoding
from .layers.commons import Collapse4D
from .layers.commons import TdnnLayer
from .layers.commons import Gelu
from .layers.attention import MultiHeadAttention
from .layers.attention import BahdanauAttention
from .layers.attention import HanAttention
from .layers.attention import MatchAttention
from .layers.transformer import Transformer
from .layers.transformer import TransformerEncoder
from .layers.transformer import TransformerDecoder
from .layers.transformer import TransformerEncoderLayer
from .layers.transformer import TransformerDecoderLayer
from .layers.resnet_block import ResnetBasicBlock
from .models.base import BaseModel
from .models.masked_pc import MaskedPredictCoding

# ASR models
from athena.models.asr.av_conformer import AudioVideoConformer
from athena.models.asr.av_mtl_seq2seq import AV_MtlTransformer
from athena.models.asr.speech_conformer import SpeechConformer
from athena.models.asr.speech_conformer_ctc import SpeechConformerCTC
from athena.models.asr.speech_transformer import SpeechTransformer
from athena.models.asr.speech_u2 import SpeechTransformerU2, SpeechConformerU2
from athena.models.asr.mtl_seq2seq import MtlTransformerCtc

#AV ASR model
from athena.models.asr.av_conformer import AudioVideoConformer

#VAD model
from athena.models.vad.vad_marblenet import VadMarbleNet
from athena.models.vad.vad_dnn import VadDnn

#LM model
from athena.models.lm import RNNLM, TransformerLM

#TTS model
from athena.models.tts.fastspeech import FastSpeech
from athena.models.tts.fastspeech2 import FastSpeech2
from athena.models.tts.tacotron2 import Tacotron2
from athena.models.tts.tts_transformer import TTSTransformer

# kws model
from .models.kws.cnn_wakeup import CnnModel
from .models.kws.conformer_wakeup import KWSConformer
from .models.kws.crnn_wakeup import CRnnModel
from .models.kws.dnn_wakeup import DnnModel
from .models.kws.misp_wakeup import MISPModel
from .models.kws.transformer_wakeup_2dense import KWSTransformer_2Dense
from .models.kws.transformer_wakeup import KWSTransformer
from .models.kws.transformer_av_wakeup import KWSAVTransformer
from .models.kws.transformer_wakeup_resnet import KWSTransformerRESNET
from .models.kws.transformer_wakeup_focal_loss import KWSTransformer_FocalLoss

# solver & loss & accuracy
# Base solver
from .solver import BaseSolver
from .solver import HorovodSolver
from .solver import DecoderSolver

# AV sovloer for ASR
from .solver import AVSolver
from .solver import AVHorovodSolver
from .solver import AVDecoderSolver

# VAD solver
from .solver import VadSolver

# TTS solver
from .solver import SynthesisSolver

# loss
from .loss import CTCLoss
from .loss import Seq2SeqSparseCategoricalCrossentropy

# Accuracy calculation
from .metrics import CTCAccuracy
from .metrics import Seq2SeqSparseCategoricalAccuracy

# utils
from .utils.checkpoint import Checkpoint
from .utils.learning_rate import WarmUpLearningSchedule, WarmUpAdam, WarmUpLearningSchedule1, WarmUpAdam1
from .utils.learning_rate import (
    ExponentialDecayLearningRateSchedule,
    ExponentialDecayAdam,
)
from .utils.hparam import HParams, register_and_parse_hparams
from .utils.misc import generate_square_subsequent_mask, generate_square_subsequent_mask_u2
from .utils.misc import get_wave_file_length
from .utils.misc import set_default_summary_writer
from .utils.misc import get_dict_from_scp
#from .utils.logger import LOG
from .tools.ctc_scorer import CTCPrefixScoreTH

__version__ = "2.0"
