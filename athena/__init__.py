# Copyright (C) 2017 Beijing Didi Infinity Technology and Development Co.,Ltd.
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
# data
from .data import SpeechRecognitionDatasetBuilder
from .data import SpeechRecognitionDatasetKaldiIOBuilder
from .data import SpeechSynthesisDatasetBuilder
from .data import SpeechDatasetBuilder
from .data import SpeechDatasetKaldiIOBuilder
from .data import SpeakerRecognitionDatasetBuilder
from .data import SpeakerVerificationDatasetBuilder
from .data import LanguageDatasetBuilder
from .data import VoiceConversionDatasetBuilder
from .data import FeatureNormalizer
from .data import WorldFeatureNormalizer
from .data.text_featurizer import TextFeaturizer

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

# models
from .models.base import BaseModel
from .models.speech_transformer import SpeechTransformer, SpeechTransformer2
from .models.tacotron2 import Tacotron2
from .models.tts_transformer import TTSTransformer
from .models.fastspeech import FastSpeech
from .models.masked_pc import MaskedPredictCoding
from .models.deep_speech import DeepSpeechModel
from .models.mtl_seq2seq import MtlTransformerCtc
from .models.rnn_lm import RNNLM
from .models.translate_transformer import NeuralTranslateTransformer
from .models.stargan_vc import StarganModel
from .models.speaker_resnet import SpeakerResnet

# solver & loss & accuracy
from .solver import BaseSolver
from .solver import GanSolver
from .solver import HorovodSolver
from .solver import DecoderSolver
from .solver import SynthesisSolver
from .solver import ConvertSolver
from .loss import CTCLoss
from .loss import Seq2SeqSparseCategoricalCrossentropy
from .loss import StarganLoss
from .metrics import CTCAccuracy
from .metrics import Seq2SeqSparseCategoricalAccuracy

# utils
from .utils.checkpoint import Checkpoint
from .utils.learning_rate import WarmUpLearningSchedule, WarmUpAdam
from .utils.learning_rate import (
    ExponentialDecayLearningRateSchedule,
    ExponentialDecayAdam,
)
from .utils.hparam import HParams, register_and_parse_hparams
from .utils.misc import generate_square_subsequent_mask
from .utils.misc import get_wave_file_length
from .utils.misc import set_default_summary_writer

# tools
from .tools.beam_search import BeamSearchDecoder
