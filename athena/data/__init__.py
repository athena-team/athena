# coding=utf-8
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
""" data """
# Speech dataset
from .datasets.speech_set import SpeechDatasetBuilder

#ASR dataset
from athena.data.datasets.asr.speech_recognition import SpeechRecognitionDatasetBuilder
from athena.data.datasets.asr.speech_recognition_batch_bins import SpeechRecognitionDatasetBatchBinsBuilder
from athena.data.datasets.asr.speech_recognition_kaldiio import SpeechRecognitionDatasetKaldiIOBuilder
from athena.data.datasets.asr.speech_recognition_batch_bins_kaldiio import SpeechRecognitionDatasetBatchBinsKaldiIOBuilder

#AV-ASR dataset
from athena.data.datasets.asr.audio_video_recognition import AudioVedioRecognitionDatasetBuilder
from athena.data.datasets.asr.audio_video_recognition_batch_bins import AudioVedioRecognitionDatasetBatchBinsBuilder

#TTS dataset
from athena.data.datasets.tts.speech_synthesis import SpeechSynthesisDatasetBuilder
from athena.data.datasets.tts.speech_fastspeech2 import SpeechFastspeech2DatasetBuilder
from athena.data.datasets.tts.speech_synthesis_test import SpeechSynthesisTestDatasetBuilder

#KWS dataset
from .datasets.kws.speech_wakeup_framewise_kaldiio import SpeechWakeupFramewiseDatasetKaldiIOBuilder
from .datasets.kws.speech_wakeup_kaldiio import SpeechWakeupDatasetKaldiIOBuilder
from .datasets.kws.speech_wakeup_kaldiio_av import SpeechWakeupDatasetKaldiIOBuilderAVCE

# LM dataset
from .datasets.language_set import LanguageDatasetBuilder

#MPC dataset
from athena.data.datasets.mpc.mpc_speech_set import MpcSpeechDatasetBuilder
from athena.data.datasets.mpc.mpc_speech_set_kaldiio import MpcSpeechDatasetKaldiIOBuilder

#VAD dataset
from athena.data.datasets.vad.vad_set_kaldiio import VoiceActivityDetectionDatasetKaldiIOBuilder

#Feature normal
from .feature_normalizer import FeatureNormalizer
from .feature_normalizer import FS2FeatureNormalizer

#Text processing
from .text_featurizer import TextFeaturizer, SentencePieceFeaturizer, TextTokenizer

