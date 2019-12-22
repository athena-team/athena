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
"""The model provides a general interface for feature extraction."""

import tensorflow as tf
from athena.transform import feats


class AudioFeaturizer:
    """
        Interface of audio Features extractions.
    """
    #pylint: disable=dangerous-default-value
    def __init__(self, config={"type": "Fbank"}):
        """init
        :param name Feature name, eg fbank, mfcc, plp ...
        :param config
        'type': 'ReadWav', 'Fbank', 'Spectrum'
        The config for fbank
          'sample_rate' 16000
          'window_length' 0.025
          'frame_length' 0.010
          'upper_frequency_limit' 20
          'filterbank_channel_count' 40
        The config for Spectrum
          'sample_rate' 16000
          'window_length' 0.025
          'frame_length' 0.010
          'output_type' 1
        """

        assert "type" in config

        self.name = config["type"]
        self.feat = getattr(feats, self.name).params(config).instantiate()

        if self.name != "ReadWav":
            self.read_wav = getattr(feats, "ReadWav").params(config).instantiate()

    #pylint:disable=invalid-name
    def __call__(self, audio=None, sr=None, speed=1.0):
        """extract feature from audo data
        :param audio data or audio file
        :sr sample rate
        :return feature
        """

        if audio is not None and not tf.is_tensor(audio):
            audio = tf.convert_to_tensor(audio)
        if sr is not None and not tf.is_tensor(sr):
            sr = tf.convert_to_tensor(sr)

        return self.__impl(audio, sr, speed)

    @tf.function
    def __impl(self, audio=None, sr=None, speed=1.0):
        """
        :param audio data or audio file, a tensor
        :sr sample rate, a tensor
        :return feature
        """
        if self.name == "ReadWav" or self.name == "CMVN":
            return self.feat(audio, speed)
        elif audio.dtype is tf.string:
            audio_data, sr = self.read_wav(audio, speed)
            return self.feat(audio_data, sr)
        else:
            return self.feat(audio, sr)

    @property
    def dim(self):
        """return the dimension of the feature
    if only ReadWav, return 1
    """
        return self.feat.dim()

    @property
    def num_channels(self):
        """return the channel of the feature"""
        return self.feat.num_channels()
