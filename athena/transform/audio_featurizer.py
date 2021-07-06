# Copyright (C) ATHENA AUTHORS; Xiaoning Lei; Shuaijiang Zhao
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
import kaldiio
from athena.transform.feats.ops import py_x_ops

def ReadLongWave(wavfile, speed=1.0):
    sample_rate, sample_data = kaldiio.load_mat(wavfile)
    if speed == 1.0:
        return sample_data, sample_rate
    else:
        resample_rate = tf.cast(sample_rate, dtype=tf.float32) * tf.cast(1.0 / speed, dtype=tf.float32)

        speed_data = py_x_ops.speed(sample_data,
                       tf.cast(sample_rate, dtype=tf.int32),
                       tf.cast(resample_rate, dtype=tf.int32),
                       lowpass_filter_width=5)
        return tf.squeeze(speed_data), tf.cast(sample_rate, dtype=tf.int32)

class AudioFeaturizer:
    """Interface of audio features extractions. The kernels of features are based on
    Kaldi (Povey D, Ghoshal A, Boulianne G, et al. The Kaldi speech recognition toolkit[C]//IEEE
    2011 workshop on automatic speech recognition and understanding. IEEE Signal Processing Society,
    2011 (CONF). ) and Librosa.

    Now Transform supports the following 7 features: Spectrum, MelSpectrum, Framepow,
    Pitch, Mfcc, Fbank, FbankPitch.

    Args:
        config: a dictionary contains parameters of feature extraction.

    Examples::
        >>> fbank_op = AudioFeaturizer(config={'type':'Fbank', 'filterbank_channel_count':40,
        >>> 'lower_frequency_limit': 60, 'upper_frequency_limit':7600})
        >>> fbank_out = fbank_op('test.wav')
    """

    def __init__(self, config={"type": "Fbank"}):
        """Init config of AudioFeaturizer.

        Args:
            config: 'type': The name of the feature you want to extract. That is,
                    'ReadWav', 'WriteWav', 'Fbank', 'Spectrum', 'MelSpectrum'
                    'Framepow', 'Pitch', 'Mfcc', 'Fbank', 'FbankPitch'

            Note:
                Specific parameters of different features can be seen in the docs of Transform class.

        """

        assert "type" in config

        self.name = config["type"]
        if self.name == 'FbankFromLongWave':
            self.read_wav = ReadLongWave
            self.feat = getattr(feats, 'Fbank').params(config).instantiate()
        else:
            self.feat = getattr(feats, self.name).params(config).instantiate()
            if self.name != "ReadWav":
                self.read_wav = getattr(feats, "ReadWav").params(config).instantiate()
                self.feat = getattr(feats, self.name).params(config).instantiate()


    def __call__(self, audio=None, sr=None, speed=1.0, seg_wav=None):
        """Extract feature from audio data.

        Args:
            audio: filename of wav or audio data.
            sr: the sample rate of the signal we working with. (default=None)
            speed: adjust audio speed. (default=1.0)

        Shape:
            - audio: string or array with :math:`(1, L)`.
            - sr: int
            - speed: float
            - output: see the docs in the feature class.
        """

        if audio is not None and not tf.is_tensor(audio):
            if self.name != 'FbankFromLongWave':
                audio = tf.convert_to_tensor(audio)
            pass
        if sr is not None and not tf.is_tensor(sr):
            sr = tf.convert_to_tensor(sr)

        return self.__impl(audio, sr, speed, seg_wav=seg_wav)

    def __impl(self, audio=None, sr=None, speed=1.0, seg_wav=None):
        """Call OP of features to extract features.

        Args:
            audio: a tensor of audio data or audio file
            sr: a tensor of sample rate
        """
        if self.name == "ReadWav" or self.name == "CMVN":
            return self.feat(audio, speed)

        elif self.name == 'FbankFromLongWave':
            if seg_wav is None:
                audio_data, sr = self.read_wav(audio, speed)
                return self.feat(audio_data, sr)
            else:
                if speed == 1.0:
                    sr, audio_data = seg_wav[audio]
                else:
                    sample_rate, sample_data = seg_wav[audio]

                    resample_rate = tf.cast(sample_rate, dtype=tf.float32) * tf.cast(1.0 / speed, dtype=tf.float32)

                    speed_data = py_x_ops.speed(sample_data,
                                                tf.cast(sample_rate, dtype=tf.int32),
                                                tf.cast(resample_rate, dtype=tf.int32),
                                                lowpass_filter_width=5)
                    audio_data = tf.squeeze(speed_data)
                    sr = tf.cast(sample_rate, dtype=tf.int32)

                return self.feat(audio_data, sr)

        elif audio.dtype is tf.string:
            audio_data, sr = self.read_wav(audio, speed)
            return self.feat(audio_data, sr)
        else:
            return self.feat(audio, sr)

    @property
    def dim(self):
        """Return the dimension of the feature, if only ReadWav, return 1. Else,
        see the docs in the feature class.
        """
        return self.feat.dim()

    @property
    def num_channels(self):
        """return the channel of the feature"""
        return self.feat.num_channels()
