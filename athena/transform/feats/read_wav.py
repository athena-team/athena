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
"""The model reads audio sample from wav file."""

import tensorflow as tf
from athena.utils.hparam import HParams
from athena.transform.feats.base_frontend import BaseFrontend
from athena.transform.feats.ops import py_x_ops


class ReadWav(BaseFrontend):
    """
    Read audio sample from wav file, return sample data and sample rate.
    """
    def __init__(self, config: dict):
        super().__init__(config)

    @classmethod
    def params(cls, config=None):
        """
          Set params.
           :param config: contains one optional parameters: audio_channels(int, default=1).
           :return: An object of class HParams, which is a set of hyperparameters as
                    name-value pairs.
           """
        audio_channels = 1

        hparams = HParams(cls=cls)
        hparams.add_hparam('type', 'ReadWav')
        hparams.add_hparam('audio_channels', audio_channels)

        if config is not None:
            hparams.parse(config, True)

        return hparams

    def call(self, wavfile, speed=1.0):
        """
        Get audio data and sample rate from a wavfile.
        :param wavfile: filepath of wav
               speed: Speed of sample channels wanted (float, default=1.0)
        :return: 2 values. The first is a Tensor of audio data.
            The second return value isthe sample rate of the input wav
            file, which is a tensor with float dtype.
        """
        p = self.config
        contents = tf.io.read_file(wavfile)
        audio_data, sample_rate = tf.compat.v1.audio.decode_wav(
            contents, desired_channels=p.audio_channels)
        if (speed == 1.0):
            return tf.squeeze(audio_data * 32768, axis=-1), tf.cast(sample_rate, dtype=tf.int32)
        else:
            resample_rate = tf.cast(sample_rate, dtype=tf.float32) * tf.cast(1.0 / speed, dtype=tf.float32)
            speed_data = py_x_ops.speed(tf.squeeze(audio_data * 32768, axis=-1),
                                        tf.cast(sample_rate, dtype=tf.int32),
                                        tf.cast(resample_rate, dtype=tf.int32),
                                        lowpass_filter_width=5)
            return tf.squeeze(speed_data), tf.cast(sample_rate, dtype=tf.int32)


def read_wav(wavfile, audio_channels=1):
    """ read wav from file
    args: audio_channels = 1
    returns: tf.squeeze(audio_data * 32768, axis=-1), tf.cast(sample_rate, dtype=tf.int32)
    """
    contents = tf.io.read_file(wavfile)
    audio_data, sample_rate = tf.compat.v1.audio.decode_wav(
        contents, desired_channels=audio_channels
    )

    return tf.squeeze(audio_data * 32768, axis=-1), tf.cast(sample_rate, dtype=tf.int32)
