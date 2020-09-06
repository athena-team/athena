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
""""This model extracts framepow features per frame."""

import tensorflow as tf
from athena.utils.hparam import HParams
from athena.transform.feats.ops import py_x_ops
from athena.transform.feats.base_frontend import BaseFrontend


class Framepow(BaseFrontend):
    """Compute power of every frame in speech.

     Args:
         config: contains four optional parameters.

     Shape:
        - output: :math:`(T, 1)`.

     Examples::
        >>> config = {'window_length': 0.25, 'remove_dc_offset': True}
        >>> framepow_op = Framepow.params(config).instantiate()
        >>> framepow_out = framepow_op('test.wav', 16000)
    """
    def __init__(self, config: dict):
        super().__init__(config)

    @classmethod
    def params(cls, config=None):
        """Set params.

        Args:
            config: contains the following four optional parameters:

            'window_length': Window length in seconds. (float, default = 0.025)
            'frame_length': Hop length in seconds. (float, default = 0.010)
            'snip_edges': If 1, the last frame (shorter than window_length) will be
                          cutoff. If 2, 1 // 2 frame_length data will be padded
                          to data. (int, default = 1)
            'remove_dc_offset': Subtract mean from waveform on each frame.
                                (bool, default = true)

        Note:
            Return an object of class HParams, which is a set of hyperparameters as
            name-value pairs.
        """

        window_length = 0.025
        frame_length = 0.010
        snip_edges = 1
        remove_dc_offset = True

        hparams = HParams(cls=cls)
        hparams.add_hparam("window_length", window_length)
        hparams.add_hparam("frame_length", frame_length)
        hparams.add_hparam("snip_edges", snip_edges)
        hparams.add_hparam("remove_dc_offset", remove_dc_offset)

        if config is not None:
            hparams.parse(config, True)

        return hparams

    def call(self, audio_data, sample_rate):
        """Caculate power of every frame in speech.

        Args:
            audio_data: the audio signal from which to compute spectrum.
            sample_rate: the sample rate of the signal we working with.

        Shape:
            - audio_data: :math:`(1, N)`
            - sample_rate: float
        """

        p = self.config
        with tf.name_scope('framepow'):
            sample_rate = tf.cast(sample_rate, dtype=float)
            framepow = py_x_ops.frame_pow(
                audio_data,
                sample_rate,
                snip_edges=p.snip_edges,
                remove_dc_offset=p.remove_dc_offset,
                window_length=p.window_length,
                frame_length=p.frame_length)

            return tf.squeeze(framepow)

    def dim(self):
        return 1