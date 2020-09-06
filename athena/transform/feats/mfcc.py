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
"""This model extracts MFCC features per frame."""

import tensorflow as tf
from athena.utils.hparam import HParams
from athena.transform.feats.ops import py_x_ops
from athena.transform.feats.base_frontend import BaseFrontend
from athena.transform.feats.framepow import Framepow
from athena.transform.feats.fbank import Fbank
from athena.transform.feats.cmvn import CMVN


class Mfcc(BaseFrontend):
    """Compute MFCC features of every frame in speech.

    Args:
         config: contains fifteen optional parameters.

    Shape:
        - output: :math:`(C, T, F)`.

    Examples::
        >>> config = {'cepstral_lifter': 22.0, 'coefficient_count': 13}
        >>> mfcc_op = Mfcc.params(config).instantiate()
        >>> mfcc_out = mfcc_op('test.wav', 16000)
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.framepow = Framepow(config)
        self.fbank = Fbank(config)

    @classmethod
    def params(cls, config=None):
        """Set params.

        Args:
            config: contains the following fifteen optional parameters:

            'window_length': Window length in seconds. (float, default = 0.025),
            'frame_length': Hop length in seconds. (float, default = 0.010),
            'snip_edges': If 1, the last frame (shorter than window_length) will be
                          cutoff. If 2, 1 // 2 frame_length data will be padded
                          to data. (int, default = 1),
            'preEph_coeff': Coefficient for use in frame-signal preemphasis.
                            (float, default = 0.97),
            'window_type': Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria").
                            (string, default = "povey")
            'remove_dc_offset': Subtract mean from waveform on each frame.
                                (bool, default = true)
            'is_fbank': If true, compute power spetrum without frame energy.
                          If false, using the frame energy instead of the
                          square of the constant component of the signal.
                          (bool, default = true)
            'coefficient_count': Number of cepstra in MFCC computation. (int, default = 13)
            'output_type': If 1, return power spectrum. If 2, return log-power
                            spectrum. (int, default = 1)
            'upper_frequency_limit': High cutoff frequency for mel bins (if <= 0, offset
                                      from Nyquist) (float, default = 0)
            'lower_frequency_limit': Low cutoff frequency for mel bins. (float, default = 20)
            'filterbank_channel_count': Number of triangular mel-frequency bins.
                                        (float, default = 23)
            'dither': Dithering constant (0.0 means no dither).
                      (float, default = 1) [add robust to training]
            'cepstral_lifter': Constant that controls scaling of MFCCs. (float, default = 22)
            'use_energy': Use energy (not C0) in MFCC computation. (bool, default = True)

        Note:
            Return an object of class HParams, which is a set of hyperparameters as
            name-value pairs.
        """

        upper_frequency_limit = 0.0
        lower_frequency_limit = 20.0
        filterbank_channel_count = 23.0
        window_length = 0.025
        frame_length = 0.010
        output_type = 1
        snip_edges = 1
        raw_energy = 1
        preEph_coeff = 0.97
        window_type = "povey"
        remove_dc_offset = True
        is_fbank = True
        cepstral_lifter = 22.0
        coefficient_count = 13
        use_energy = True
        dither = 0.0
        delta_delta = False
        order = 2
        window = 2

        hparams = HParams(cls=cls)
        hparams.add_hparam("upper_frequency_limit", upper_frequency_limit)
        hparams.add_hparam("lower_frequency_limit", lower_frequency_limit)
        hparams.add_hparam("filterbank_channel_count", filterbank_channel_count)
        hparams.add_hparam("window_length", window_length)
        hparams.add_hparam("frame_length", frame_length)
        hparams.add_hparam("output_type", output_type)
        hparams.add_hparam("snip_edges", snip_edges)
        hparams.add_hparam("raw_energy", raw_energy)
        hparams.add_hparam("preEph_coeff", preEph_coeff)
        hparams.add_hparam("window_type", window_type)
        hparams.add_hparam("remove_dc_offset", remove_dc_offset)
        hparams.add_hparam("is_fbank", is_fbank)
        hparams.add_hparam("cepstral_lifter", cepstral_lifter)
        hparams.add_hparam("coefficient_count", coefficient_count)
        hparams.add_hparam("use_energy", use_energy)
        hparams.add_hparam("dither", dither)
        hparams.add_hparam("delta_delta", delta_delta)
        hparams.add_hparam("order", order)
        hparams.add_hparam("window", window)
        hparams.add_hparam("channel", 1)

        hparams.append(CMVN.params())

        if config is not None:
            hparams.parse(config, True)

        return hparams

    def call(self, audio_data, sample_rate):
        """Caculate mfcc features of audio data.

        Args:
            audio_data: the audio signal from which to compute mfcc.
            sample_rate: the sample rate of the signal we working with.

        Shape:
            - audio_data: :math:`(1, N)`
            - sample_rate: float
        """
        p = self.config
        with tf.name_scope('mfcc'):
            fbank_feats = self.fbank(audio_data, sample_rate)
            sample_rate = tf.cast(sample_rate, dtype=tf.int32)
            shape = tf.shape(fbank_feats)
            nframe = shape[0]
            nfbank = shape[1]
            fbank_feats = tf.reshape(fbank_feats, (p.channel, nframe, nfbank))
            framepow_feats = self.framepow(audio_data, sample_rate)
            mfcc = py_x_ops.mfcc(
                fbank_feats,
                framepow_feats,
                sample_rate,
                use_energy=p.use_energy,
                cepstral_lifter=p.cepstral_lifter,
                coefficient_count=p.coefficient_count)
            return mfcc

    def dim(self):
        p = self.config
        return int(p.coefficient_count)