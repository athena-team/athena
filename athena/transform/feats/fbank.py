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
"""This model extracts Fbank features per frame."""

import tensorflow as tf
from athena.utils.hparam import HParams
from athena.transform.feats.ops import py_x_ops
from athena.transform.feats.base_frontend import BaseFrontend
from athena.transform.feats.spectrum import Spectrum
from athena.transform.feats.cmvn import CMVN


class Fbank(BaseFrontend):
    """
    Computing filter banks is applying triangular filters on a Mel-scale to the power
     spectrum to extract frequency bands. Return a float tensor with shape
     (num_channels, num_frames, num_frequencies).
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.spect = Spectrum(config)
        self.cmvn = CMVN(config)

        # global cmvn dim == feature dim
        if config.type == "Fbank" and self.cmvn.global_cmvn:
            assert config.filterbank_channel_count * config.channel == len(
                config.global_mean
            ), "Error, feature dim {} is not equals to cmvn dim {}".format(
                config.filterbank_channel_count * config.channel,
                len(config.global_mean),
            )
        print("Fbank params: ", self.config)

    @classmethod
    def params(cls, config=None):
        """
        Set params.
        :param config: contains thirteen optional parameters:
                --window_length				: Window length in seconds. (float, default = 0.025)
                --frame_length				: Hop length in seconds. (float, default = 0.010)
                --snip_edges				: If 1, the last frame (shorter than window_length) will be
                                              cutoff. If 2, 1 // 2 frame_length data will be padded
                                              to data. (int, default = 1)
                ---raw_energy				: If 1, compute frame energy before preemphasis and
                                              windowing. If 2,  compute frame energy after
                                              preemphasis and windowing. (int, default = 1)
                --preEph_coeff				: Coefficient for use in frame-signal preemphasis.
                                             (float, default = 0.97)
                --window_type				: Type of window ("hamm"|"hann"|"povey"|"rect"|"blac"|"tria").
                                             (string, default = "povey")
                --remove_dc_offset			: Subtract mean from waveform on each frame.
                                              (bool, default = true)
                --is_fbank					: If true, compute power spetrum without frame energy.
                                              If false, using the frame energy instead of the
                                              square of the constant component of the signal.
                                              (bool, default = true)
                --output_type				: If 1, return power spectrum. If 2, return log-power
                                              spectrum. (int, default = 1)
                --upper_frequency_limit		: High cutoff frequency for mel bins (if <= 0, offset
                                             from Nyquist) (float, default = 0)
                --lower_frequency_limit		: Low cutoff frequency for mel bins (float, default = 20)
                --filterbank_channel_count	: Number of triangular mel-frequency bins.
                                             (float, default = 23)
                --dither			    	: Dithering constant (0.0 means no dither).
                                             (float, default = 1) [add robust to training]
        :return: An object of class HParams, which is a set of hyperparameters as name-value pairs.
        """

        hparams = HParams(cls=cls)

        # spectrum
        hparams.append(Spectrum.params({"output_type": 1, "is_fbank": True}))

        # fbank
        upper_frequency_limit = 0
        lower_frequency_limit = 60
        filterbank_channel_count = 40
        hparams.add_hparam("upper_frequency_limit", upper_frequency_limit)
        hparams.add_hparam("lower_frequency_limit", lower_frequency_limit)
        hparams.add_hparam("filterbank_channel_count", filterbank_channel_count)

        # delta
        delta_delta = False  # True
        order = 2
        window = 2
        hparams.add_hparam("delta_delta", delta_delta)
        hparams.add_hparam("order", order)
        hparams.add_hparam("window", window)

        if config is not None:
            hparams.parse(config, True)

        hparams.type = "Fbank"

        hparams.add_hparam("channel", 1)
        if hparams.delta_delta:
            hparams.channel = hparams.order + 1

        return hparams

    def call(self, audio_data, sample_rate):
        """
        Caculate fbank features of audio data.
        :param audio_data: the audio signal from which to compute spectrum.
                          Should be an (1, N) tensor.
        :param sample_rate: [option]the samplerate of the signal we working with,
                            default is 16kHz.
        :return: A float tensor of size (num_channels, num_frames, num_frequencies) containing
               fbank features of every frame in speech.
        """
        p = self.config

        with tf.name_scope('fbank'):

            spectrum = self.spect(audio_data, sample_rate)
            spectrum = tf.expand_dims(spectrum, 0)
            sample_rate = tf.cast(sample_rate, dtype=tf.int32)

            fbank = py_x_ops.fbank(spectrum,
                                   sample_rate,
                                   upper_frequency_limit=p.upper_frequency_limit,
                                   lower_frequency_limit=p.lower_frequency_limit,
                                   filterbank_channel_count=p.filterbank_channel_count)

            fbank = tf.squeeze(fbank, axis=0)
            shape = tf.shape(fbank)
            nframe = shape[0]
            nfbank = shape[1]
            if p.delta_delta:
                fbank = py_x_ops.delta_delta(fbank, p.order, p.window)
            if p.type == 'Fbank':
                fbank = self.cmvn(fbank)

            fbank = tf.reshape(fbank, (nframe, nfbank, p.channel))

            return fbank

    def dim(self):
        p = self.config
        return p.filterbank_channel_count

    def num_channels(self):
        p = self.config
        return p.channel
