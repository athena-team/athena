# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Ruixiong Zhang; Lan Yu;
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
# Only support eager mode
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
""" Implementation of GriffinLim vocoder """
import os
import numpy as np
import librosa
from scipy.io.wavfile import write as write_wav

class GriffinLim:
    """python implementation of griffinlim algorithm"""

    def __init__(self, data_descriptions):
        """Reference: to paper "Multiband Excitation Vocoder"
        """
        assert data_descriptions.audio_featurizer is not None
        assert data_descriptions.audio_featurizer.feat is not None
        assert data_descriptions.hparams.audio_config is not None
        params_func = data_descriptions.audio_featurizer.feat.params
        params = params_func(data_descriptions.hparams.audio_config)
        self.channels = params.filterbank_channel_count
        self.sample_rate = params.sample_rate
        self.window_length = int(params.window_length * self.sample_rate)
        self.hop_length = int(params.frame_length * self.sample_rate)
        self.n_fft = self._get_nfft(self.window_length)
        self.lower_frequency_limit = params.lower_frequency_limit
        self.upper_frequency_limit = params.upper_frequency_limit
        self.window_type = params.window_type
        self.EPS = 1e-10

    def _get_nfft(self, window_length):
        """n_fft is an exponential power of 2 closest to and larger than win_length"""
        nfft = 2
        while nfft < window_length:
            nfft *= 2
        return nfft

    def __call__(self, feats, hparams, name=None):
        linear_feats = self._logmel_to_linear(feats)
        samples = self._griffin_lim(linear_feats, hparams.gl_iters)
        samples = samples / 32768
        if not os.path.exists(hparams.output_directory):
            os.makedirs(hparams.output_directory)
        output_path = os.path.join(hparams.output_directory, '%s.wav' % str(name))
        write_wav(output_path,
                  self.sample_rate,
                  (samples * np.iinfo(np.int16).max).astype(np.int16))
        seconds = float(samples.shape[0]) / self.sample_rate
        return seconds

    def _logmel_to_linear(self, feats):
        """Convert FBANK to linear spectrogram.
        Args:
            feats: FBANK feats, shape: [length, channels]
        Returns:
            linear_feats: Linear spectrogram
        """
        assert feats.shape[1] == self.channels

        linear_feats = np.power(10.0, feats)
        linear_basis = librosa.filters.mel(self.sample_rate,
                                           self.n_fft,
                                           self.channels,
                                           self.lower_frequency_limit,
                                           self.upper_frequency_limit)
        linear_basis = np.linalg.pinv(linear_basis)
        linear_feats = np.maximum(self.EPS, np.dot(linear_basis, linear_feats.T).T)
        return linear_feats

    def _griffin_lim(self, linear_feats, gl_iters):
        """Convert linear spectrogram into waveform

        Args:
            linear_feats: linear spectrogram
            gl_iters: num of gl iterations
        Returns:
            waveform: Reconstructed waveform (N,).

        """
        assert linear_feats.shape[1] == self.n_fft // 2 + 1
        linear_feats = np.abs(linear_feats.T)
        samples = librosa.griffinlim(S=linear_feats,
                                     n_iter=gl_iters,
                                     hop_length=self.hop_length,
                                     win_length=self.window_length,
                                     window=self.window_type)
        return samples

