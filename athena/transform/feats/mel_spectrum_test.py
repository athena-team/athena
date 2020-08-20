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
"""The model tests MelSpectrum FE."""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from athena.transform.feats.read_wav import ReadWav
from athena.transform.feats.mel_spectrum import MelSpectrum

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class MelSpectrumTest(tf.test.TestCase):
    """MelSpectrum extraction test."""
    def test_mel_spectrum(self):
        # 16kHz test
        wav_path_16k = str(
            Path(os.environ['MAIN_ROOT']).joinpath('examples/sm1_cln.wav'))

        with self.session():
            # value test
            read_wav = ReadWav.params().instantiate()
            input_data, sample_rate = read_wav(wav_path_16k)
            config = {'type': 'MelSpectrum', 'window_type': 'hann',
                      'upper_frequency_limit': 7600, 'filterbank_channel_count': 80,
                      'lower_frequency_limit': 80, 'dither': 0.0,
                      'window_length': 0.025, 'frame_length': 0.010,
                      'remove_dc_offset': False, 'preEph_coeff': 0.0,
                      'output_type': 3}
            mel_spectrum = MelSpectrum.params(config).instantiate()
            mel_spectrum_test = mel_spectrum(input_data, sample_rate)
            if tf.executing_eagerly():
                print(mel_spectrum_test.numpy()[0:2, 0:10])
            else:
                print(mel_spectrum_test.eval()[0:2, 0:10])


if __name__ == '__main__':

    is_eager = True
    if not is_eager:
        disable_eager_execution()
    else:
        if tf.__version__ < '2.0.0':
            tf.compat.v1.enable_eager_execution()
    tf.test.main()



