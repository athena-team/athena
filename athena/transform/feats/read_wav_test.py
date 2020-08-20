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
"""The model tests OP of read_wav """

import os
from pathlib import Path
import librosa
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from athena.transform.feats.read_wav import ReadWav

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class ReadWavTest(tf.test.TestCase):
    """ReadWav OP test."""
    def test_read_wav(self):
        wav_path = str(Path(os.environ['MAIN_ROOT']).joinpath('examples/sm1_cln.wav'))

        with self.session():
            speed = 0.9
            read_wav = ReadWav.params().instantiate()
            input_data, sample_rate = read_wav(wav_path, speed)

            audio_data_true, sample_rate_true = librosa.load(wav_path, sr=16000)
            if (speed == 1.0):
                if tf.executing_eagerly():
                    self.assertAllClose(input_data.numpy() / 32768, audio_data_true)
                    self.assertAllClose(sample_rate.numpy(), sample_rate_true)
                else:
                    self.assertAllClose(input_data.eval() / 32768, audio_data_true)
                    self.assertAllClose(sample_rate.eval(), sample_rate_true)


if __name__ == '__main__':

    is_eager = False
    if not is_eager:
        disable_eager_execution()
    else:
        if tf.__version__ < '2.0.0':
            tf.compat.v1.enable_eager_execution()
    tf.test.main()
