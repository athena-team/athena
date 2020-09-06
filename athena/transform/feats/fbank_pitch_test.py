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
"""The model tests Fbank&&Pitch FE."""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from athena.transform.feats.read_wav import ReadWav
from athena.transform.feats.fbank_pitch import FbankPitch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class FbankPitchTest(tf.test.TestCase):
    """Fbank && Pitch extraction test."""
    def test_FbankPitch(self):
        wav_path = str(Path(os.environ['MAIN_ROOT']).joinpath('examples/sm1_cln.wav'))

        with self.session():
            read_wav = ReadWav.params().instantiate()
            input_data, sample_rate = read_wav(wav_path)
            config = {'window_length': 0.025, 'output_type': 1, 'frame_length': 0.010, 'dither': 0.0}
            fbank_pitch = FbankPitch.params(config).instantiate()
            fbank_pitch_test = fbank_pitch(input_data, sample_rate)

            if tf.executing_eagerly():
                self.assertEqual(tf.rank(fbank_pitch_test).numpy(), 3)
                print(fbank_pitch_test.numpy()[0:2, :, 0])
            else:
                self.assertEqual(tf.rank(fbank_pitch_test).eval(), 3)
                print(fbank_pitch_test.eval()[0:2, :, 0])

if __name__ == '__main__':

    is_eager = True
    if not is_eager:
        disable_eager_execution()
    else:
        if tf.__version__ < '2.0.0':
            tf.compat.v1.enable_eager_execution()
    tf.test.main()
