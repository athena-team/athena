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
"""The model tests pitch FE."""

import os
from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from athena.transform.feats.read_wav import ReadWav
from athena.transform.feats.pitch import Pitch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class PitchTest(tf.test.TestCase):
    """
    Pitch extraction test.
    """
    def test_pitch(self):
        wav_path_16k = str(
            Path(os.environ['MAIN_ROOT']).joinpath('examples/sm1_cln.wav'))

        with self.session():
            for wav_file in [wav_path_16k]:
                read_wav = ReadWav.params().instantiate()
                input_data, sample_rate = read_wav(wav_file)

                pitch = Pitch.params({
                    'window_length': 0.025,
                    'soft_min_f0': 10.0
                }).instantiate()
                pitch_test = pitch(input_data, sample_rate)

                if tf.executing_eagerly():
                    self.assertEqual(tf.rank(pitch_test).numpy(), 2)
                else:
                    self.assertEqual(tf.rank(pitch_test).eval(), 2)

                output_true = np.array(
                    [
                        [0.03881124, 0.3000031, - 0.02324523],
                        [0.006756478, 0.3000097, 0.01047742],
                        [0.02455365, 0.3000154, 0.00695902],
                        [0.02453586, 0.3000221, 0.008448198],
                        [0.03455311, 0.3000307, - 0.07547269],
                        [0.04293294, 0.3000422, - 0.04193667]
                    ]
                )

                if wav_file == wav_path_16k:
                    if tf.executing_eagerly():
                        print('Transform: ', pitch_test.numpy()[0:6, :])
                        print('kaldi:', output_true)
                        self.assertAllClose(pitch_test.numpy()[0:6, :],
                                            output_true,
                                            rtol=1e-05,
                                            atol=1e-05)
                    else:
                        print('Transform: ', pitch_test.eval())
                        print('kaldi:', output_true)
                        self.assertAllClose(pitch_test.eval()[0:6, :],
                                            output_true,
                                            rtol=1e-05,
                                            atol=1e-05)


if __name__ == "__main__":

    is_eager = True
    if not is_eager:
        disable_eager_execution()
    else:
        if tf.__version__ < "2.0.0":
            tf.compat.v1.enable_eager_execution()
    tf.test.main()
