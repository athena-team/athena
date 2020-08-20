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
"""The model tests framepow FE."""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from athena.transform.feats.read_wav import ReadWav
from athena.transform.feats.framepow import Framepow

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class FramePowTest(tf.test.TestCase):
    """Framepow extraction test."""
    def test_framepow(self):
        wav_path_16k = str(
            Path(os.environ["MAIN_ROOT"]).joinpath("examples/sm1_cln.wav")
        )

        with self.session():
            read_wav = ReadWav.params().instantiate()
            input_data, sample_rate = read_wav(wav_path_16k)
            config = {"snip_edges": 1}
            framepow = Framepow.params(config).instantiate()
            framepow_test = framepow(input_data, sample_rate)

            real_framepow_feats = np.array(
                [9.819611, 9.328745, 9.247337, 9.26451, 9.266059]
            )

            if tf.executing_eagerly():
                self.assertAllClose(
                    framepow_test.numpy()[0:5],
                    real_framepow_feats,
                    rtol=1e-05,
                    atol=1e-05,
                )
                print(framepow_test.numpy()[0:5])
            else:
                self.assertAllClose(
                    framepow_test.eval()[0:5],
                    real_framepow_feats,
                    rtol=1e-05,
                    atol=1e-05,
                )
                print(framepow_test.eval()[0:5])


if __name__ == "__main__":
    is_eager = True
    if not is_eager:
        disable_eager_execution()
    else:
        if tf.__version__ < "2.0.0":
            tf.compat.v1.enable_eager_execution()
    tf.test.main()
