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
"""The model tests CMVN OP."""

import numpy as np
import tensorflow as tf
from athena.transform.feats.cmvn import CMVN


class CMVNTest(tf.test.TestCase):
    """CMVN test."""
    def test_cmvn(self):
        dim = 40
        cmvn = CMVN.params(
            {"mean": np.zeros(dim).tolist(), "variance": np.ones(dim).tolist()}
        ).instantiate()
        audio_feature = tf.random_uniform(shape=[3, 40], dtype=tf.float32, maxval=1.0)
        print(audio_feature)
        normalized = cmvn(audio_feature)
        print("normalized = ", normalized)
        print("dim is ", cmvn.dim())

        cmvn = CMVN.params(
            {
                "mean": np.zeros(dim).tolist(),
                "variance": np.ones(dim).tolist(),
                "cmvn": False,
            }
        ).instantiate()
        normalized = cmvn(audio_feature)
        self.assertAllClose(audio_feature, normalized)


if __name__ == "__main__":
    if tf.__version__ < "2.0.0":
        tf.compat.v1.enable_eager_execution()
    tf.test.main()
