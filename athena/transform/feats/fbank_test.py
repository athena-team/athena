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
"""The model tests OP of fbank. """

import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from athena.transform.feats.read_wav import ReadWav
from athena.transform.feats.fbank import Fbank

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class FbankTest(tf.test.TestCase):
    """Test OP of Fbank."""
    def test_fbank(self):
        """Test Fbank using 16kHz && 8kHz wav."""
        wav_path_16k = str(
            Path(os.environ["MAIN_ROOT"]).joinpath("examples/sm1_cln.wav")
        )
        wav_path_8k = str(
            Path(os.environ["MAIN_ROOT"]).joinpath("examples/english.wav")
        )

        with self.session():
            # value test
            read_wav = ReadWav.params().instantiate()
            input_data, sample_rate = read_wav(wav_path_16k)
            fbank = Fbank.params({"delta_delta": False}).instantiate()
            fbank_test = fbank(input_data, sample_rate)
            real_fank_feats = np.array(
                [
                    [3.768338, 4.946218, 6.289874, 6.330853, 6.761764, 6.884573],
                    [3.803553, 5.450971, 6.547878, 5.796172, 6.397846, 7.242926],
                ]
            )
            # self.assertAllClose(np.squeeze(fbank_test.eval()[0:2, 0:6, 0]),
            #                     real_fank_feats, rtol=1e-05, atol=1e-05)
            if tf.executing_eagerly():
                print(fbank_test.numpy()[0:2, 0:6, 0])
            else:
                print(fbank_test.eval()[0:2, 0:6, 0])
            count = 1

            for wav_file in [wav_path_8k, wav_path_16k]:

                read_wav = ReadWav.params().instantiate()
                input_data, sample_rate = read_wav(wav_file)
                if tf.executing_eagerly():
                    print(wav_file, sample_rate.numpy())
                else:
                    print(wav_file, sample_rate.eval())

                conf = {
                    "delta_delta": True,
                    "lower_frequency_limit": 100,
                    "upper_frequency_limit": 0,
                }
                fbank = Fbank.params(conf).instantiate()
                fbank_test = fbank(input_data, sample_rate)
                if tf.executing_eagerly():
                    print(fbank_test.numpy())
                else:
                    print(fbank_test.eval())
                print(fbank.num_channels())

                conf = {
                    "delta_delta": False,
                    "lower_frequency_limit": 100,
                    "upper_frequency_limit": 0,
                }
                fbank = Fbank.params(conf).instantiate()
                fbank_test = fbank(input_data, sample_rate)
                print(fbank_test)
                print(fbank.num_channels())
                count += 1
                del read_wav
                del fbank


if __name__ == "__main__":

    is_eager = True
    if not is_eager:
        disable_eager_execution()
    else:
        if tf.__version__ < "2.0.0":
            tf.compat.v1.enable_eager_execution()
    tf.test.main()
