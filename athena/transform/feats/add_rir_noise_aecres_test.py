# Copyright (C) ATHENA AUTHORS
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
"""The model tests OP of Add_noise_rir """

import os
from pathlib import Path
import tensorflow as tf
from athena.transform.feats.read_wav import ReadWav
from athena.transform.feats.write_wav import WriteWav
from athena.transform.feats.add_rir_noise_aecres import Add_rir_noise_aecres
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def change_file_path(scp_path, filetype, newfilePath):
  with open(scp_path + filetype, 'r') as f:
    s = f.readlines()
  f.close()
  with open(scp_path + newfilePath, 'w') as f:
    for line in s:
      f.write(scp_path + line)
  f.close()


class AddRirNoiseAecresTest(tf.test.TestCase):
  """
  AddNoiseRIR OP test.
  """

  def test_add_rir_noise_aecres(self):
    wav_path = 'sm1_cln.wav'

    # reset path of noise && rir
    noise_file = 'noiselist.scp'
    rir_file = 'rirlist.scp'

    with self.cached_session(use_gpu=False, force_gpu=False) as sess:
      read_wav = ReadWav.params().instantiate()
      input_data, sample_rate = read_wav(wav_path)
      config = {
          'if_add_noise': True,
          'noise_filelist': noise_file,
          'if_add_rir': True,
          'rir_filelist': rir_file
      }
      add_rir_noise_aecres = Add_rir_noise_aecres.params(config).instantiate()
      add_rir_noise_aecres_test = add_rir_noise_aecres(input_data, sample_rate)
      print('Clean Data:', input_data)
      print('Noisy Data:', add_rir_noise_aecres_test)

      new_noise_file = 'sm1_cln_noisy.wav'
      write_wav = WriteWav.params().instantiate()
      writewav_op = write_wav(new_noise_file, add_rir_noise_aecres_test / 32768,
                              sample_rate)
      sess.run(writewav_op)


if __name__ == '__main__':
  tf.test.main()
