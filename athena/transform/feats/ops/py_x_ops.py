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
""" python custom ops """
import tensorflow.compat.v1 as tf
from absl import logging

so_lib_file = tf.io.gfile.glob(tf.resource_loader.get_data_files_path() + "/x_ops*.so")[
    0
].split("/")[-1]
path = tf.resource_loader.get_path_to_datafile(so_lib_file)
logging.info("x_ops*.so path:{}".format(path))

gen_x_ops = tf.load_op_library(tf.resource_loader.get_path_to_datafile(so_lib_file))

spectrum = gen_x_ops.spectrum
mel_spectrum = gen_x_ops.mel_spectrum
fbank = gen_x_ops.fbank
delta_delta = gen_x_ops.delta_delta
pitch = gen_x_ops.pitch
mfcc = gen_x_ops.mfcc_dct
frame_pow = gen_x_ops.frame_pow
speed = gen_x_ops.speed
