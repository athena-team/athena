# Copyright (C) 2019 ATHENA AUTHORS; Xiaochunxin; Jiangdongwei
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
# Only support eager mode and TF>=2.0.0
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes
""" a implementation of deep speech 2 model can be used as a sample for ctc model """

import tensorflow as tf
from absl import logging
from ..utils.hparam import register_and_parse_hparams
from .base import BaseModel
from ..loss import StarganLoss
from ..metrics import StarganAccuracy
from ..layers.commons import SUPPORTED_RNNS


def gated_linear_layer(inputs, name=None):
    h1_glu = tf.keras.layers.multiply(inputs=[inputs, tf.sigmoid(inputs)], name=name)
    return h1_glu

def conv2d_layer(inputs, filters, kernel_size, strides, padding: list = None, activation=None):
    p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    out = tf.pad(inputs, p)
    conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                        activation=activation,
                                        padding="valid")(out)
    return conv_layer

class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DownSampleBlock, self).__init__()

        self.conv1= tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.norm1 = tf.keras.layers.BatchNormalization(epsilon=1e-8)

    def call(self, x,padding=None):
        h1 = self.conv1(x)
        h1_norm=self.norm1(h1)
        h1_glu = gated_linear_layer(inputs=h1_norm)
        return h1_glu


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(UpSampleBlock, self).__init__()
        self.conv1= tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.norm1 = tf.keras.layers.BatchNormalization(epsilon=1e-8)

    def call(self, x):
        h1 = self.conv1(x)
        h1_norm=self.norm1(h1)
        h1_glu = gated_linear_layer(inputs=h1_norm)
        return h1_glu


class generator(tf.keras.layers.Layer):
    def __init__(self):
        super(generator, self).__init__()

        inner = tf.keras.layers.Input(shape=[36, None, 1], dtype=tf.float32)
        inner1 = DownSampleBlock(filters=32, kernel_size=[3, 9], strides=[1, 1])(inner)
        inner2 = DownSampleBlock(filters=64, kernel_size=[4, 8], strides=[2, 2])(inner1)
        inner3 = DownSampleBlock(filters=128, kernel_size=[4, 8], strides=[2, 2])(inner2)
        inner4 = DownSampleBlock(filters=64, kernel_size=[3, 5], strides=[1, 1])(inner3)
        inner5 = DownSampleBlock(filters=5, kernel_size=[9, 5], strides=[9, 1])(inner4)
        self.conv_layer = tf.keras.Model(inputs=inner, outputs=inner5)

        self.upsample_1 = UpSampleBlock(filters=64, kernel_size=[9, 5], strides=[9, 1])
        self.upsample_2 = UpSampleBlock(filters=128, kernel_size=[3, 5], strides=[1, 1])
        self.upsample_3 = UpSampleBlock(filters=64, kernel_size=[4, 8], strides=[2, 2])
        self.upsample_4 = UpSampleBlock(filters=32, kernel_size=[4, 8], strides=[2, 2])


    def call(self, inputs, spk):
        d5 = self.conv_layer(inputs)

        spk = tf.convert_to_tensor(spk, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(spk, [-1, 1, 1, spk.shape.dims[-1].value]), tf.float32)
        c = tf.tile(c_cast, [1, d5.shape.dims[1].value, d5.shape.dims[2].value, 1])
        concated = tf.keras.layers.concatenate(inputs=[d5, c], axis=-1)

        u1 = self.upsample_1(concated)
        c1 = tf.tile(c_cast, [1, u1.shape.dims[1].value, u1.shape.dims[2].value, 1])
        u1_concat = tf.keras.layers.concatenate(inputs=[u1, c1], axis=-1)

        u2 = self.upsample_2(u1_concat)
        c2 = tf.tile(c_cast, [1, u2.shape[1], u2.shape[2], 1])
        u2_concat = tf.keras.layers.concatenate(inputs=[u2, c2], axis=-1)

        u3 = self.upsample_3(u2_concat)
        c3 = tf.tile(c_cast, [1, u3.shape[1], u3.shape[2], 1])
        u3_concat = tf.keras.layers.concatenate(inputs=[u3, c3], axis=-1)

        u4 = self.upsample_4(u3_concat)
        c4 = tf.tile(c_cast, [1, u4.shape[1], u4.shape[2], 1])
        u4_concat = tf.keras.layers.concatenate(inputs=[u4, c4], axis=-1)

        u5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 9], strides=[1, 1], padding='same',
                                             name='generator_last_deconv')(u4_concat)
        return u5

class discriminator(tf.keras.layers.Layer):
    def __init__(self, data_descriptions=None):
        super(discriminator,self).__init__()
        self.downsample_1 = DownSampleBlock(filters=32, kernel_size=[3, 9], strides=[1, 1])
        self.downsample_2 = DownSampleBlock(filters=32, kernel_size=[3, 8], strides=[1, 2])
        self.downsample_3 = DownSampleBlock(filters=32, kernel_size=[3, 8], strides=[1, 2])
        self.downsample_4 = DownSampleBlock(filters=32, kernel_size=[3, 6], strides=[1, 2])

    def call(self, inner, spk=None):
        c_cast = tf.cast(tf.reshape(spk, [-1, 1, 1, spk.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, inner.shape[1], inner.shape[2], 1])
        concated = tf.concat([inner, c], axis=-1)
        d1 = self.downsample_1(concated, padding=[1, 4])
        c1 = tf.tile(c_cast, [1, d1.shape[1], d1.shape[2], 1])
        d1_concat = tf.concat([d1, c1], axis=-1)

        d2 = self.downsample_2(d1_concat, padding=[1, 3])
        c2 = tf.tile(c_cast, [1, d2.shape[1], d2.shape[2], 1])
        d2_concat = tf.concat([d2, c2], axis=-1)

        d3 = self.downsample_3(d2_concat, padding=[1, 3])
        c3 = tf.tile(c_cast, [1, d3.shape[1], d3.shape[2], 1])
        d3_concat = tf.concat([d3, c3], axis=-1)

        d4 = self.downsample_4(d3_concat, padding=[1, 2])
        c4 = tf.tile(c_cast, [1, d4.shape[1], d4.shape[2], 1])
        d4_concat = tf.concat([d4, c4], axis=-1)

        c1 = conv2d_layer(inputs=d4_concat, filters=1, kernel_size=[36, 5], strides=[36, 1], padding=[0, 1],
                          activation="sigmoid")

        out = tf.reduce_mean(c1, keepdims=True)
        return out


class domain_classifier(tf.keras.layers.Layer):
    def __init__(self, speaker_num=2):
        super(domain_classifier, self).__init__()
        inner = tf.keras.layers.Input(shape=[8, None, 1])

        d1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(inner)
        d1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(d1)
        d2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(d1)
        d2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(d2)
        d3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(d2)
        d3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(d3)
        d4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 4), strides=(1, 2), activation="relu",
                                    padding="same")(d3)
        d4 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='same')(d4)
        d5 = tf.keras.layers.Conv2D(filters=speaker_num, kernel_size=(1, 4), strides=(1, 2), activation="relu",
                                    padding="same")(d4)
        d5 = tf.keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2), padding='same')(d5)

        p = tf.keras.layers.GlobalAveragePooling2D()(d5)

        a, dim = p.get_shape().as_list()
        o_r = tf.keras.layers.Reshape([-1, 1, dim])(p)
        self.x_net = tf.keras.Model(inputs=inner, outputs=o_r, name="x_net")

    def call(self, inputs):
        # Take the first 8 dimensional coded_sp to domain classifier
        slice = inputs[:, 0:8, :, :]
        x0 = self.x_net(slice)

        return x0

class StarganModel(BaseModel):
    default_config = {
        "codedsp_dim":36,
        "speaker_num": 2,
        "lambda_cycle": 10,
        "lambda_identity": 5,
        "lambda_classifier": 3,
    }
    def __init__(self,data_descriptions, config = None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.num_features = self.hparams.codedsp_dim
        self.speaker_num = data_descriptions.speaker_num
        self.classifier = domain_classifier()
        self.discriminator = discriminator()
        self.generator = generator()
        self.loss_function = StarganLoss()
        self.metric = StarganAccuracy(name="StarganAccuracy")

    def call(self, samples, training: bool = None):
        input_real, target_real, target_label, source_label = samples["src_coded_sp"], samples["tar_coded_sp"], \
                                                              samples["tar_speaker"], samples["src_speaker"]

        target_label_reshaped = tf.reshape(target_label, [target_label.shape[0], 1, 1, self.speaker_num])

        # Classifier training process
        self.domain_out_real = self.classifier(target_real)

        # Generator training process
        self.generated_forward = self.generator(input_real, target_label)
        self.discirmination = self.discriminator(self.generated_forward, target_label)
        self.generated_back = self.generator(self.generated_forward, source_label)
        # self.domain_out_real = self.classifier(target_real)
        self.identity_map = self.generator(input_real, source_label)

        # Discriminator training process
        self.discrimination_real = self.discriminator(target_real, target_label)
        self.discirmination_fake = self.discriminator(self.generated_forward, target_label)
        self.domain_out_fake = self.classifier(self.generated_forward)

        return {
                "generated_forward": self.generated_forward, "generated_back": self.generated_back,
                "domain_out_real": self.domain_out_real, "identity_map": self.identity_map,
                "discrimination_real": self.discrimination_real, "discirmination_fake": self.discirmination_fake,
                "domain_out_fake": self.domain_out_fake, "target_label_reshaped": target_label_reshaped,
                "discirmination": self.discirmination
                }


    def convert(self, src_coded_sp, src_speaker):
        """
        Convert acoustic features
        Args:
            samples: the data source to be convert
        Returns:
            tar_coded_sp: the converted acoustic features
        """
        tar_coded_sp = self.generator(src_coded_sp, src_speaker)

        return tar_coded_sp