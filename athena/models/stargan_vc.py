# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li
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
from ..loss import CTCLoss
from ..loss import StarganLoss
from ..metrics import CTCAccuracy
from ..metrics import StarganAccuracy
from ..layers.commons import SUPPORTED_RNNS
import numpy as np


def gated_linear_layer(inputs, gates=None, name=None):
    # activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)
    h1_glu = tf.keras.layers.multiply(inputs=[inputs, tf.sigmoid(inputs)], name=name)
    return h1_glu


def InstanceNormalization(x):
    # mean = tf.keras.backend.mean(x, axis=(1,2), keepdims=True)
    # std = tf.keras.backend.std(x, axis=(1,2), keepdims=True)
    # return (x - mean) / (std + 1e-6)
    return tf.keras.layers.BatchNormalization()(x)


def conv2d_layer(inputs, filters, kernel_size, strides, padding: list = None, activation=None, kernel_initializer=None,
                 name=None):
    p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    out = tf.pad(inputs, p, name=name + 'conv2d_pad')
    conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                        activation=activation,
                                        padding="valid")(out)

    return conv_layer


def downsample2d_block(inputs, filters, kernel_size, strides, padding, name_prefix):
    h1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                padding="same")(inputs)
    h1_norm = InstanceNormalization(h1)
    h1_gates = conv2d_layer(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None,
        name=name_prefix + 'h1_gates')
    h1_norm_gates = InstanceNormalization(h1_gates)
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'downsample_h1_glu')
    # c2 = tf.tile(spk, [1, h1_glu.shape[1], h1_glu.shape[2], 1])
    # h1_glu = tf.concat([h1_glu, c2], axis=-1)
    return h1_glu

class DownSampleBlock(tf.keras.layers.Layer):
    """Down-sampling layers."""
    def __init__(self, filters, kernel_size, strides):
        super(DownSampleBlock, self).__init__()

        self.conv1= tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.norm1 = tf.keras.layers.BatchNormalization(epsilon=1e-8)  #, input_shape=(d_model,))
        self.norm2 = tf.keras.layers.BatchNormalization(epsilon=1e-8)  # , input_shape=(d_model,))

        # self.conv_layer = tf.keras.Model(inputs=inner, outputs=c1_red, name="discriminator_net")
    def call(self, x, padding=None):
        h1 = self.conv1(x)
        h1_norm=self.norm1(h1)
        # h1_norm=InstanceNormalization(h1)
        # p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
        # hg = tf.pad(x, p)
        # hg = self.conv2(x)
        # h1_gates = self.norm1(hg)
        # h1_gates = InstanceNormalization(hg)
        # print("h1_norm:", h1_norm.shape.as_list() , h1_gates.shape.as_list())
        h1_glu = gated_linear_layer(inputs=h1_norm)#, gates=h1_gates)
        return h1_glu

def upsample2d_block(inputs, filters, kernel_size, strides, name_prefix):
    t1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
    t2 = InstanceNormalization(t1)
    x1_gates = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
    x1_norm_gates = InstanceNormalization(x1_gates)
    x1_glu = gated_linear_layer(t2, x1_norm_gates)
    return x1_glu

class UpSampleBlock(tf.keras.layers.Layer):
    """Down-sampling layers."""
    def __init__(self, filters, kernel_size, strides):
        super(UpSampleBlock, self).__init__()
        # p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
        # inner = tf.pad(inputs, p)
        self.conv1= tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        # self.conv_layer = tf.keras.Model(inputs=[inner, spk], outputs=c1_red, name="discriminator_net")
    def call(self, x):
        h1 = self.conv1(x)
        h1_norm=InstanceNormalization(h1)
        # p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
        # hg = tf.pad(x, p)
        # hg = self.conv2(x)
        # h1_gates = InstanceNormalization(hg)
        h1_glu = gated_linear_layer(inputs=h1_norm) #, gates=h1_gates)
        return h1_glu


class generator_gatedcnn(tf.keras.layers.Layer):
    '''(inputs, speaker_id=None, reuse=False, scope_name='generator_gatedcnn'):
    #input shape [batchsize, h, w, c]
    #speaker_id [batchsize, one_hot_vector]
    #one_hot_vector：[0,1,0,0]'''

    def __init__(self, reuse=False, speaker_num=2, data_descriptions=None,scope_name='generator_gatedcnn'):
        super(generator_gatedcnn, self).__init__()
        self.reuse = reuse
        self.downsample_1 = DownSampleBlock(filters=32, kernel_size=[3, 9], strides=[1, 1])
        self.downsample_2 = DownSampleBlock(filters=64, kernel_size=[4, 8], strides=[2, 2])
        self.downsample_3 = DownSampleBlock(filters=128, kernel_size=[4, 8], strides=[2, 2])
        self.downsample_4 = DownSampleBlock(filters=64, kernel_size=[3, 5], strides=[1, 1])
        self.downsample_5 = DownSampleBlock(filters=5, kernel_size=[9, 5], strides=[9, 1])

        self.upsample_1 = UpSampleBlock(filters=64, kernel_size=[9, 5], strides=[9, 1])
        self.upsample_2 = UpSampleBlock(filters=128, kernel_size=[3, 5], strides=[1, 1])
        self.upsample_3 = UpSampleBlock(filters=64, kernel_size=[4, 8], strides=[2, 2])
        self.upsample_4 = UpSampleBlock(filters=32, kernel_size=[4, 8], strides=[2, 2])


    def call(self, inputs, spk=None):
        print("inputs:", inputs.shape.as_list())
        d1 = self.downsample_1(inputs, padding=[1, 4])
        d2 = self.downsample_2(d1, padding=[1, 3])
        d3 = self.downsample_3(d2, padding=[1, 3])
        d4 = self.downsample_4(d3, padding=[1, 2])
        d5 = self.downsample_5(d4, padding=[1, 2])

        spk = tf.convert_to_tensor(spk, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(spk, [-1, 1, 1, spk.shape.dims[-1].value]), tf.float32)
        c = tf.tile(c_cast, [1, d5.shape.dims[1].value, d5.shape.dims[2].value, 1])
        # concated = tf.concat([d5, c], axis=-1)
        concated = tf.keras.layers.concatenate(inputs=[d5, c], axis=-1)

        u1 = self.upsample_1(concated)
        c1 = tf.tile(c_cast, [1, u1.shape.dims[1].value, u1.shape.dims[2].value, 1])
        # print(f'c1 shape: {c1.shape}')
        # u1_concat = tf.concat([u1, c1], axis=-1)
        u1_concat = tf.keras.layers.concatenate(inputs=[u1, c1], axis=-1)

        # u2 = upsample2d_block(u1_concat, 128, [3, 5], [1, 1], name_prefix='gen_up_u2')
        u2 = self.upsample_2(u1_concat)
        # print(f'u2.shape :{u2.shape.as_list()}')
        c2 = tf.tile(c_cast, [1, u2.shape[1], u2.shape[2], 1])
        # u2_concat = tf.concat([u2, c2], axis=-1)
        u2_concat = tf.keras.layers.concatenate(inputs=[u2, c2], axis=-1)

        u3 = self.upsample_3(u2_concat)
        # print(f'u3.shape :{u3.shape.as_list()}')
        c3 = tf.tile(c_cast, [1, u3.shape[1], u3.shape[2], 1])
        # u3_concat = tf.concat([u3, c3], axis=-1)
        u3_concat = tf.keras.layers.concatenate(inputs=[u3, c3], axis=-1)

        u4 = self.upsample_4(u3_concat)
        c4 = tf.tile(c_cast, [1, u4.shape[1], u4.shape[2], 1])
        u4_concat = tf.keras.layers.concatenate(inputs=[u4, c4], axis=-1)

        u5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 9], strides=[1, 1], padding='same',
                                             name='generator_last_deconv')(u4_concat)
        print("u5:", u5.shape.as_list())
        return u5

class discriminator(tf.keras.layers.Layer):
    def __init__(self, reuse=False, speaker_num=2, data_descriptions=None, scope_name='discriminator'):
        super(discriminator,self).__init__()
        self.reuse = reuse    # def discriminator( inputs, speaker_id=None):
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

        # print("c1_shape:",d3_concat.shape.as_list())
        d4 = self.downsample_4(d3_concat, padding=[1, 2])
        c4 = tf.tile(c_cast, [1, d4.shape[1], d4.shape[2], 1])
        d4_concat = tf.concat([d4, c4], axis=-1)

        c1 = conv2d_layer(inputs=d4_concat, filters=1, kernel_size=[36, 5], strides=[36, 1], padding=[0, 1],
                          activation="sigmoid", name='discriminator-last-conv')

        # print("c1_shape:",c1.shape.as_list())
        out = tf.reduce_mean(c1, keepdims=True)
        return out


class domain_classifier(tf.keras.layers.Layer):
    def __init__(self, batchsize=1, reuse=False, speaker_num=2, data_descriptions=None,scope_name='classifier'):
        super(domain_classifier, self).__init__()
        self.scope_name = scope_name
        # todo maxpooling
        inner = tf.keras.layers.Input(shape=[8, None, 1])
        d1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(inner)
        d2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(d1)
        d3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(d2)
        d4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 4), strides=(1, 2), activation="relu",
                                    padding="same")(d3)
        d5 = tf.keras.layers.Conv2D(filters=speaker_num, kernel_size=(1, 4), strides=(1, 2), activation="relu",
                                    padding="same")(d4)
        p = tf.keras.layers.GlobalAveragePooling2D()(d5)
        # print(p.get_shape())
        a, dim = p.get_shape().as_list()
        # o_r = tf.reshape(p, [-1, 1, 1, p.shape.dims[1].value])
        o_r = tf.keras.layers.Reshape([-1, 1, dim])(p)
        # _, _, dim, channels = inner.get_shape().as_list()
        # output_dim = dim * channels
        # inner = layers.Reshape((-1, output_dim))(inner)
        self.x_net = tf.keras.Model(inputs=inner, outputs=o_r, name="x_net")
        # print(self.x_net.summary())

    def call(self, inputs):
        # Take the first 8 dimensional coded_sp to domain classifier
        slice = inputs[:, 0:8, :, :]
        x0 = self.x_net(slice)

        return x0

class StarGANVC(tf.keras.layers.Layer):
    default_config={}
    def __init__(self,
                 num_features=36,
                 # frames=512,
                 data_descriptions = None,
                 batch_size=1,
                 sample_signature = None,
                 config = None,
                 speaker_num=2
    ):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.sample_signature = sample_signature
        self.num_features = num_features
        self.speaker_num = speaker_num
        self.batch_size = batch_size
        self.vcinput_shape = data_descriptions.sample_shape["src_coded_sp"]     #[ num_features, None, 1]
        # data_descriptions.sample_shape["input"]
        self.label_shape = data_descriptions.sample_shape["src_speaker"]  #[ self.speaker_num]
        self.classifier = domain_classifier(data_descriptions=data_descriptions, speaker_num=self.speaker_num)
        self.discriminator = discriminator(data_descriptions=data_descriptions, speaker_num=self.speaker_num)
        self.generator = generator_gatedcnn(data_descriptions=data_descriptions, speaker_num=self.speaker_num)
        layers = tf.keras.layers
        # input_real = layers.Input(shape=data_descriptions.sample_shape["src_coded_sp"], name='input_real')
        # target_real = layers.Input(shape=data_descriptions.sample_shape["src_coded_sp"], name='target_real')
        # target_label = layers.Input(shape=data_descriptions.sample_shape["src_speaker"], name='target_label')
        # source_label = layers.Input(shape = data_descriptions.sample_shape["src_speaker"], name='source_label')
        #
        # target_label_reshaped = layers.Input(shape=[1, 1, self.speaker_num],
        #                                             name='reshaped_label_for_classifier')
        # self.domain_out_real = self.classifier(target_real)
        # self.C_net = tf.keras.Model(inputs=[target_real],outputs=[self.domain_out_real],name='C_net')
        #
        # self.generated_forward = self.generator(input_real, target_label)
        # self.discirmination = self.discriminator(self.generated_forward, target_label)
        # self.generated_back = self.generator(self.generated_forward, source_label)
        # self.domain_out_real = self.classifier(target_real)
        # # Identity loss
        # self.identity_map = self.generator(input_real, source_label)
        # self.G_net = tf.keras.Model(inputs=[input_real, target_real,target_label,source_label,
        # target_label_reshaped], outputs=[self.discirmination,self.generated_forward,
        # self.generated_back,self.domain_out_real,self.identity_map], name="G_net")
        # print(self.G_net.summary())
        #
        # # discrimination_real, discirmination_fake, generated_forward, target_label_reshaped, domain_out_fake = inputs
        # # discirmination
        # self.discrimination_real = self.discriminator(target_real, target_label)
        # self.discirmination_fake = self.discriminator(self.generated_forward, target_label)
        # self.domain_out_fake = self.classifier(self.generated_forward)
        # # calculate `x_hat`
        #
        # # epsilon = tf.random.uniform((self.batchsize, 1, 1, 1), 0.0, 1.0)
        # # x_hat = epsilon * self.generated_forward + (1.0 - epsilon) * input_real
        # # with tf.GradientTape(persistent=True) as tape:
        # #     tmp = self.discriminator(x_hat, target_label)
        # # gradients = tape.gradient(tmp, x_hat)
        #
        # self.D_net = tf.keras.Model(inputs=[input_real, target_label,target_real], outputs=
        # [self.discrimination_real, self.discirmination_fake, self.domain_out_fake], name="D_net")
        # print(self.D_net.summary())

    def call(self, samples, training: bool = None):
        # print("calling!!!!!!!!")
        input_real, target_real, target_label, source_label = samples["src_coded_sp"], \
                                                              samples["tar_coded_sp"], samples["tar_speaker"], samples[
                                                                  "src_speaker"]

        target_label_reshaped = tf.reshape(target_label, [target_label.shape[0], 1, 1, self.speaker_num])
        # target_label = tf.reshape(target_label, [1, 2], dtype=tf.float32)
        # source_label = tf.reshape(source_label, [1, 2], dtype=tf.float32)  # self.batch_size, self.speaker_num

        self.domain_out_real = self.classifier(target_real)

        self.generated_forward = self.generator(input_real, target_label)
        self.discirmination = self.discriminator(self.generated_forward, target_label)
        self.generated_back = self.generator(self.generated_forward, source_label)
        self.domain_out_real = self.classifier(target_real)
        # Identity loss
        self.identity_map = self.generator(input_real, source_label)

        # discrimination_real, discirmination_fake, generated_forward, target_label_reshaped, domain_out_fake = inputs
        # discirmination
        self.discrimination_real = self.discriminator(target_real, target_label)
        self.discirmination_fake = self.discriminator(self.generated_forward, target_label)
        self.domain_out_fake = self.classifier(self.generated_forward)

        return {"domain_classifier": domain_classifier, "discirmination": self.discirmination,
                "generated_forward": self.generated_forward, "generated_back": self.generated_back, \
                "domain_out_real": self.domain_out_real, "identity_map": self.identity_map,
                "discrimination_real": self.discrimination_real, "discirmination_fake": self.discirmination_fake, \
                "domain_out_fake": self.domain_out_fake, "target_label_reshaped": target_label_reshaped
                }
        # print("calling!!!!!!!!")
        # input_real, target_real, target_label, source_label= samples["src_coded_sp"],\
        # samples["tar_coded_sp"],samples["tar_speaker"],samples["src_speaker"]
        #
        # target_label_reshaped = tf.reshape(target_label, [target_label.shape[0], 1, 1, self.speaker_num])
        # target_label = tf.reshape(target_label,[1,2], dtype=tf.float32)
        # source_label = tf.reshape(source_label, [1,2], dtype=tf.float32)   #self.batch_size, self.speaker_num
        #
        # domain_classifier = self.C_net(target_real, training=training)
        # self.discirmination, self.generated_forward, self.generated_back, self.domain_out_real, self.identity_map = \
        #     self.G_net([input_real, target_real, target_label, source_label, target_label_reshaped], training=training)
        #
        # self.discrimination_real, self.discirmination_fake, self.domain_out_fake = self.D_net([input_real,
        # target_label,target_real],training=training)
        #
        # return {"domain_classifier": domain_classifier, "discirmination": self.discirmination,
        #         "generated_forward": self.generated_forward, "generated_back":self.generated_back, \
        #        "domain_out_real": self.domain_out_real, "identity_map": self.identity_map,
        #         "discrimination_real": self.discrimination_real, "discirmination_fake": self.discirmination_fake, \
        #        "domain_out_fake": self.domain_out_fake,"target_label_reshaped": target_label_reshaped
        #         }

class StarganModel(BaseModel):
    """ a sample implementation of CTC model """
    default_config = {
        "conv_filters": 256,
        "rnn_hidden_size": 1024,
        "num_rnn_layers": 6,
        "rnn_type": "gru"
    }
    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.loss_function = StarganLoss()
        self.metric = StarganAccuracy(name="StarganAccuracy")
        # self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.StarGAN = StarGANVC(data_descriptions=data_descriptions,speaker_num=2)

    def call(self, samples, training=None):
        """ call function """
        # return self.net(samples["input"], training=training)
        return self.StarGAN(samples)

    def compute_logit_length(self, samples):
        """ used for get logit length """
        input_length = tf.cast(samples["input_length"], tf.float32)
        logit_length = tf.math.ceil(input_length / 2)
        logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length

if __name__ == '__main__':
    vcinput_shape = [1,36, 512, 1]  # [None, num_features, frames, 1]
    label_shape = [1,2]
    resha_label = [1,1,1,2]
    samples ={"input_real":tf.ones(vcinput_shape,dtype="float32"),
              "target_real":tf.ones(vcinput_shape,dtype="float32"),
              "target_label":tf.ones(label_shape,dtype="float32"),
              "source_label": tf.ones(label_shape, dtype="float32"),
              "target_label_reshaped": tf.ones(label_shape, dtype="float32")
              }
    input_real, target_real, target_label, source_label, target_label_reshaped = samples["input_real"], \
                                                                                 samples["target_real"], samples[
                                                                                     "target_label"], samples[
                                                                                     "source_label"], samples[
                                                                                     "target_label_reshaped"]
    starganvc = StarGANVC(36)
    a,b,c,d,e,f,g,h,ij = starganvc(samples)
    print(a)
    print(b)