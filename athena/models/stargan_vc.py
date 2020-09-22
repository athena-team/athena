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
""" an implementation of stargan for voice conversion """

import tensorflow as tf
from ..utils.hparam import register_and_parse_hparams
from ..utils.learning_rate import  WarmUpAdam
from .base import BaseModel
from ..loss import StarganLoss
from ..layers.commons import DownSampleBlock, UpSampleBlock


class Generator(BaseModel):
    """ generator for stargan
    """
    def __init__(self, optimizer_config=None):
        super(Generator, self).__init__()
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

        self.tran_conv = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 9], strides=[1, 1], padding='same',
                                        name='generator_last_deconv')
        self.optimizer = WarmUpAdam(config=optimizer_config)

    def call(self, inputs, spk):
        d5 = self.conv_layer(inputs)

        spk = tf.convert_to_tensor(spk, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(spk, [-1, 1, 1, tf.shape(spk)[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, tf.shape(d5)[1], tf.shape(d5)[2], 1])
        concated = tf.concat([d5, c], axis=-1)

        u1 = self.upsample_1(concated)
        c1 = tf.tile(c_cast, [1, tf.shape(u1)[1], tf.shape(u1)[2], 1])
        u1_concat = tf.concat([u1, c1], axis=-1)

        u2 = self.upsample_2(u1_concat)
        c2 = tf.tile(c_cast, [1, tf.shape(u2)[1], tf.shape(u2)[2], 1])
        u2_concat = tf.concat([u2, c2], axis=-1)

        u3 = self.upsample_3(u2_concat)
        c3 = tf.tile(c_cast, [1, tf.shape(u3)[1], tf.shape(u3)[2], 1])
        u3_concat = tf.concat([u3, c3], axis=-1)

        u4 = self.upsample_4(u3_concat)
        c4 = tf.tile(c_cast, [1, tf.shape(u4)[1], tf.shape(u4)[2], 1])
        u4_concat = tf.concat([u4, c4], axis=-1)

        u5 = self.tran_conv(u4_concat)
        return u5

class Discriminator(BaseModel):
    """ discriminator for stargan
    """
    def __init__(self, optimizer_config=None):
        super(Discriminator, self).__init__()
        self.downsample_1 = DownSampleBlock(filters=32, kernel_size=[3, 9], strides=[1, 1])
        self.downsample_2 = DownSampleBlock(filters=32, kernel_size=[3, 8], strides=[1, 2])
        self.downsample_3 = DownSampleBlock(filters=32, kernel_size=[3, 8], strides=[1, 2])
        self.downsample_4 = DownSampleBlock(filters=32, kernel_size=[3, 6], strides=[1, 2])
        self.discriminator_conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=[36, 5], strides=[36, 1],
                                            activation="sigmoid", padding="same")
        self.optimizer = WarmUpAdam(config=optimizer_config)

    def call(self, inner, spk=None):
        c_cast = tf.cast(tf.reshape(spk, [-1, 1, 1, spk.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, tf.shape(inner)[1], tf.shape(inner)[2], 1])
        concated = tf.concat([inner, c], axis=-1)
        d1 = self.downsample_1(concated)
        c1 = tf.tile(c_cast, [1, tf.shape(d1)[1], tf.shape(d1)[2], 1])
        d1_concat = tf.concat([d1, c1], axis=-1)

        d2 = self.downsample_2(d1_concat)
        c2 = tf.tile(c_cast, [1, tf.shape(d2)[1], tf.shape(d2)[2], 1])
        d2_concat = tf.concat([d2, c2], axis=-1)

        d3 = self.downsample_3(d2_concat)
        c3 = tf.tile(c_cast, [1, tf.shape(d3)[1], tf.shape(d3)[2], 1])
        d3_concat = tf.concat([d3, c3], axis=-1)

        d4 = self.downsample_4(d3_concat)
        c4 = tf.tile(c_cast, [1, tf.shape(d4)[1], tf.shape(d4)[2], 1])
        d4_concat = tf.concat([d4, c4], axis=-1)

        c1 = self.discriminator_conv_layer(d4_concat)

        out = tf.reduce_mean(c1, keepdims=True)
        return out


class Classifier(BaseModel):
    """ classifier for stargan
    """
    def __init__(self, speaker_num=2, optimizer_config=None):
        super(Classifier, self).__init__()
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

        a, dim = tf.shape(p)[0], tf.shape(p)[1]
        o_r = tf.reshape(p, [-1, 1, dim])
        self.x_net = tf.keras.Model(inputs=inner, outputs=o_r)
        self.optimizer = WarmUpAdam(config=optimizer_config)

    def call(self, inputs):
        # Take the first 8 dimensional coded_sp to domain classifier
        slice = tf.gather(inputs, axis=1, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        x0 = self.x_net(slice)
        return x0


class StarganModel(BaseModel):
    """ definination for stargan model, it consists of generator, discriminator and classifier
    """
    default_config = {
        "codedsp_dim":36,
        "lambda_cycle": 10,
        "lambda_identity": 5,
        "lambda_classifier": 3,
        "optimizer_config": {
            "d_model": 512,
            "warmup_steps": 1244,
            "k": 0.12,
            "decay_steps": 100000,
            "decay_rate": 0.1
        }
    }
    def __init__(self, data_descriptions, config=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.num_features = self.hparams.codedsp_dim
        self.batchsize = 1
        # speaker number will automatic count from data
        self.speaker_num = data_descriptions.spk_num
        self.classifier = Classifier(speaker_num=self.speaker_num, optimizer_config=self.hparams.optimizer_config)
        self.discriminator = Discriminator()
        self.generator = Generator()
        # Used for weight calculation of Generator loss function
        self.lambda_cycle = self.hparams.lambda_cycle
        self.lambda_identity = self.hparams.lambda_identity
        self.lambda_classifier = self.hparams.lambda_classifier
        self.loss_function = StarganLoss(self.lambda_cycle, self.lambda_identity, self.lambda_classifier)
        self.metric_c = tf.keras.metrics.Mean(name="metrics_c")
        self.metric_g = tf.keras.metrics.Mean(name="metrics_g")
        self.metric_d = tf.keras.metrics.Mean(name="metrics_d")
        # we have to assign metrics to metrics_g for checkpointer restoration
        self.metric = self.metric_g

    def call(self, samples, training: bool = None, stage=None):
        input_real, target_real, target_label, source_label = samples["src_coded_sp"], samples["tar_coded_sp"], \
                                                              samples["tar_speaker"], samples["src_speaker"]

        target_label_reshaped = tf.reshape(target_label, [-1, 1, 1, self.speaker_num])
        # "classifier"  "generator"  "discirmination"
        if stage == "classifier":
            # Classifier training process
            domain_out_real = self.classifier(target_real)
            return {"domain_out_real": domain_out_real, "target_label_reshaped": target_label_reshaped}

        elif stage == "generator":
            # Generator training process
            generated_forward = self.generator(input_real, target_label)
            discirmination = self.discriminator(generated_forward, target_label)
            generated_back = self.generator(generated_forward, source_label)
            identity_map = self.generator(input_real, source_label)
            domain_out_real = self.classifier(target_real)

            return {"discirmination": discirmination,
                    "generated_back": generated_back, "identity_map": identity_map,
                    "target_label_reshaped": target_label_reshaped, "domain_out_real": domain_out_real
                    }

        elif stage == "discriminator":
            # Discriminator training process
            discrimination_real = self.discriminator(target_real, target_label)
            generated_forward = self.generator(input_real, target_label)
            domain_out_fake = self.classifier(generated_forward)
            discirmination = self.discriminator(generated_forward, target_label)

            # apply penalty to make convergence more stable, reference can be found at https://arxiv.org/abs/1704.00028
            epsilon = tf.random.uniform([self.batchsize, 1, 1, 1], 0.0, 1.0)
            length = tf.math.minimum(tf.shape(generated_forward)[2], tf.shape(input_real)[2])
            generated_forward = generated_forward[:, :, 0: length, :]
            x_hat = epsilon * generated_forward + (1 - epsilon) * input_real
            with tf.GradientTape() as tp:
                tp.watch(x_hat)
                d_hat = self.discriminator(x_hat, target_label)
            gradients = tp.gradient(d_hat, x_hat)
            gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

            return {"discrimination_real": discrimination_real,
                    "domain_out_fake": domain_out_fake,
                    "target_label_reshaped": target_label_reshaped,
                    "discirmination": discirmination,
                    "gradient_penalty": gradient_penalty
                    }

    def restore_from_pretrained_model(self, pretrained_model, model_type=""):
        """ A more general-purpose interface for pretrained model restoration

        Args:
            pretrained_model: checkpoint path of mpc model
            model_type: the type of pretrained model to restore
        """
        self.generator = pretrained_model.generator
        self.discriminator = pretrained_model.discriminator
        self.classifier = pretrained_model.classifier

    def convert(self, src_coded_sp, tar_speaker):
        """
        Convert acoustic features

        Args:
            src_coded_sp: the data source to be converted
            tar_speaker：the convert target speaker
        Returns::

            tar_coded_sp: the converted acoustic features
        """
        tar_coded_sp = self.generator(src_coded_sp, tar_speaker)
        return tar_coded_sp

    def get_stage_model(self, stage):
        """ get stargan model of different stage, we need to run these parts sepeartely 
            in solver
        """
        if stage == "classifier":
            return self.classifier
        elif stage == "generator":
            return self.generator
        if stage == "discriminator":
            return self.discriminator

    def get_loss(self, outputs, samples, training=None, stage="classifier"):
        """ get stargan loss of different stage
        """
        if stage == "classifier":
            loss = self.loss_function(outputs, samples, stage=stage)
            self.metric_c.update_state(loss)
            metrics_c = self.metric_c.result()
            return loss, metrics_c
        elif stage == "generator":
            loss = self.loss_function(outputs, samples, stage=stage)
            self.metric_g.update_state(loss)
            metrics_g = self.metric_g.result()
            return loss, metrics_g
        else:
            loss = self.loss_function(outputs, samples, stage=stage)
            self.metric_d.update_state(loss)
            metrics_d = self.metric_d.result()
            return loss, metrics_d

