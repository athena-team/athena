# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Xiangang Li; Dongwei Jiang; Xiaoning Lei
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
# Only support eager mode
# pylint: disable=no-member, invalid-name, relative-beyond-top-level
# pylint: disable=too-many-locals, too-many-statements, too-many-arguments, too-many-instance-attributes

""" speech simclr implementation"""

import tensorflow as tf
from .base import BaseModel
from ..loss import ContrastiveLoss, MPCLoss
from ..layers.commons import PositionalEncoding
from ..layers.transformer import TransformerEncoder, TransformerEncoderLayer
from ..utils.hparam import register_and_parse_hparams
import numpy
import pdb


class SimCLR(BaseModel):
    """ Implementation of a SimCLR for speech
    """
    default_config = {
        "num_filters": 512,
        "d_model": 512,
        "num_heads": 8,
        "num_encoder_layers": 12,
        "dff": 1280,
        "temperature": 0.1,
        "normalization": True,
        "rate": 0.1,
        "label_smoothing_rate": 0.0
    }

    def __init__(self, data_descriptions, config=None, ps=None):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)

        self.loss_function = ContrastiveLoss(
            temperature=self.hparams.temperature,
            normalization=self.hparams.normalization, ps=ps)
        self.mpc_loss_function = MPCLoss()
        self.metric = tf.keras.metrics.Mean(name="Accuracy")
        self.downsample_scale = 1
        self.num_class = data_descriptions.num_class * self.downsample_scale

        # for the x_net
        num_filters = self.hparams.num_filters
        d_model = self.hparams.d_model
        layers = tf.keras.layers
        input_features = layers.Input(shape=data_descriptions.sample_shape["input"], dtype=tf.float32)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(input_features)
        inner = layers.LayerNormalization()(inner)
        inner = tf.nn.relu6(inner)
        inner = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            data_format="channels_last",
        )(inner)
        inner = layers.LayerNormalization()(inner)

        inner = tf.nn.relu6(inner)
        _, _, dim, channels = inner.get_shape().as_list()
        output_dim = dim * channels
        inner = layers.Reshape((-1, output_dim))(inner)

        inner = layers.Dense(d_model, activation=tf.nn.relu6)(inner)
        inner = PositionalEncoding(d_model, scale=False)(inner)
        #inner = layers.Dropout(self.hparams.rate)(inner)  # self.hparams.rate
        self.x_net = tf.keras.Model(inputs=input_features, outputs=inner, name="x_net")
        print(self.x_net.summary())

        # transformer layer
        encoder_layers = [
            TransformerEncoderLayer(
                self.hparams.d_model,
                self.hparams.num_heads,
                self.hparams.dff,
                self.hparams.rate,
                "gelu"
            )
            for _ in range(self.hparams.num_encoder_layers)
        ]
        self.encoder = TransformerEncoder(encoder_layers)
        self.dense = layers.Dense(d_model, activation=tf.nn.relu)
        self.dense2 = layers.Dense(d_model, activation=tf.nn.relu)

        # last layer for output
        self.final_layer = layers.Dense(d_model, input_shape=(d_model,))

        # last layer for masked prediction coding
        self.mpc_final_layer = layers.Dense(self.num_class, input_shape=(d_model,))

        # some temp function
        self.random_num = tf.random_uniform_initializer(0, 1)

    def call(self, samples, training: bool = None):
        x0 = samples["input"]
        input_length = self.compute_logit_length(samples)
        input_mask = self._create_masks(x0, input_length)
        x0 = tf.reshape(x0, [tf.shape(x0)[0], tf.shape(x0)[1], tf.shape(x0)[2]])
        x = self.dense2(x0)
        encoder_output = self.encoder(x, input_mask, training=training)
        mask = tf.expand_dims(tf.sequence_mask(input_length, tf.shape(x)[1], dtype=tf.float32), 2)
        encoder_output = encoder_output * mask
        encoder_avg_pool = tf.reduce_mean(encoder_output, axis=1)
        h = self.dense(encoder_avg_pool, training=training)
        y = self.final_layer(h, training=training)
        return y, self.mpc_final_layer(encoder_output)

    def get_loss(self, logits, logits_mpc, samples, training=None):
        loss, logits_ab, labels = self.loss_function(logits, samples)
        logit_length = self.compute_logit_length(samples)
        mpc_loss = self.mpc_loss_function(logits_mpc, samples, logit_length)
        #pdb.set_trace()
        #print("mpc loss", mpc_loss,"simclr loss", loss)
        mtl_loss = 0.5 * mpc_loss + 0.5 * loss
        predictions = tf.argmax(tf.nn.softmax(logits_ab), axis=1)
        truth = tf.argmax(labels, axis=1)
        contrast_acc = tf.equal(predictions, truth)
        contrast_acc = tf.reduce_mean(tf.cast(contrast_acc, tf.float32))
        self.metric.update_state(contrast_acc)
        metrics = {self.metric.name: self.metric.result()}

        return mtl_loss, metrics


    @staticmethod
    def _create_masks(x, input_length):
        r""" Generate a square mask for the sequence. The masked positions are
        filled with float(1.0). Unmasked positions are filled with float(0.0).
        """
        input_mask, output_mask = None, None
        if x is not None:
            input_mask = 1.0 - tf.sequence_mask(
                input_length, tf.shape(x)[1], dtype=tf.float32
            )
            input_mask = input_mask[:, tf.newaxis, tf.newaxis, :]
            input_mask.set_shape([None, None, None, None])
        return input_mask

    def compute_logit_length(self, samples):
        """ used for get logit length """
        input_length = tf.cast(samples["input_length"], tf.float32)
        logit_length = input_length
        # logit_length = tf.math.ceil(input_length / 2)
        # logit_length = tf.math.ceil(logit_length / 2)
        logit_length = tf.cast(logit_length, tf.int32)
        return logit_length

    def prepare_samples(self, samples):
        """ for special data prepare
        carefully: do not change the shape of samples
        """
        input_a = samples["input"]
        input_b = samples["output"]
        input_a = self.feat_masking(input_a, axis=1, mask_num=100)
        input_a = self.feat_masking(input_a, axis=2, mask_num=27)
        input_b = self.feat_masking(input_b, axis=1, mask_num=100)
        input_b = self.feat_masking(input_b, axis=2, mask_num=27)
        bs, _, dim, n_channel = input_a.get_shape().as_list()
        padding_len = tf.abs(input_a.shape[1] - input_b.shape[1])
        padding = tf.zeros([bs, padding_len, dim, n_channel])
        if input_a.shape[1] < input_b.shape[1]:
            input_a = tf.concat([input_a, padding], axis=1)
        elif input_a.shape[1] > input_b.shape[1]:
            input_b = tf.concat([input_b, padding], axis=1)

        #add mask prediction coding
        input_a, mask_input_a, mask_input_a_len = self.mpc_mask(input_a, samples["input_length"])
        input_b, mask_input_b, mask_input_b_len = self.mpc_mask(input_b, samples["output_length"])
        samples["input"] = tf.concat([mask_input_a, mask_input_b], axis=0)
        samples["output"] = tf.concat([input_a, input_b], axis=0)
        samples["input_length"] = tf.concat([mask_input_a_len, mask_input_b_len], axis=0)
        samples["output_length"] = samples["input_length"]
        #samples["input_length"] = tf.concat(
        #    [samples["input_length"], samples["output_length"]],
        #    axis=0)
        return samples

    def feat_masking(self, feat, axis=1, mask_num=0, mask_cols=1, mask_type="mean"):
        ''' masking for spec augment
         Args:
             feat: audio features, shape should be
             [batch_size, time_steps, dimension, channels]
             axis: the axis to be masked
             mask_num: masked time or frequency range
         Returns:
             feat: masked features
        '''

        # time_steps, freq_dim, channels = feat.shape
        batch_size, time_steps, freq_dim, channels = feat.shape
        dim_size = tf.shape(feat)[axis]
        if mask_num > dim_size or mask_num == 0 or mask_cols == 0:
            return feat
        total_t = tf.random.uniform([mask_cols, batch_size, 1], 1, mask_num, tf.int32)
        max_t = tf.reduce_max(total_t, axis=-1)
        t_0 = tf.random.uniform([1], minval=0, maxval=tf.cast(dim_size - max_t, dtype=tf.float32))
        t_0 = tf.reshape(tf.cast(t_0, dtype=tf.int32), [mask_cols, batch_size, 1])
        t_end = t_0 + total_t
        t_0 = tf.tile(t_0, [1, 1, dim_size])
        t_end = tf.tile(t_end, [1, 1, dim_size])
        base_mask = tf.range(dim_size)[tf.newaxis, tf.newaxis, :]
        base_mask = tf.tile(base_mask, [mask_cols, batch_size, 1])
        mask = tf.math.logical_xor(t_end <= base_mask, base_mask < t_0)
        final_mask = mask[0]
        for mask_bran in mask:
            final_mask = tf.math.logical_and(final_mask, mask_bran)
        if axis == 1:
            final_mask = tf.tile(final_mask[:, :, tf.newaxis, tf.newaxis], [1, 1, freq_dim, channels])
        elif axis == 2:
            final_mask = tf.tile(final_mask[:, tf.newaxis, :, tf.newaxis], [1, time_steps, 1, channels])
        if mask_type == "mean":
            mask_metrics = tf.tile(tf.reduce_mean(feat, axis=[1, 2, 3], keepdims=True),
                                   [1, time_steps, freq_dim, channels])
        else:
            mask_metrics = tf.zeros([batch_size, time_steps, freq_dim, channels])
        feat = tf.where(final_mask, feat, mask_metrics)
        return feat

    def mpc_mask(self, fea, fea_len):
        """
        mask fea, return fea, masked_fea and masked fea length
        """
        seq_len = (
            tf.math.floordiv(tf.shape(fea)[1], self.downsample_scale)
            * self.downsample_scale
            )
        fea = fea[:, :seq_len, :, :]
        mask_fea = fea * self.generate_mpc_mask(fea)
        input_length = fea_len
        max_input_length = tf.ones_like(input_length) * seq_len
        mask_len = tf.where(
             tf.less(input_length, max_input_length),
             input_length,
             max_input_length
        )
        return fea, mask_fea, mask_len


    def generate_mpc_mask(self, input_data):
        """ generate mask for pretraining
        Args:
            acoustic features: i.e F-bank
        Return:
            mask tensor
        """
        dtype = input_data.dtype
        chunk_size = 1 * self.downsample_scale
        #chunk_size = self.hparams.chunk_size * self.downsample_scale
        batch, seq_len, dim, num_channels = input_data.get_shape().as_list()
        keep_probability = 0.8
        #if (1 - self.hparams.keep_probability) * seq_len <= chunk_size:
        if (1 - keep_probability) * seq_len <= chunk_size:
            chunk_size = self.downsample_scale
        num_chunk = tf.cast(
            tf.math.ceil(
                tf.divide(tf.cast(seq_len, tf.float32), tf.cast(chunk_size, tf.float32))
            ),
            tf.int32,
        )

        # generate mask with shape [batch, num_chunk]: 1.0 for keep, 0.0 for masked
        ones = tf.ones([batch, num_chunk], dtype=dtype)
        zeros = tf.zeros([batch, num_chunk], dtype=dtype)
        probability = ones * keep_probability
        #probability = ones * self.hparams.keep_probability
        random = self.random_num([batch, num_chunk], dtype=dtype)
        mask = tf.where(tf.less(random, probability), ones, zeros)

        # change the mask for [batch, seq_len]
        mask = tf.tile(mask[:, :, tf.newaxis], [1, 1, chunk_size])
        mask = tf.reshape(mask, [batch, -1])[:, :seq_len]

        # change the mask into masks with the shape [batch, seq_len, dim, num_channels]
        masks = tf.tile(mask[:, :, tf.newaxis, tf.newaxis], [1, 1, dim, num_channels])
        return masks
