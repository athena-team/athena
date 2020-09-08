# coding=utf-8
# Copyright (C) 2019 ATHENA AUTHORS; Miao Cao
# All rights reserved.

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
# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-many-locals, too-few-public-methods, too-many-arguments

""" the LAS model """
import tensorflow as tf
from .commons import ACTIVATIONS
from .transformer import TransformerEncoder, TransformerEncoderLayer


class LASLayer(tf.keras.layers.Layer):
    """A LAS model. User is able to modify the attributes as needed. The architecture
        is based on the paper "Listen, Attend and Spell". Chan W, Jaitly N, Le Q V, et al.
        Listen, attend and spell[J].(https://arxiv.org/abs/1508.01211).
    """

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        decdoer_dim=320,
        attention_dim=128,
        num_class=3651,
        dropout=0.1,
        activation='gelu',
        custom_encoder=None,
        custom_decoder=None,
    ):
        super().__init__()
        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layers = [
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                )
                for _ in range(num_encoder_layers)
            ]
            self.encoder = TransformerEncoder(encoder_layers)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            self.decoder = LASDecoder(decdoer_dim, attention_dim, d_model, num_class)

    def call(self, src, tgt, src_mask=None, return_encoder_output=False,
             schedual_sampling_rate=1.0, training=None):
        """Take in and process masked source/target sequences.
        """
        encoder_outputs = self.encoder(src, src_mask=src_mask, training=training)
        output = self.decoder(tgt,
                              memory=encoder_outputs,
                              schedual_sampling_rate=schedual_sampling_rate,
                              training=training)
        if return_encoder_output:
            return output, encoder_outputs

        return output


class LASDecoder(tf.keras.layers.Layer):
    """Decoder block in LAS model.
    """
    def __init__(self, decoder_dim, attention_dim, d_model, num_class, activation="gelu"):
        super(LASDecoder, self).__init__()

        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.num_class = num_class
        self.attention_context = LASAttention(self.attention_dim)
        self.rnn1 = tf.keras.layers.GRUCell(self.decoder_dim)
        self.rnn2 = tf.keras.layers.GRUCell(self.decoder_dim)
        self.random_num = tf.random_uniform_initializer(0, 1)

        self.dense = tf.keras.layers.Dense(d_model, activation=ACTIVATIONS[activation])
        # last layer for output
        self.final_layer = tf.keras.layers.Dense(num_class)

    def call(self, inputs, memory, schedual_sampling_rate=1.0, training=False,
             states=None, return_states=False):
        """Pass the inputs (and mask) through the decoder layer in turn.
        """
        batch_size = tf.shape(inputs)[0]
        output_length = tf.shape(inputs)[1]
        if states is not None:
            (context, rnn1_states, rnn2_states) = states
            rnn1_states = [rnn1_states]
            rnn2_states = [rnn2_states]
        else:
            context = tf.zeros([batch_size, self.attention_dim])
            rnn1_states = [tf.zeros([batch_size, self.decoder_dim])]
            rnn2_states = [tf.zeros([batch_size, self.decoder_dim])]

        result = tf.TensorArray(dtype=tf.float32, size=output_length)
        for step in tf.range(output_length):
            if step > 0 and self.random_num([1]) > schedual_sampling_rate:
                y = self.dense(tf.transpose(result.stack(), [1, 0, 2]), training=training)
                y = self.final_layer(y, training=training)
                seqs_last_step = tf.argmax(y[:, -1, :], axis=1)
                tgt = tf.concat([tf.one_hot(seqs_last_step, self.num_class), context], axis=1)
            else:
                tgt = tf.concat([inputs[:, step, :], context], axis=1)
            rnn1_out, rnn1_states = self.rnn1(inputs=tgt, states=rnn1_states, training=training)
            context = self.attention_context(rnn1_states[0], memory, training=training)
            rnn2_input = tf.concat([rnn1_out, context], axis=1)
            out, rnn2_states = self.rnn2(inputs=rnn2_input, states=rnn2_states, training=training)
            result = result.write(step, out)

        y = tf.transpose(result.stack(), [1, 0, 2])
        y = self.dense(y, training=training)
        output = self.final_layer(y, training=training)
        if return_states:
            return output, (context, rnn1_states[0], rnn2_states[0])
        return output


class LASAttention(tf.keras.layers.Layer):
    """Attention block in LAS model.
    """
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.dense_state = tf.keras.layers.Dense(self.dim)
        self.dense_memory = tf.keras.layers.Dense(self.dim)

    def call(self, decoder_states, memory, training=False):
        """Pass the inputs (and mask) through the attention layer in turn.
        """
        # Linear FC
        state_mapping = self.dense_state(decoder_states, training=training)
        memory_mapping = self.dense_memory(memory, training=training)

        # Linear blendning
        alpha = tf.keras.backend.expand_dims(state_mapping)
        alpha = tf.matmul(memory_mapping, alpha)
        alpha = tf.keras.backend.squeeze(alpha, axis=-1)

        # softmax_vector
        softmaxed = tf.nn.softmax(alpha, axis=-1)
        softmaxed = tf.keras.backend.expand_dims(softmaxed, axis=1)

        # Wheighted vector fetures
        context = tf.matmul(softmaxed, memory_mapping)
        context = tf.keras.backend.squeeze(context, axis=-2)

        return context
