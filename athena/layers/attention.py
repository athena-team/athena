# coding=utf-8
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
# pylint: disable=too-few-public-methods, invalid-name
# pylint: disable=too-many-instance-attributes, no-self-use, too-many-arguments

""" Attention layers. """

from absl import logging
import tensorflow as tf


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    def call(self, q, k, v, mask):
        """This is where the layer's logic lives."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += mask * -1e9

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        # (..., seq_len_q, seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """ Multi-head attention

    Multi-head attention consists of four parts: * Linear layers and split into
    heads. * Scaled dot-product attention. * Concatenation of heads. * Final linear layer.
    Each multi-head attention block gets three inputs; Q (query), K (key), V (value).
    These are put through linear (Dense) layers and split up into multiple heads.
    The scaled_dot_product_attention defined above is applied to each head (broadcasted for
    efficiency). An appropriate mask must be used in the attention step. The attention
    output for each head is then concatenated (using tf.transpose, and tf.reshape) and
    put through a final Dense layer.
    Instead of one single attention head, Q, K, and V are split into multiple heads because
    it allows the model to jointly attend to information at different positions from
    different representational spaces. After the split each head has a reduced dimensionality,
    so the total computation cost is the same as a single head attention with full
    dimensionality.
    """

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )
        self.wk = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )
        self.wv = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )

        self.attention = ScaledDotProductAttention()

        self.dense = tf.keras.layers.Dense(
            d_model,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(d_model,),
        )

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).

        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """ call function """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, hiddn_dim)
        k = self.wk(k)  # (batch_size, seq_len, hiddn_dim)
        v = self.wv(v)  # (batch_size, seq_len, hiddn_dim)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = self.attention(q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class BahdanauAttention(tf.keras.Model):
    """ the Bahdanau Attention """

    def __init__(self, units, input_dim=1024):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(
            units,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(input_dim,),
        )
        self.W2 = tf.keras.layers.Dense(
            units,
            kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(input_dim,),
        )
        self.V = tf.keras.layers.Dense(
            1, kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02),
            input_shape=(units,),
        )

    def call(self, query, values):
        """ call function """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)  # (64, 1, 1024)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class HanAttention(tf.keras.layers.Layer):
    """
    Refer to [Hierarchical Attention Networks for Document Classification]
    (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
    wrap `with tf.variable_scope(name, reuse=tf.AUTO_REUSE):`
    Input shape: (Batch size, steps, features)
    Output shape: (Batch size, features)
    """

    def __init__(
        self,
        W_regularizer=None,
        u_regularizer=None,
        b_regularizer=None,
        W_constraint=None,
        u_constraint=None,
        b_constraint=None,
        use_bias=True,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.supports_masking = True
        self.init = tf.keras.initializers.get("glorot_uniform")

        self.W_regularizer = tf.keras.regularizers.get(W_regularizer)
        self.u_regularizer = tf.keras.regularizers.get(u_regularizer)
        self.b_regularizer = tf.keras.regularizers.get(b_regularizer)

        self.W_constraint = tf.keras.constraints.get(W_constraint)
        self.u_constraint = tf.keras.constraints.get(u_constraint)
        self.b_constraint = tf.keras.constraints.get(b_constraint)

        self.use_bias = use_bias

    def build(self, input_shape):
        """ build in keras layer """
        # pylint: disable=attribute-defined-outside-init
        assert len(input_shape) == 3

        self.W = self.add_weight(
            name="{}_W".format(self.name),
            shape=(int(input_shape[-1]), int(input_shape[-1]),),
            initializer=self.init,
            regularizer=self.W_regularizer,
            constraint=self.W_constraint,
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="{}_b".format(self.name),
                shape=(int(input_shape[-1]),),
                initializer="zero",
                regularizer=self.b_regularizer,
                constraint=self.b_constraint,
            )

        self.attention_context_vector = self.add_weight(
            name="{}_att_context_v".format(self.name),
            shape=(int(input_shape[-1]),),
            initializer=self.init,
            regularizer=self.u_regularizer,
            constraint=self.u_constraint,
        )
        self.built = True

    def call(self, inputs, training=None, mask=None):
        """ call function in keras """
        batch_size = tf.shape(inputs)[0]
        W_3d = tf.tile(tf.expand_dims(self.W, axis=0), tf.stack([batch_size, 1, 1]))
        # [batch_size, steps, features]
        input_projection = tf.matmul(inputs, W_3d)

        if self.use_bias:
            input_projection += self.b

        input_projection = tf.tanh(input_projection)

        # [batch_size, steps, 1]
        similaritys = tf.reduce_sum(
            tf.multiply(input_projection, self.attention_context_vector),
            axis=2,
            keep_dims=True,
        )

        # [batch_size, steps, 1]
        if mask is not None:
            attention_weights = self._masked_softmax(similaritys, mask, axis=1)
        else:
            attention_weights = tf.nn.softmax(similaritys, axis=1)

        # [batch_size, features]
        attention_output = tf.reduce_sum(tf.multiply(inputs, attention_weights), axis=1)
        return attention_output

    # pylint: disable=no-self-use
    def compute_output_shape(self, input_shape):
        """compute output shape"""
        return input_shape[0], input_shape[-1]

    def _masked_softmax(self, logits, mask, axis):
        """Compute softmax with input mask."""
        e_logits = tf.exp(logits)
        masked_e = tf.multiply(e_logits, mask)
        sum_masked_e = tf.reduce_sum(masked_e, axis, keep_dims=True)
        ones = tf.ones_like(sum_masked_e)
        # pay attention to a situation that if len of mask is zero,
        # denominator should be set to 1
        sum_masked_e_safe = tf.where(tf.equal(sum_masked_e, 0), ones, sum_masked_e)
        return masked_e / sum_masked_e_safe


class MatchAttention(tf.keras.layers.Layer):
    """
    Refer to [Learning Natural Language Inference with LSTM]
    (https://www.aclweb.org/anthology/N16-1170)
    wrap `with tf.variable_scope(name, reuse=tf.AUTO_REUSE):`
    Input shape: (Batch size, steps, features)
    Output shape: (Batch size, steps, features)
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        logging.info("Initialize MatchAttention {}...".format(self.name))
        self.fc_num_units = config["model"]["net"]["structure"]["fc_num_units"]
        self.middle_layer = tf.keras.layers.Dense(self.fc_num_units, activation="tanh")
        self.attn = tf.keras.layers.Dense(1)

    # pylint: disable=arguments-differ
    def call(self, tensors):
        """Attention layer."""
        left, right = tensors

        len_left = left.shape[1]
        len_right = right.shape[1]
        tensor_left = tf.expand_dims(left, axis=2)
        tensor_right = tf.expand_dims(right, axis=1)
        tensor_left = tf.tile(tensor_left, [1, 1, len_right, 1])
        tensor_right = tf.tile(tensor_right, [1, len_left, 1, 1])
        tensor_merged = tf.concat([tensor_left, tensor_right], axis=-1)
        middle_output = self.middle_layer(tensor_merged)
        attn_scores = self.attn(middle_output)
        attn_scores = tf.squeeze(attn_scores, axis=3)
        exp_attn_scores = tf.exp(
            attn_scores - tf.reduce_max(attn_scores, axis=-1, keepdims=True)
        )
        exp_sum = tf.reduce_sum(exp_attn_scores, axis=-1, keepdims=True)
        attention_weights = exp_attn_scores / exp_sum
        return tf.matmul(attention_weights, right)
