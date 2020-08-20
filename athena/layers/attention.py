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
    def __init__(self, unidirectional=False, look_ahead=0):
        super().__init__()
        self.uni = unidirectional
        self.look_ahead = look_ahead

    def call(self, q, k, v, mask):
        """This is where the layer's logic lives."""
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if self.uni:
            uni_mask = tf.ones(tf.shape(scaled_attention_logits))
            uni_mask = tf.linalg.band_part(uni_mask, -1, self.look_ahead)
            scaled_attention_logits += (1 - uni_mask) * -1e9
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
    """Multi-head attention consists of four parts:

    * Linear layers and split into heads. 
    
    * Scaled dot-product attention. 
    
    * Concatenation of heads. 
    
    * Final linear layer.

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

    def __init__(self, d_model, num_heads, unidirectional=False, look_ahead=0):
        """initialization of multihead attention block

        Args:
            d_model: dimension of multi-head attention
            num_heads: number of attention heads
            unidirectional: whether the self attention is unidirectional. Defaults to False.
            look_ahead: how many frames to look ahead in unidirectional attention. Defaults to 0.
        """        
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

        self.attention = ScaledDotProductAttention(unidirectional, look_ahead=look_ahead)

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
        """call function"""
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
    """the Bahdanau Attention"""

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
        """call function"""
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
    """Refer to [Hierarchical Attention Networks for Document Classification]
    (https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

    >>> Input shape: (Batch size, steps, features)
    >>> Output shape: (Batch size, features)
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
        """build in keras layer"""
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
        """call function in keras"""
        batch_size = tf.shape(inputs)[0]
        W_3d = tf.tile(tf.expand_dims(self.W, axis=0), tf.stack([batch_size, 1, 1]))
        # [batch_size, steps, features]
        input_projection = tf.matmul(inputs, W_3d)

        if self.use_bias:
            input_projection += self.b

        input_projection = tf.tanh(input_projection)

        # [batch_size, steps, 1]
        similarities = tf.reduce_sum(
            tf.multiply(input_projection, self.attention_context_vector),
            axis=2,
            keep_dims=True,
        )

        # [batch_size, steps, 1]
        if mask is not None:
            attention_weights = self._masked_softmax(similarities, mask, axis=1)
        else:
            attention_weights = tf.nn.softmax(similarities, axis=1)

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
    """Refer to [Learning Natural Language Inference with LSTM]
    (https://www.aclweb.org/anthology/N16-1170)

    >>> Input shape: (Batch size, steps, features)
    >>> Output shape: (Batch size, steps, features)
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


class LocationAttention(tf.keras.layers.Layer):
    """location-aware attention

    Reference: Attention-Based Models for Speech Recognition (https://arxiv.org/pdf/1506.07503.pdf)

    """

    def __init__(self, attn_dim, conv_channel, aconv_filts, scaling=1.0):
        super().__init__()
        layers = tf.keras.layers
        self.attn_dim = attn_dim
        self.value_dense_layer = layers.Dense(attn_dim)
        self.query_dense_layer = layers.Dense(attn_dim, use_bias=False)
        self.location_dense_layer = layers.Dense(attn_dim, use_bias=False)

        self.location_conv = layers.Conv1D(filters=conv_channel,
                                           kernel_size=2*aconv_filts+1,
                                           strides=1,
                                           padding="same",
                                           use_bias=False,
                                           data_format="channels_last")
        self.score_dense_layer = layers.Dense(1, name='score_dense_layer')
        self.score_function = None
        # scaling: used to scale softmax scores
        self.scaling = scaling

    def compute_score(self, value, value_length, query, accum_attn_weight):
        """
        Args:
            value_length: the length of value, shape: [batch]
            max_len: the maximun length
        Returns:
            initialized_weights: initializes to uniform distributions, shape: [batch, max_len]
        """
        batch = tf.shape(value)[0]
        x_steps = tf.shape(value)[1]
        # densed_value shape: [batch, x_step, attn_dim]
        densed_value = self.value_dense_layer(value)
        densed_query = tf.reshape(self.query_dense_layer(query), [batch, 1, self.attn_dim])

        accum_attn_weight = tf.reshape(accum_attn_weight, [batch, x_steps, 1])
        attn_location = self.location_conv(accum_attn_weight) # (batch, x_steps, channel)
        attn_location = self.location_dense_layer(attn_location) # (batch, x_steps, attn_dim)
        # [batch, x_step, attn_dim] -> [batch, x_step]
        unscaled_weights = self.score_function(attn_location + densed_value + densed_query)
        masks = tf.sequence_mask(value_length, maxlen=x_steps) # (batch, x_steps)
        masks = (1 - tf.cast(masks, dtype=tf.float32)) * -1e9
        unscaled_weights += masks
        return unscaled_weights

    def initialize_weights(self, value_length, max_len):
        """
        Args:
            value_length: the length of value, shape: [batch]
            max_len: the maximun length
        Returns:
            initialized_weights: initializes to uniform distributions, shape: [batch, max_len]
        """
        prev_attn_weight = tf.sequence_mask(value_length, max_len, dtype=tf.float32)
        # value_length shape: [batch_size, 1]
        value_length = tf.expand_dims(tf.cast(value_length, dtype=tf.float32), axis=1)
        prev_attn_weight = prev_attn_weight / value_length
        return prev_attn_weight

    def call(self, attn_inputs, prev_states, training=True):
        """
        Args:
            attn_inputs (tuple) : it contains 2 params:
                value, shape: [batch, x_steps, eunits]
                value_length, shape: [batch]
            prev_states (tuple) : it contains 3 params:
                query: previous rnn state, shape: [batch, dunits]
                accum_attn_weight: previous accumulated attention weights, shape: [batch, x_steps]
                prev_attn_weight: previous attention weights, shape: [batch, x_steps]
            training: if it is in the training step
        Returns:
            attn_c: attended vector, shape: [batch, eunits]
            attn_weight: attention scores, shape: [batch, x_steps]

        """
        value, value_length = attn_inputs
        query, accum_attn_weight, _ = prev_states
        batch = tf.shape(value)[0]
        x_steps = tf.shape(value)[1]
        self.score_function = lambda x: tf.squeeze(self.score_dense_layer(tf.nn.tanh(x)), axis=2)
        unscaled_weights = self.compute_score(value, value_length, query, accum_attn_weight)
        attn_weight = tf.nn.softmax(self.scaling * unscaled_weights)
        attn_c = tf.reduce_sum(value * tf.reshape(attn_weight, [batch, x_steps, 1]), axis=1)
        return attn_c, attn_weight


class StepwiseMonotonicAttention(LocationAttention):
    """Stepwise monotonic attention

    Reference: Robust Sequence-to-Sequence Acoustic Modeling with Stepwise Monotonic Attention for Neural TTS (https://arxiv.org/pdf/1906.00672.pdf)

    """

    def __init__(self, attn_dim, conv_channel, aconv_filts, sigmoid_noise=2.0,
                 score_bias_init=0.0, mode='soft'):
        super().__init__(attn_dim, conv_channel, aconv_filts)
        self.sigmoid_noise = sigmoid_noise
        self.score_bias_init = score_bias_init
        self.mode = mode

    def build(self, _):
        """A Modified Energy Function is used and the params are defined here.
            Reference: Online and Linear-Time Attention by Enforcing Monotonic Alignments
            (https://arxiv.org/pdf/1704.00784.pdf).
        """
        self.attention_v = self.add_weight(
            name="attention_v", shape=[self.attn_dim], initializer=tf.initializers.GlorotUniform()
        )
        self.attention_g = self.add_weight(
            name="attention_g",
            shape=(),
            initializer=tf.constant_initializer(tf.math.sqrt(1.0 / self.attn_dim).numpy())
        )
        self.attention_b = self.add_weight(
            name="attention_b", shape=[self.attn_dim], initializer=tf.zeros_initializer()
        )
        self.score_bias = self.add_weight(
            name="score_bias",
            shape=(),
            initializer=tf.constant_initializer(self.score_bias_init),
        )

    def initialize_weights(self, value_length, max_len):
        """
        Args:
            value_length: the length of value, shape: [batch]
            max_len: the maximun length
        Returns:
            initialized_weights: initializes to dirac distributions, shape: [batch, max_len]
        Examples:
            An initialized_weights the shape of which is [2, 4]:
            
            >>> [[1, 0, 0, 0],
            >>> [1, 0, 0, 0]]
        """
        batch = tf.shape(value_length)[0]
        return tf.one_hot(tf.zeros((batch,), dtype=tf.int32), max_len)

    def step_monotonic_function(self, sigmoid_probs, prev_weights):
        """hard mode can only be used in the synthesis step

        Args:
            sigmoid_probs: sigmoid probabilities, shape: [batch, x_steps]
            prev_weights: previous attention weights, shape: [batch, x_steps]

        Returns:
            weights: new attention weights, shape: [batch, x_steps]
        """
        if self.mode == "hard":
            move_next_mask = tf.concat([tf.zeros_like(prev_weights[:, :1]), prev_weights[:, :-1]],
                                       axis=1)
            stay_prob = tf.reduce_sum(sigmoid_probs * prev_weights, axis=1, keepdims=True)
            weights = tf.where(stay_prob > 0.5, prev_weights, move_next_mask)
        else:
            pad = tf.zeros([tf.shape(sigmoid_probs)[0], 1], dtype=sigmoid_probs.dtype)
            weights = prev_weights * sigmoid_probs + \
                      tf.concat([pad, prev_weights[:, :-1] * (1.0 - sigmoid_probs[:, :-1])], axis=1)
        return weights

    def call(self, attn_inputs, prev_states, training=True):
        """
        Args:
            attn_inputs (tuple) : it contains 2 params:
                value, shape: [batch, x_steps, eunits]
                value_length, shape: [batch]
            prev_states (tuple) : it contains 3 params:
                query: previous rnn state, shape: [batch, dunits]
                accum_attn_weight: previous accumulated attention weights, shape: [batch, x_steps]
                prev_attn_weight: previous attention weights, shape: [batch, x_steps]
            training: if it is in the training step
        Returns:
            attn_c: attended vector, shape: [batch, eunits]
            attn_weight: attention scores, shape: [batch, x_steps]

        """
        value, value_length = attn_inputs
        query, accum_attn_weight, prev_attn_weight = prev_states
        batch = tf.shape(value)[0]
        x_steps = tf.shape(value)[1]
        normed_v = self.attention_g * self.attention_v * \
                   tf.math.rsqrt(tf.reduce_sum(tf.square(self.attention_v)))

        self.score_function = lambda x: tf.reduce_sum(normed_v * tf.nn.tanh(x + self.attention_b),
                                                      axis=2) + self.score_bias
        unscaled_weights = self.compute_score(value, value_length, query, accum_attn_weight)

        if training:
            noise = tf.random.normal(tf.shape(unscaled_weights), dtype=unscaled_weights.dtype)
            unscaled_weights += self.sigmoid_noise * noise
        if self.mode == 'hard':
            sigmoid_probs = tf.cast(unscaled_weights > 0, unscaled_weights.dtype)
        else:
            sigmoid_probs = tf.nn.sigmoid(unscaled_weights)
        attn_weight = self.step_monotonic_function(sigmoid_probs, prev_attn_weight)
        attn_c = tf.reduce_sum(value * tf.reshape(attn_weight, [batch, x_steps, 1]), axis=1)
        return attn_c, attn_weight

