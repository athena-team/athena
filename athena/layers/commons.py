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
# pylint: disable=no-self-use, missing-function-docstring
"""Utils for common layers."""

import tensorflow as tf
from athena.layers.functional import make_positional_encoding, collapse4d, gelu, glu
from athena.layers.functional import splice
from athena.utils.misc import gated_linear_layer


class PositionalEncoding(tf.keras.layers.Layer):
    """positional encoding can be used in transformer"""

    def __init__(self, d_model, max_position=800, scale=False):
        super().__init__()
        self.d_model = d_model
        self.scale = scale
        self.pos_encoding = make_positional_encoding(max_position, d_model)

    def call(self, x):
        """ call function """
        seq_len = tf.shape(x)[1]
        if self.scale:
            x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        return x


class ScaledPositionalEncoding(PositionalEncoding):
    """scaled positional encoding,
       reference: https://arxiv.org/pdf/1809.08895.pdf"""
    def __init__(self, d_model, max_position=800):
        super().__init__(d_model, max_position, scale=False)

    def build(self, _):
        self.alpha = self.add_weight(
            name="alpha", initializer=tf.keras.initializers.constant(1)
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x += self.alpha * self.pos_encoding[:, :seq_len, :]
        return x


class Collapse4D(tf.keras.layers.Layer):
    """collapse4d can be used in cnn-lstm for speech processing
       reshape from [N T D C] -> [N T D*C]
    """

    def call(self, x):
        return collapse4d(x)


class Gelu(tf.keras.layers.Layer):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU. Original paper: https://arxiv.org/abs/1606.08415

    Args:
        x: float Tensor to perform activation.
    Returns:
        x: with the GELU activation applied.
    """

    def call(self, x):
        return gelu(x)


class TdnnLayer(tf.keras.layers.Layer):
    """An implementation of Tdnn Layer
    Args:
        context: a int of left and right context, or a list of context indexes, e.g. (-2, 0, 2).
        output_dim: the dim of the linear transform
    """

    def __init__(self, context, output_dim, use_bias=False, **kwargs):
        super().__init__(**kwargs)

        if hasattr(context, "__iter__"):
            self.context_size = len(context)
            self.context_list = context
        else:
            self.context_size = context * 2 + 1
            self.context_list = range(-context, context + 1)

        self.output_dim = output_dim
        self.linear = tf.keras.layers.Dense(output_dim, use_bias=use_bias)

    def call(self, x, training=None, mask=None):
        x = splice(x, self.context_list)
        x = self.linear(x, training=training, mask=mask)
        return x

class GroupNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        groups: int = 2,
        axis: int = -1,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer = "zeros",
        gamma_initializer = "ones",
        beta_regularizer = None,
        gamma_regularizer = None,
        beta_constraint = None,
        gamma_constraint = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape
        )

        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
            "gamma_initializer": tf.keras.initializers.serialize(
                self.gamma_initializer
            ),
            "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": tf.keras.regularizers.serialize(
                self.gamma_regularizer
            ),
            "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
            "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True
        )

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon,
        )
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                "Axis " + str(self.axis) + " of "
                "input tensor should have a defined dimension "
                "but the layer received an input with shape " + str(input_shape) + "."
            )

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") cannot be "
                "more than the number of channels (" + str(dim) + ")."
            )

        if dim % self.groups != 0:
            raise ValueError(
                "Number of groups (" + str(self.groups) + ") must be a "
                "multiple of the number of channels (" + str(dim) + ")."
            )

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead"
            )

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim}
        )

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name="gamma",
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
            )
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name="beta",
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
            )
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape


class InstanceNormalization(GroupNormalization):
    """Instance normalization layer.
    References
        - [Instance Normalization: The Missing Ingredient for Fast Stylization]
        (https://arxiv.org/abs/1607.08022)
    """

    def __init__(self, **kwargs):

        kwargs["groups"] = -1
        super().__init__(**kwargs)


class DownSampleBlock(tf.keras.layers.Layer):
    """conv2d downsample block for stargan, instance norm is used because batch size is 1
    """
    def __init__(self, filters, kernel_size, strides):
        super(DownSampleBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.norm1 = InstanceNormalization(epsilon=1e-8)
        self.norm2 = InstanceNormalization(epsilon=1e-8)

    def call(self, x):
        h1 = self.conv1(x)
        h1_norm = self.norm1(h1)
        h1_gates = self.conv2(x)
        h1_gates_norm = self.norm2(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_gates_norm)
        return h1_glu


class UpSampleBlock(tf.keras.layers.Layer):
    """conv2d upsample block for stargan, instance norm is used because batch size is 1
    """
    def __init__(self, filters, kernel_size, strides):
        super(UpSampleBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")
        self.norm1 = InstanceNormalization(epsilon=1e-8)
        self.norm2 = InstanceNormalization(epsilon=1e-8)

    def call(self, x):
        h1 = self.conv1(x)
        h1_norm = self.norm1(h1)
        h1_gates = self.conv2(x)
        h1_gates_norm = self.norm2(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_gates_norm)
        return h1_glu

class ConditionalInstanceNormalisation(tf.keras.layers.Layer):
    """CIN Block."""
    def __init__(self, in_channel):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.dim_in = in_channel
        self.gamma = tf.keras.layers.Dense(in_channel)
        self.beta = tf.keras.layers.Dense(in_channel)

    def call(self, x, c):
        u = tf.math.reduce_mean(x, axis=1, keepdims=True)
        var = tf.math.reduce_mean((x - u) * (x - u), axis=1, keepdims=True)
        std = tf.math.sqrt(var + 1e-8)

        gamma = self.gamma(c)
        gamma = tf.reshape(gamma, [-1, 1, self.dim_in])
        beta = self.beta(c)
        beta = tf.reshape(beta, [-1, 1, self.dim_in])

        h = (x - u) / std
        h = h * gamma + beta

        return h


class ResidualBlock(tf.keras.layers.Layer):
    """Residual Block with instance normalization."""

    def __init__(self, out_channel):
        super(ResidualBlock, self).__init__()
        self.conv_1 = tf.keras.layers.Conv1D(filters=out_channel, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.cin_1 = ConditionalInstanceNormalisation(out_channel)

    def call(self, x, c):
        x = self.conv_1(x)
        x = self.cin_1(x, c)
        x = gated_linear_layer(inputs=x, gates=x)
        return x


class Down2d_init(tf.keras.layers.Layer):
    def __init__(self, filters , kernel_size, stride):
        super(Down2d_init, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same")

    def call(self, x):
        x1 = self.c1(x)
        x1 = gated_linear_layer(inputs=x1, gates=x1)
        return x1

class Down2d(tf.keras.layers.Layer):
    def __init__(self, filters , kernel_size, stride):
        super(Down2d, self).__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding="same")

        self.norm1 = InstanceNormalization(epsilon=1e-8)
    def call(self, x):
        x1 = self.c1(x)
        x1 = self.norm1(x1)
        x1 = gated_linear_layer(inputs=x1, gates=x1)
        return x1


class Up2d(tf.keras.layers.Layer):
    """docstring for Up2d."""

    def __init__(self, filters, kernel_size, stride):
        super(Up2d, self).__init__()
        self.c1 = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride, padding="same")
        self.norm1 = InstanceNormalization(epsilon=1e-8)

    def call(self, x):
        x1 = self.c1(x)
        x1 = self.norm1(x1)
        x1 = gated_linear_layer(inputs=x1, gates=x1)

        return x1

class ZoneOutCell(tf.keras.layers.LSTMCell):
    """Wrapper for LSTM cell to create ZoneOut Cell

    inspired by:
    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py
    Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.
    """

    def __init__(self, zoneout_rate=0., **kwargs):
        super().__init__(**kwargs)
        self.zoneout_rate = zoneout_rate
        self.drop_layer = tf.keras.layers.Dropout(self.zoneout_rate)

    def call(self, inputs, states, training=False):
        """Runs vanilla LSTM Cell and applies zoneout.
        """
        # Apply vanilla LSTM
        outputs, new_states = super().call(inputs, states, training=training)
        if self.zoneout_rate == 0:
            return outputs, new_states
        # Apply zoneout
        h = (1 - self.zoneout_rate) * \
            self.drop_layer(new_states[0] - states[0], training=training) + \
            states[0]
        c = (1 - self.zoneout_rate) * \
            self.drop_layer(new_states[1] - states[1], training=training) + \
            states[1]
        return outputs, [h, c]

    def get_config(self):
        config = super().get_config()
        config['zoneout_rate'] = self.zoneout_rate
        return config


SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTMCell,
    "gru": tf.keras.layers.GRUCell,
    "cudnnlstm": tf.keras.layers.LSTMCell,
    "cudnngru": tf.keras.layers.GRUCell
}


ACTIVATIONS = {
    "relu": tf.nn.relu,
    "relu6": tf.nn.relu6,
    "elu": tf.nn.elu,
    "selu": tf.nn.selu,
    "gelu": gelu,
    "glu": glu,
    "leaky_relu": tf.nn.leaky_relu,
    "sigmoid": tf.nn.sigmoid,
    "softplus": tf.nn.softplus,
    "softsign": tf.nn.softsign,
    "tanh": tf.nn.tanh,
}
