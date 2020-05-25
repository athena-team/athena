# coding=utf-8
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
# Only support eager mode
# pylint: disable=too-few-public-methods, no-member, too-many-arguments, unused-argument
""" learning rate """
import tensorflow as tf
from ..utils.hparam import register_and_parse_hparams
import copy

class WarmUpLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ WarmUp Learning rate schedule for Adam

    Used as :
        optimizer = tf.keras.optimizers.Adam(learning_rate = WarmUpLearningSchedule(512),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    Args :
        model_dim is the something related to total model parameters
        warmup_steps is the highest learning rate iters
    Returns:
        return the learning rate
    Idea from the paper: Attention Is All You Need
    """

    def __init__(self, model_dim=512, warmup_steps=4000, k=1.0,
        decay_steps=99999999, decay_rate=1.0):
        super().__init__()

        self.model_dim = tf.cast(model_dim, tf.float32)
        self.warmup_steps = warmup_steps
        self.k = k
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        k = self.k * tf.cast(self.decay_rate ** (step // self.decay_steps), tf.float32)

        return k * tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)


class WarmUpAdam(tf.keras.optimizers.Adam):
    """WarmUpAdam Implementation """
    default_config = {
        "d_model": 512,
        "warmup_steps": 8000,
        "k": 0.5,
        "decay_steps": 100000,
        "decay_rate": 1.0
    }
    def __init__(self, config=None, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 amsgrad=False, name="WarmUpAdam", **kwargs):
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        super().__init__(
            learning_rate=WarmUpLearningSchedule(
                self.hparams.d_model,
                self.hparams.warmup_steps,
                self.hparams.k,
                self.hparams.decay_steps,
                self.hparams.decay_rate
            ),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
        )


class ExponentialDecayLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ ExponentialDecayLearningRateSchedule

    Used as :
        optimizer = tf.keras.optimizers.Adam(
        learning_rate = ExponentialDecayLearningRate(0.01, 100))
    Args :
        initial_lr, decay_steps
    Returns:
        initial_lr * (0.5 ** (step // decay_steps))
    """

    def __init__(self, initial_lr=0.005, decay_steps=10000, decay_rate=0.5):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        factor = tf.cast(self.decay_rate ** (step // self.decay_steps), tf.float32)
        return self.initial_lr * factor


class ExponentialDecayAdam(tf.keras.optimizers.Adam):
    """WarmUpAdam Implementation """
    default_config = {
        "initial_lr": 0.005,
        "decay_steps": 10000,
        "decay_rate": 0.5
    }

    def __init__(self, config=None, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 amsgrad=False, name="WarmUpAdam", **kwargs):
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        super().__init__(
            learning_rate=ExponentialDecayLearningRateSchedule(
                self.hparams.initial_lr,
                self.hparams.decay_steps,
                self.hparams.decay_rate
            ),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
        )

class WarmUpLearningSchedulePer_layer(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ WarmUp Learning rate schedule for Adam

    Used as :
        optimizer = tf.keras.optimizers.Adam(learning_rate = WarmUpLearningSchedule(512),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    Args :
        model_dim is the something related to total model parameters
        warmup_steps is the highest learning rate iters
    Returns:
        return the learning rate
    Idea from the paper: Attention Is All You Need
    """

    def __init__(self, model_dim=512, warmup_steps=4000, k=1.0,
        decay_steps=99999999, decay_rate=1.0, adaptive_factor=1.0):
        super().__init__()

        self.model_dim = tf.cast(model_dim, tf.float32)
        self.warmup_steps = warmup_steps
        self.k = k
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)
        self.adaptive_factor = (0.9)**adaptive_factor

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        k = self.k * tf.cast(self.decay_rate ** (step // self.decay_steps), tf.float32)
        return self.adaptive_factor * k * tf.math.rsqrt(self.model_dim) * tf.math.minimum(arg1, arg2)


class WarmUpAdamPerLayer(tf.keras.optimizers.Adam):
    """WarmUpAdam Implementation """
    default_config = {
        "d_model": 512,
        "warmup_steps": 8000,
        "k": 0.5,
        "decay_steps": 100000,
        "decay_rate": 1.0
    }
    def __init__(self, config=None, beta_1=0.9, beta_2=0.999, epsilon=1e-7,
                 amsgrad=False, name="WarmUpAdam", **kwargs):
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        super().__init__(
            learning_rate=WarmUpLearningSchedule(
                self.hparams.d_model,
                self.hparams.warmup_steps,
                self.hparams.k,
                self.hparams.decay_steps,
                self.hparams.decay_rate
            ),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
        )
        # We found the adaptive learning rate schedule by probing task
        factors = [abs(5.5 - abs(5.5-i)) for i in range(12)]
        factors += [0.0]
        self.opts = []
        for i in factors:
            opt = tf.keras.optimizers.Adam(
                learning_rate=WarmUpLearningSchedulePer_layer(
                    self.hparams.d_model,
                    self.hparams.warmup_steps,
                    self.hparams.k,
                    self.hparams.decay_steps,
                    self.hparams.decay_rate,
                    i
                ),
                beta_1=beta_1,
                beta_2=beta_2,
                epsilon=epsilon,
                amsgrad=amsgrad,
                name=name,
            )
            self.opts.append(opt)

    def apply_gradients(self, grads_variables):
        opts2gv = {}
        """
        To train encoder with discriminative learning rate,
        we give each encoder layer a individual optimizer.
        """
        # Transformer feedforward layer id
        dense_id = [(140 + i * 6,i + 12) for i in range(12)]
        dense_id += [(141 + i * 6,i + 12) for i in range(12)] 
        dense_layers = dict([("dense_%d"%i[0], i[1]) for i in dense_id])
        # layer id betweem CNN and Transformer layer
        dense_layers['dense_135'] = 12
        # CNN layer id
        cnn_id = [2,3]
        cnn_layers = dict([("conv2d_%d"%i, 12)for i in cnn_id])
        # batchnorm layer id
        norm_id = [2,3]
        norm_layers = dict([("batch_normalization_%d"%i, 12)for i in norm_id])
        freeze_layers = {**cnn_layers,**norm_layers,**dense_layers}
        for g, item in grads_variables:
            if item.name.startswith('masked_predict_coding/transformer_encoder_1') or item.name.split('/')[0] in freeze_layers.keys():
                if "masked_predict_coding" in item.name:
                    layer = int(item.name.split('/')[2].split('_')[-1])
                else:
                    layer = freeze_layers[item.name.split('/')[0]]
                # Transformer attention layer id start from 12 to 23
                if layer <= 23:
                    opt = self.opts[layer-12]
            else:
                opt = self.opts[12]
            if opt not in opts2gv.keys():
                opts2gv[opt] = [[],[]]
            opts2gv[opt][0].append(g)
            opts2gv[opt][1].append(item)

        for opt, gvs in opts2gv.items():
            opt.apply_gradients(zip(gvs[0],gvs[1]))

