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
"""base class for learning rate """
import tensorflow as tf
from ..utils.hparam import register_and_parse_hparams


class WarmUpLearningSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """WarmUp Learning rate schedule for Adam

    Example:
        
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate = WarmUpLearningSchedule(512),
        >>>        beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    Idea from the paper: Attention Is All You Need
    """

    def __init__(self, model_dim=512, warmup_steps=4000, k=1.0,
        decay_steps=99999999, decay_rate=1.0):
        """parameters for warmup learning rate schedule

        Args:
            model_dim (int, optional): usually dim of self-attention vector of transformer model. Defaults to 512.
            warmup_steps (int, optional): learning rate increases slowly till warmup_steps. Defaults to 4000.
            decay_steps (int, optional): learning rate decay starts after decay_steps. Defaults to 99999999.
        """        
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
    """WarmUpAdam Implementation"""
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
    """ExponentialDecayLearningRateSchedule

    Example:

        >>> optimizer = tf.keras.optimizers.Adam(learning_rate = ExponentialDecayLearningRate(0.01, 100))

    Args:
        initial_lr, decay_steps
    
    Returns:
        initial_lr * (0.5 ** (step // decay_steps))
    """

    def __init__(self, initial_lr=0.005, decay_steps=10000, decay_rate=0.5,
                 start_decay_steps=30000, final_lr=1e-5):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_steps = tf.cast(decay_steps, tf.float32)
        self.decay_rate = tf.cast(decay_rate, tf.float32)
        self.start_decay_steps = start_decay_steps
        self.final_lr = final_lr

    def __call__(self, step):
        factor = tf.cond(step < self.start_decay_steps,
                         lambda: tf.cast(1, tf.float32),
                         lambda: self.decay_rate ** ((step - self.start_decay_steps) // self.decay_steps))
        lr = self.initial_lr * factor
        return tf.minimum(tf.maximum(lr, self.final_lr), self.initial_lr)


class ExponentialDecayAdam(tf.keras.optimizers.Adam):
    """WarmUpAdam Implementation"""
    default_config = {
        "initial_lr": 0.005,
        "decay_steps": 10000,
        "decay_rate": 0.5,
        "start_decay_steps": 30000,
        "final_lr": 1e-5
    }

    def __init__(self, config=None, beta_1=0.9, beta_2=0.999, epsilon=1e-6,
                 amsgrad=False, name="WarmUpAdam", **kwargs):
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        super().__init__(
            learning_rate=ExponentialDecayLearningRateSchedule(
                self.hparams.initial_lr,
                self.hparams.decay_steps,
                self.hparams.decay_rate,
                self.hparams.start_decay_steps,
                self.hparams.final_lr
            ),
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
        )
