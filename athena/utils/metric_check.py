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
"""MetricChecker"""
import time
import tensorflow as tf


class MetricChecker:
    """Hold and save best metric checkpoint
    
    Args:
        name: MetricChecker name
        maximum: more greater more better
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.time_last_call = time.time()
        self.steps_last_call = 0

    def __call__(self, loss, metrics, evaluate_epoch=-1):
        """summary the basic metrics like loss, lr
        Args:
            metrics: average loss of all previous steps in one epoch
                if training is False, it must be provided
            evaluate_epoch:
                if evaluate_epoch >= 0: <evaluate mode>
                if evaluate_epoch == -1: <train mode>
                if evaluate_epoch < -1: <evaluate_log mode> (no tf.summary.write)
        Returns:
            logging_str: return average and best(if improved) loss if training is False
        """
        if evaluate_epoch is -1:
            return self.summary_train(loss, metrics)
        return self.summary_evaluate(loss, metrics, evaluate_epoch)

    def summary_train(self, loss, metrics):
        """generate summary of learning_rate, loss, metrics, speed and write on Tensorboard"""
        global_steps = tf.convert_to_tensor(self.optimizer.iterations)
        learning_rate = (
            self.optimizer.lr
            if isinstance(self.optimizer.lr, tf.Variable)
            else (self.optimizer.lr(global_steps))
        )

        total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        tf.summary.scalar("total_loss", total_loss, step=global_steps)
        if isinstance(loss, dict):
            for name in loss:
                tf.summary.scalar(name, loss[name], step=global_steps)
        tf.summary.scalar("learning_rate", learning_rate, step=global_steps)
        if metrics is not None:
            for name in metrics:
                metric = metrics[name]
                tf.summary.scalar(name, metric, step=global_steps)

        reports = ""
        reports += "global_steps: %d\t" % (global_steps)
        reports += "learning_rate: %.4e\t" % (learning_rate)
        reports += "loss: %.4f\t" % (total_loss)
        if isinstance(loss, dict):
            for name in loss:
                reports += "%s: %.4f\t" % (name, loss[name])
        if metrics is not None:
            for name in metrics:
                metric = metrics[name]
                reports += "%s: %.4f\t" % (name, metric)
        right_now = time.time()
        duration = right_now - self.time_last_call
        self.time_last_call = right_now
        if self.steps_last_call != 0:
            # average duration over log_interval steps
            sec_per_iter = duration / tf.cast(
                (global_steps - self.steps_last_call), tf.float32
            )
            reports += "sec/iter: %.4f" % (sec_per_iter)
        self.steps_last_call = global_steps

        return reports

    def summary_evaluate(self, loss, metrics, epoch=-1):
        """If epoch > 0, return a summary of loss and metrics on dev set and write on Tensorboard
        Otherwise, just return evaluate loss and metrics
        """
        reports = ""
        global_steps = tf.convert_to_tensor(self.optimizer.iterations)
        total_loss = sum(list(loss.values())) if isinstance(loss, dict) else loss
        if epoch >= 0:
            tf.summary.scalar("evaluate_total_loss", total_loss, step=global_steps)
            if isinstance(loss, dict):
                for name in loss:
                    tf.summary.scalar("evaluate_" + name, loss[name], step=global_steps)
            if metrics is not None:
                for name in metrics:
                    metric = metrics[name]
                    tf.summary.scalar("evaluate_" + name, metric, step=global_steps)
            reports += "epoch: %d\t" % (epoch)
        reports += "loss: %.4f\t" % (total_loss)
        if isinstance(loss, dict):
            for name in loss:
                reports += "%s: %.4f\t" % (name, loss[name])
        if metrics is not None:
            for name in metrics:
                metric = metrics[name]
                reports += "%s: %.4f\t" % (name, metric)
        return reports
