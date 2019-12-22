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
# pylint: disable=invalid-name
r""" check point manager """
import os
import tensorflow as tf
from absl import logging
import numpy as np

class Checkpoint(tf.train.Checkpoint):
    """ A wrapper for Tensorflow checkpoint

    Args:
        checkpoint_directory: the directory for checkpoint
        summary_directory: the directory for summary used in Tensorboard
        __init__ provide the optimizer and model
        __call__ save the model

    Example:
        transformer = SpeechTransformer(target_vocab_size=dataset_builder.target_dim)
        optimizer = tf.keras.optimizers.Adam()
        ckpt = Checkpoint(checkpoint_directory='./train', summary_directory='./event',
            transformer=transformer, optimizer=optimizer)
        solver = BaseSolver(transformer)
        for epoch in dataset:
            ckpt()
    """

    def __init__(self, checkpoint_directory=None, **kwargs):
        super().__init__(**kwargs)
        self.best_loss = np.inf
        if checkpoint_directory is None:
            checkpoint_directory = os.path.join(os.path.expanduser("~"), ".athena")
        self.checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.checkpoint_directory = checkpoint_directory
        logging.info("trying to restore from : %s" % checkpoint_directory)
        # load from checkpoint if previous models exist in checkpoint dir
        self.restore(tf.train.latest_checkpoint(checkpoint_directory))

    def _compare_and_save_best(self, loss, save_path):
        """ compare and save the best model in best_loss """
        if loss is None:
            return
        if loss < self.best_loss:
            self.best_loss = loss
            with open(os.path.join(self.checkpoint_directory, 'best_loss'), 'w') as wf:
                checkpoint = save_path.split('/')[-1]
                wf.write('model_checkpoint_path: "%s"' % checkpoint)

    def __call__(self, loss=None):
        logging.info("saving model in :%s" % self.checkpoint_prefix)
        save_path = self.save(file_prefix=self.checkpoint_prefix)
        self._compare_and_save_best(loss, save_path)

    def restore_from_best(self):
        """ restore from the best model """
        self.restore(
            tf.train.latest_checkpoint(
                self.checkpoint_directory,
                latest_filename='best_loss'
            )
        )