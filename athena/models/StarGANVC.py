import os
import tensorflow as tf
from datetime import datetime
import numpy as np
from absl import logging
from ..utils.hparam import register_and_parse_hparams
from ..models.base import BaseModel
from ..models.StarGAN_module import *


class StarGANVC(tf.keras.layers.Layer):
    default_config={}
    def __init__(self,
                 num_features,
                 frames=512,
                 sample_signature = None,
                 config = None,
                 speaker_num=2
    ):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.sample_signature = sample_signature
        self.num_features = num_features
        self.vcinput_shape = [ num_features, frames, 1]
        self.label_shape = [ speaker_num]
        self.classifier = domain_classifier()
        self.discriminator = discriminator()
        self.generator = generator_gatedcnn()
        layers = tf.keras.layers
        input_real = layers.Input(shape=self.vcinput_shape, name='input_real')
        target_real = layers.Input(shape=self.vcinput_shape, name='target_real')
        target_label = layers.Input(shape=self.label_shape, name='target_label')
        source_label = layers.Input(shape = self.label_shape, name='source_label')

        target_label_reshaped = layers.Input(shape=[1, 1, speaker_num],
                                                    name='reshaped_label_for_classifier')
        self.domain_out_real = self.classifier(target_real)
        self.C_net = tf.keras.Model(inputs=[target_real],outputs=[self.domain_out_real],name='C_net')

        self.generated_forward = self.generator(input_real, speaker_id=target_label)
        self.discirmination = self.discriminator(self.generated_forward, target_label)
        self.generated_back = self.generator(self.generated_forward, source_label)
        self.domain_out_real = self.classifier(target_real)
        # Identity loss
        self.identity_map = self.generator(input_real, source_label)
        self.G_net = tf.keras.Model(inputs=[input_real, target_real,target_label,source_label,
        target_label_reshaped], outputs=[self.discirmination,self.generated_forward,
        self.generated_back,self.domain_out_real,self.identity_map], name="G_net")
        print(self.G_net.summary())

        # discrimination_real, discirmination_fake, generated_forward, target_label_reshaped, domain_out_fake = inputs
        # discirmination
        self.discrimination_real = self.discriminator(target_real, target_label)
        self.discirmination_fake = self.discriminator(self.generated_forward, target_label)
        self.domain_out_fake = self.classifier(self.generated_forward)
        # calculate `x_hat`

        # epsilon = tf.random.uniform((self.batchsize, 1, 1, 1), 0.0, 1.0)
        # x_hat = epsilon * self.generated_forward + (1.0 - epsilon) * input_real
        # with tf.GradientTape(persistent=True) as tape:
        #     tmp = self.discriminator(x_hat, target_label)
        # gradients = tape.gradient(tmp, x_hat)

        self.D_net = tf.keras.Model(inputs=[input_real, target_label,target_real], outputs=
        [self.discrimination_real, self.discirmination_fake, self.domain_out_fake], name="D_net")  #, gradients
        print(self.D_net.summary())

    def call(self, samples, training: bool = None):
        input_real, target_real, target_label, source_label= samples["input_real"],\
        samples["target_real"],samples["target_label"],samples["source_label"]

        target_label_reshaped = np.reshape(target_label, [target_label.shape[0], 1, 1, self.speaker_num])
        domain_classifier = self.C_net(target_real, training=training)
        self.discirmination, self.generated_forward, self.generated_back, self.domain_out_real, self.identity_map = \
            self.G_net([input_real, target_real, target_label, source_label, target_label_reshaped], training=training)

        self.discrimination_real, self.discirmination_fake, self.domain_out_fake = self.D_net([input_real,
        target_label,target_real],training=training)
        return domain_classifier, self.discirmination, self.generated_forward, self.generated_back, \
               self.domain_out_real, self.identity_map, self.discrimination_real, self.discirmination_fake, \
               self.domain_out_fake


    def test(self, input_test, target_label):
        generation_test = generator_gatedcnn(input_test, target_label)
        return generation_test

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, filepath):
        self.saver.restore(self.sess, filepath)


if __name__ == '__main__':

    vcinput_shape = [2,36, 512, 1]  # [None, num_features, frames, 1]
    label_shape = [2,4]
    resha_label = [2,1,1,4]
    samples ={"input_real":tf.ones(vcinput_shape,dtype="float32"),
              "target_real":tf.ones(vcinput_shape,dtype="float32"),
              "target_label":tf.ones(label_shape,dtype="float32"),
              "source_label": tf.ones(label_shape, dtype="float32"),
              "target_label_reshaped": tf.ones(label_shape, dtype="float32")
              }
    input_real, target_real, target_label, source_label, target_label_reshaped = samples["input_real"], \
                                                                                 samples["target_real"], samples[
                                                                                     "target_label"], samples[
                                                                                     "source_label"], samples[
                                                                                     "target_label_reshaped"]
    starganvc = StarGANVC(36)
    a,b,c,d,e,f,g,h,ij = starganvc(samples)
    print(a)
    print(b)
#     domain_classifier, self.discirmination, self.generated_forward, self.generated_back, \
#                self.domain_out_real, self.identity_map, self.discrimination_real, self.discirmination_fake, \
#                self.domain_out_fake
