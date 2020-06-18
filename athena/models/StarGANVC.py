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
                 batchsize=1,
                 mode='train',
                 log_dir='./log',
                 generator_learning_rate=0.0001, \
                 discriminator_learning_rate=0.0001, \
                 classifier_learning_rate=0.0001,
                 sample_signature = None,
                 config = None,
                 SPEAKERS_NUM=9
    ):
        super().__init__()
        self.hparams = register_and_parse_hparams(self.default_config, config, cls=self.__class__)
        self.sample_signature = sample_signature
        self.num_features = num_features
        self.batchsize = batchsize
        self.speaker_num = SPEAKERS_NUM
        self.vcinput_shape = [num_features, frames, 1]#[None, num_features, frames, 1]
        self.label_shape = [ SPEAKERS_NUM]#[None, SPEAKERS_NUM]
        self.mode = mode
        self.log_dir = log_dir
        self.classifier = domain_classifier()    #(batchsize=1, reuse=True)
        self.discriminator = discriminator()#(reuse=True)
        self.generator = generator_gatedcnn()#(reuse=True)
        # self.C_net = C_net()
        self.lambda_cycle = 10
        self.lambda_identity = 5
        self.lambda_classifier = 3
        # self.build_model()
        self.classifier_optimizer = tf.keras.optimizers.Adam(classifier_learning_rate)
        self.generator_optimizer = tf.keras.optimizers.Adam(generator_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(discriminator_learning_rate)
        layers = tf.keras.layers
    # def build_model(self):
        input_real = layers.Input(shape=self.vcinput_shape, name='input_real',batch_size=self.batchsize)
        target_real = layers.Input(shape=self.vcinput_shape, name='target_real',batch_size=self.batchsize)
        target_label = layers.Input(shape=self.label_shape, name='target_label',batch_size=self.batchsize)
        source_label = layers.Input(shape = self.label_shape, name='source_label',batch_size=self.batchsize)
        target_label_reshaped = layers.Input(shape=[1, 1, SPEAKERS_NUM],
                                                    name='reshaped_label_for_classifier',batch_size=self.batchsize)
        self.domain_out_real = self.classifier(target_real)
        self.C_net = tf.keras.Model(inputs=[target_real],outputs=[self.domain_out_real],name='C_net')

        # cat_feature= layers.concatenate([input_real, target_label])   generator_gatedcnn
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
        # gradient penalty
        # self.tmp = self.discriminator(x_hat, target_label)
        # gradients = tf.GradientTape(self.tmp, x_hat)

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
        # self.D_net = tf.keras.Model(inputs=[input_real, target_label,target_real], outputs=
        # [self.discrimination_real, self.discirmination_fake, self.domain_out_fake, gradients], name="D_net")
        return domain_classifier, self.discirmination, self.generated_forward, self.generated_back, \
               self.domain_out_real, self.identity_map, self.discrimination_real, self.discirmination_fake, \
               self.domain_out_fake

    # @tf.function
    def train_step(self,input_real, target_real, target_label, source_label):
        # input_real, target_real, target_label, source_label = inputs
        target_label_reshaped = np.reshape(target_label, [len(target_label), 1, 1, SPEAKERS_NUM])
        with tf.GradientTape() as C_tape,tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
            self.domain_out_real = self.C_net(target_real,self.classifier)
            self.discirmination, self.generated_forward, self.generated_back, self.domain_out_real, self.identity_map = \
                self.G_net(input_real, target_real, target_label, source_label, target_label_reshaped,
                      self.generator,self.discriminator,self.classifier)
            self.discrimination_real, self.discirmination_fake, self.domain_out_fake, gradients = \
                self.D_net(input_real,target_label, target_real,self.discriminator,self.classifier)
            C_loss = loss.ClassifyLoss(target_label_reshaped, self.domain_out_real)
            G_loss = loss.GeneratorLoss(self.discirmination, input_real, self.identity_map,
                                        target_label_reshaped,self.domain_out_real)
            D_loss = loss.DiscriminatorLoss(self.discrimination_real, self.discirmination_fake, target_label_reshaped,
                                            self.domain_out_fake,gradients )

        gradients_of_classify = C_tape.gradient(C_loss, self.C_net.trainable_variables)
        gradients_of_generator = G_tape.gradient(G_loss, self.G_net.trainable_variables)
        gradients_of_discriminator = D_tape.gradient(D_loss, self.D_net.trainable_variables)

        self.classifier_optimizer.apply_gradients(zip(gradients_of_classify, domain_classifier.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_gatedcnn.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        # return C_loss,G_loss,D_loss

    def train(self, input_real, target_real, target_label, source_label, total_batches=-1):
        """ Update the model in 1 epoch """
        train_step = self.train_step
        # if self.hparams.enable_tf_function:
        #     logging.info("please be patient, enable tf.function, it takes time ...")
        #     train_step = tf.function(train_step, input_signature=self.sample_signature)
        # for batch, (input_real, target_real, target_label, source_label) in enumerate(dataset):
            # samples = self.model.prepare_samples(samples)

        C_loss, G_loss, D_loss = \
        train_step(input_real, target_real, target_label, source_label)
        # if (epoch + 1) % 15 == 0:
        #     checkpoint.save(file_prefix=checkpoint_prefix)
        return C_loss, G_loss, D_loss



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
