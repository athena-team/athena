import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# import tensorflow_addons as tfa

def gated_linear_layer(inputs, gates, name=None):

    # activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)
    h1_glu = tf.keras.layers.multiply(inputs=[inputs,tf.sigmoid(gates)], name=name)
    return h1_glu


def instance_norm_layer(inputs, epsilon=1e-05, activation_fn=None, name=None):
    # instance_norm_layer = tf.keras.layers.LayerNormalization()(inputs)
    # instance_norm_layer = tf.contrib.layers.instance_norm(   center=True, epsilon=epsilon,name=name
    #     inputs=inputs,center=True, scale=True, epsilon=epsilon, activation_fn=activation_fn, scope=name)
    mean,var = tf.nn.moments(inputs,axis=(1,2),name=name)

    return instance_norm_layer

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        self.eps = epsilon
        super(InstanceNormalization, self).__init__(**kwargs)
    # def build(self, input_shape):
    #     self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
    #                                  initializer=tf.ones_initializer(), trainable=True)
    #     self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
    #                                 initializer=tf.zeros_initializer(), trainable=True)
    #     super(InstanceNormalization, self).build(input_shape)
    def call(self, x):
        mean = tf.keras.backend.mean(x, axis=(1,2), keepdims=True)
        std = tf.keras.backend.std(x, axis=(1,2), keepdims=True)
        return (x - mean) / (std + self.eps) #+ self.beta   self.gamma *
    def compute_output_shape(self, input_shape):
        return input_shape



def conv1d_layer(inputs, filters, kernel_size, strides=1, padding='same', activation=None, kernel_initializer=None, name=None):

    conv_layer = tf.keras.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name)

    return conv_layer


def conv2d_layer(inputs, filters, kernel_size, strides, padding: list = None, activation=None, kernel_initializer=None, name=None):

    p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    out = tf.pad(inputs, p, name=name + 'conv2d_pad')
    # d1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), activation="relu",
    #                             padding="same")(inner)
    conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=activation,
                                padding="valid")(out)

    return conv_layer


def residual1d_block(inputs, filters=1024, kernel_size=3, strides=1, name_prefix='residule_block_'):
    inputs = tf.keras.Input(shape=inputs.shape, name='img')
    h1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    h1_norm = tf.keras.layers.LayerNormalization(center=True, epsilon=1e-05,name=name_prefix + 'h1_norm')(h1)
    h1_gates=tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    h1_norm_gates=tf.keras.layers.LayerNormalization(center=True, epsilon=1e-05, name=name_prefix + 'h1_norm')(h1_gates)
    # h1_glu = gated_linear_layer(name=name_prefix + 'h1_glu')(inputs=h1_norm, gates=h1_norm_gates)
    h1_glu = tf.keras.layers.multiply(x=h1_norm, y=tf.sigmoid(h1_norm_gates), name=name_prefix + 'h1_glu')
    h2 = tf.keras.layers.Conv1D(filters=filters//2, kernel_size=kernel_size, strides=strides, padding='same')(h1_glu)
    h2_norm = tf.keras.layers.LayerNormalization(center=True, epsilon=1e-05, name=name_prefix + 'h2_norm')(h2)
    h3 = tf.keras.layers.concatenate(inputs=[inputs + h2_norm])
    return h3

def downsample1d_block(inputs, filters, kernel_size, strides, name_prefix='downsample1d_block_'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu

class downsample2d_block(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6, name_prefix='downsample2d_block_'):
        super(downsample2d_block, self).__init__()
        self.eps = epsilon
        self.layernorm1 = InstanceNormalization(epsilon=1e-6)
        self.layernorm2 = InstanceNormalization(epsilon=1e-6)
    def call(self, inputs, filters, kernel_size, strides,padding,name_prefix):
        # inputs = tf.keras.layers.Input(shape=inputs.shape, name='img')
        # inputs = tf.keras.Input(shape=inputs.shape, name='img')

        # inputs, filters = 32, kernel_size = [3, 9], strides = [1, 1], padding = [1, 4], name_prefix = 'down_1'

        # h1 = conv2d_layer(
        #     inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, name=name_prefix + 'h1_conv')
        h1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                            padding="same")(inputs)
        # h1_norm = instance_norm_layer(inputs=h1, activation_fn=None, name=name_prefix + 'h1_norm')
        h1_norm = self.layernorm2(h1)
        h1_gates = conv2d_layer(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None, name=name_prefix + 'h1_gates')
        # h1_norm_gates = instance_norm_layer(inputs=h1_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')
        h1_norm_gates = self.layernorm2(h1_gates)
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')
        return h1_glu

def upsample1d_block(inputs, filters, kernel_size, strides, shuffle_size=2, name_prefix='upsample1d_block_'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs=h1_shuffle, activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, activation=None, name=name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu

class upsample2d_block(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-6):
        super(upsample2d_block, self).__init__()
        self.eps = epsilon
        self.layernorm1 = InstanceNormalization(epsilon=1e-6)
        self.layernorm2 = InstanceNormalization(epsilon=1e-6)

    def call(self, inputs, filters, kernel_size, strides,name_prefix):
        # inputs = tf.keras.layers.Input(shape=inputs.shape, name='img')

        t1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
        # t2 = tf.keras.layers.LayerNormalization(t1, scope=name_prefix + 'instance1')
        t2 = self.layernorm1(t1)
        x1_gates = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
        x1_norm_gates = self.layernorm2(x1_gates)
        x1_glu = gated_linear_layer(t2, x1_norm_gates)
        return x1_glu


def pixel_shuffler(inputs, shuffle_size=2, name=None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow, oc], name=name)

    return outputs


class generator_gatedcnn(tf.keras.layers.Layer):
    '''(inputs, speaker_id=None, reuse=False, scope_name='generator_gatedcnn'):
    #input shape [batchsize, h, w, c]
    #speaker_id [batchsize, one_hot_vector]
    #one_hot_vector：[0,1,0,0]'''

    def __init__(self, reuse=False, scope_name='generator_gatedcnn'):
        super(generator_gatedcnn, self).__init__()
        self.reuse = reuse
        # with tf.variable_scope(scope_name) as scope:
        #     if reuse:
        #         scope.reuse_variables()
        #     else:
        #         assert scope.reuse is False
        # self.downsample2d_block = downsample2d_block()
        # self.upsample2d_block = upsample2d_block()
    def call(self, inputs, speaker_id=None):
        #downsample
        # inputs = tf.squeeze(inputs,[0])
        d1 = downsample2d_block()(inputs, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4], name_prefix='down_1')
        #print(f'd1: {d1.shape.as_list()}')

        d2 = downsample2d_block()(d1, filters=64, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3], name_prefix='down_2')
        #print(f'd2: {d2.shape.as_list()}')

        d3 = downsample2d_block()(d2, filters=128, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3], name_prefix='down_3')
        #print(f'd3: {d3.shape.as_list()}')

        d4 = downsample2d_block()(d3, filters=64, kernel_size=[3, 5], strides=[1, 1], padding=[1, 2], name_prefix='down_4')
        #print(f'd4: {d4.shape.as_list()}')
        d5 = downsample2d_block()(d4, filters=5, kernel_size=[9, 5], strides=[9, 1], padding=[1, 2], name_prefix='down_5')

        #upsample
        speaker_id = tf.convert_to_tensor(speaker_id, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape.dims[-1].value]), tf.float32)
        c = tf.tile(c_cast, [1, d5.shape.dims[1].value, d5.shape.dims[2].value, 1])
        #print(c.shape.as_list())
        # concated = tf.concat([d5, c], axis=-1)
        concated = tf.keras.layers.concatenate(inputs=[d5, c], axis=-1)
        # #print(concated.shape.as_list())

        u1 = upsample2d_block()(concated, 64, kernel_size=[9, 5], strides=[9, 1], name_prefix='gen_up_u1')
        #print(f'u1.shape :{u1.shape.as_list()}')

        c1 = tf.tile(c_cast, [1, u1.shape.dims[1].value, u1.shape.dims[2].value, 1])
        #print(f'c1 shape: {c1.shape}')
        u1_concat = tf.concat([u1, c1], axis=-1)
        u1_concat = tf.keras.layers.concatenate(inputs=[u1, c1], axis=-1)
        #print(f'u1_concat.shape :{u1_concat.shape.as_list()}')+

        u2 = upsample2d_block()(u1_concat, 128, [3, 5], [1, 1], name_prefix='gen_up_u2')
        #print(f'u2.shape :{u2.shape.as_list()}')
        c2 = tf.tile(c_cast, [1, u2.shape[1], u2.shape[2], 1])
        # u2_concat = tf.concat([u2, c2], axis=-1)
        u2_concat = tf.keras.layers.concatenate(inputs=[u2, c2], axis=-1)

        u3 = upsample2d_block()(u2_concat, 64, [4, 8], [2, 2], name_prefix='gen_up_u3')
        #print(f'u3.shape :{u3.shape.as_list()}')
        c3 = tf.tile(c_cast, [1, u3.shape[1], u3.shape[2], 1])
        # u3_concat = tf.concat([u3, c3], axis=-1)
        u3_concat = tf.keras.layers.concatenate(inputs=[u3, c3], axis=-1)

        u4 = upsample2d_block()(u3_concat, 32, [4, 8], [2, 2], name_prefix='gen_up_u4')
        #print(f'u4.shape :{u4.shape.as_list()}')
        c4 = tf.tile(c_cast, [1, u4.shape[1], u4.shape[2], 1])
        # u4_concat = tf.concat([u4, c4], axis=-1)
        u4_concat = tf.keras.layers.concatenate(inputs=[u4, c4], axis=-1)
        #print(f'u4_concat.shape :{u4_concat.shape.as_list()}')

        u5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 9], strides=[1, 1], padding='same', name='generator_last_deconv')(u4_concat)
        #print(f'u5.shape :{u5.shape.as_list()}')

        return u5

class discriminator(tf.keras.layers.Layer):
    def __init__(self, reuse=False, scope_name='discriminator'):
        super(discriminator,self).__init__()
        self.reuse = reuse
        # with tf.variable_scope(scope_name) as scope:
        #     if reuse:
        #         scope.reuse_variables()
        #     else:
        #         assert scope.reuse is False

    def call(self, inputs, speaker_id=None):
    # inputs has shape [batch_size, height,width, channels]
        #convert data type to float32
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, inputs.shape[1], inputs.shape[2], 1])

        concated = tf.concat([inputs, c], axis=-1)

        # inputs = tf.keras.Input(shape=concated.shape, name='img')
        # Downsample
        d1 = downsample2d_block()(
            inputs=concated, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4], name_prefix='downsample2d_dis_block1_')
        c1 = tf.tile(c_cast, [1, d1.shape[1], d1.shape[2], 1])
        d1_concat = tf.concat([d1, c1], axis=-1)

        d2 = downsample2d_block()(
            inputs=d1_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3], name_prefix='downsample2d_dis_block2_')
        c2 = tf.tile(c_cast, [1, d2.shape[1], d2.shape[2], 1])
        d2_concat = tf.concat([d2, c2], axis=-1)

        d3 = downsample2d_block()(
            inputs=d2_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3], name_prefix='downsample2d_dis_block3_')
        c3 = tf.tile(c_cast, [1, d3.shape[1], d3.shape[2], 1])
        d3_concat = tf.concat([d3, c3], axis=-1)

        d4 = downsample2d_block()(
            inputs=d3_concat, filters=32, kernel_size=[3, 6], strides=[1, 2], padding=[1, 2], name_prefix='downsample2d_diss_block4_')
        c4 = tf.tile(c_cast, [1, d4.shape[1], d4.shape[2], 1])
        d4_concat = tf.concat([d4, c4], axis=-1)

        c1 = conv2d_layer(d4_concat, filters=1, kernel_size=[36, 5], strides=[36, 1], padding=[0, 1], name='discriminator-last-conv')

        c1_red = tf.reduce_mean(c1, keepdims=True)

        return c1_red


class domain_classifier(tf.keras.layers.Layer):  #(inputs, reuse=False, scope_name='classifier'):
    def __init__(self, batchsize=1,reuse=False, frames=512,scope_name='classifier'):
        super(domain_classifier, self).__init__()
        self.scope_name = scope_name

        # with tf.variable_scope(scope_name) as scope:
        #     if reuse:
        #         scope.reuse_variables()
        #     else:
        #         assert scope.reuse is False

        inner = tf.keras.layers.Input(shape=[8, frames, 1])
        d1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(2,2),activation="relu",
                               padding="same")(inner)
        d2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2,2),activation="relu",
                               padding="same")(d1)
        d3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(d2)
        d4 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 4), strides=(1, 2), activation="relu",
                                    padding="same")(d3)
        d5 = tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 4), strides=(1, 2), activation="relu",
                                    padding="same")(d4)
        p = tf.keras.layers.GlobalAveragePooling2D()(d5)
        # print(p.get_shape())
        a, dim = p.get_shape().as_list()
        # o_r = tf.reshape(p, [-1, 1, 1, p.shape.dims[1].value])
        o_r =tf.keras.layers.Reshape([-1, 1, dim])(p)
        # _, _, dim, channels = inner.get_shape().as_list()
        # output_dim = dim * channels
        # inner = layers.Reshape((-1, output_dim))(inner)
        self.x_net = tf.keras.Model(inputs=inner, outputs=o_r, name="x_net")
        # print(self.x_net.summary())

    def call(self, inputs):
        inputs = inputs[:, 0:8, :, :]
        x0 = self.x_net(inputs)

        return x0
        #   add slice input shape [batchsize, 8, 512, 1]
        #get one slice
        # one_slice = inputs[:, 0:8, :, :]
        #
        # d1 = tf.keras.layers.conv2d(one_slice, 8, kernel_size=[4, 4], padding='same', name=self.scope_name + '_conv2d01')
        # d1_p = tf.keras.layers.max_pooling2d(d1, [2, 2], strides=[2, 2], name=self.scope_name + 'p1')
        # #print(f'domain_classifier_d1: {d1.shape}')
        # #print(f'domain_classifier_d1_p: {d1_p.shape}')
        #
        # d2 = tf.keras.layers.conv2d(d1_p, 16, [4, 4], padding='same', name=self.scope_name + '_conv2d02')
        # d2_p = tf.keras.layers.max_pooling2d(d2, [2, 2], strides=[2, 2], name=self.scope_name + 'p2')
        # #print(f'domain_classifier_d12: {d2.shape}')
        # #print(f'domain_classifier_d2_p: {d2_p.shape}')
        #
        # d3 = tf.keras.layers.conv2d(d2_p, 32, [4, 4], padding='same', name=self.scope_name + '_conv2d03')
        # d3_p = tf.keras.layers.max_pooling2d(d3, [2, 2], strides=[2, 2], name=self.scope_name + 'p3')
        # #print(f'domain_classifier_d3: {d3.shape}')
        # #print(f'domain_classifier_d3_p: {d3_p.shape}')
        #
        # d4 = tf.keras.layers.conv2d(d3_p, 16, [3, 4], padding='same', name=self.scope_name + '_conv2d04')
        # d4_p = tf.keras.layers.max_pooling2d(d4, [1, 2], strides=[1, 2], name=self.scope_name + 'p4')
        # #print(f'domain_classifier_d4: {d4.shape}')
        # #print(f'domain_classifier_d4_p: {d4_p.shape}')
        #
        # d5 = tf.keras.layers.conv2d(d4_p, 4, [1, 4], padding='same', name=self.scope_name + '_conv2d05')
        # d5_p = tf.keras.layers.max_pooling2d(d5, [1, 2], strides=[1, 2], name=self.scope_name + 'p5')
        # #print(f'domain_classifier_d5: {d5.shape}')
        # #print(f'domain_classifier_d5_p: {d5_p.shape}')
        #
        # p = tf.keras.layers.GlobalAveragePooling2D()(d5_p)
        #
        # o_r = tf.reshape(p, [-1, 1, 1, p.shape.dims[1].value])
        # #print(f'classifier_output: {o_r.shape}')


if __name__ == '__main__':
    # starganvc = generator_gatedcnn()
    inner = tf.ones([1,36,512,1],dtype="float32")
    print(inner.shape)
    # inner = inner.squezz([0])
    # inner =tf.squeeze(inner,[0])
    generator = discriminator()
    target_label =tf.constant((1,1,1,0,0),dtype="float32")
    # target_label = tf.concat(tf.constant([1]),onehot)
    generated_forward = generator(inner, speaker_id=target_label)
    print(inner)