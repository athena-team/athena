import tensorflow as tf

def gated_linear_layer(inputs, gates, name=None):
    # activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)
    h1_glu = tf.keras.layers.multiply(inputs=[inputs, tf.sigmoid(gates)], name=name)
    return h1_glu


def InstanceNormalization(x):
    mean = tf.keras.backend.mean(x, axis=(1, 2), keepdims=True)
    std = tf.keras.backend.std(x, axis=(1, 2), keepdims=True)
    return (x - mean) / (std + 1e-6)  # + self.beta   self.gamma *


def conv2d_layer(inputs, filters, kernel_size, strides, padding: list = None, activation=None, kernel_initializer=None,
                 name=None):
    p = tf.constant([[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    out = tf.pad(inputs, p, name=name + 'conv2d_pad')
    conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                        activation=activation,
                                        padding="valid")(out)

    return conv_layer


def downsample2d_block(inputs, filters, kernel_size, strides, padding, name_prefix):
    h1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                padding="same")(inputs)
    h1_norm = InstanceNormalization(h1)
    h1_gates = conv2d_layer(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None,
        name=name_prefix + 'h1_gates')
    h1_norm_gates = InstanceNormalization(h1_gates)
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix + 'downsample_h1_glu')
    return h1_glu


def upsample2d_block(inputs, filters, kernel_size, strides, name_prefix):
    t1 = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
    t2 = InstanceNormalization(t1)
    x1_gates = tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides, padding='same')(inputs)
    x1_norm_gates = InstanceNormalization(x1_gates)
    x1_glu = gated_linear_layer(t2, x1_norm_gates)
    return x1_glu


class generator_gatedcnn(tf.keras.layers.Layer):
    '''(inputs, speaker_id=None, reuse=False, scope_name='generator_gatedcnn'):
    #input shape [batchsize, h, w, c]
    #speaker_id [batchsize, one_hot_vector]
    #one_hot_vector：[0,1,0,0]'''

    def __init__(self, reuse=False, scope_name='generator_gatedcnn'):
        super(generator_gatedcnn, self).__init__()
        self.reuse = reuse
        # inputs!! [1, 36, 512, 1]
        # speaker_id!! [1, 4]
        inputs = tf.keras.layers.Input(shape=[36, 512, 1])
        speaker_id = tf.keras.layers.Input(shape=[1, 4])

        d1 = downsample2d_block(inputs, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4],
                                name_prefix='down_1')
        # print(f'd1: {d1.shape.as_list()}')

        d2 = downsample2d_block(d1, filters=64, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3],
                                name_prefix='down_2')
        # print(f'd2: {d2.shape.as_list()}')

        d3 = downsample2d_block(d2, filters=128, kernel_size=[4, 8], strides=[2, 2], padding=[1, 3],
                                name_prefix='down_3')
        # print(f'd3: {d3.shape.as_list()}')

        d4 = downsample2d_block(d3, filters=64, kernel_size=[3, 5], strides=[1, 1], padding=[1, 2],
                                name_prefix='down_4')
        # print(f'd4: {d4.shape.as_list()}')
        d5 = downsample2d_block(d4, filters=5, kernel_size=[9, 5], strides=[9, 1], padding=[1, 2], name_prefix='down_5')

        # upsample
        speaker_id = tf.convert_to_tensor(speaker_id, dtype=tf.float32)
        c_cast = tf.cast(tf.reshape(speaker_id, [-1, 1, 1, speaker_id.shape.dims[-1].value]), tf.float32)
        c = tf.tile(c_cast, [1, d5.shape.dims[1].value, d5.shape.dims[2].value, 1])
        # print(c.shape.as_list())
        # concated = tf.concat([d5, c], axis=-1)
        concated = tf.keras.layers.concatenate(inputs=[d5, c], axis=-1)
        # #print(concated.shape.as_list())

        u1 = upsample2d_block(concated, 64, kernel_size=[9, 5], strides=[9, 1], name_prefix='gen_up_u1')
        # print(f'u1.shape :{u1.shape.as_list()}')

        c1 = tf.tile(c_cast, [1, u1.shape.dims[1].value, u1.shape.dims[2].value, 1])
        # print(f'c1 shape: {c1.shape}')
        u1_concat = tf.concat([u1, c1], axis=-1)
        u1_concat = tf.keras.layers.concatenate(inputs=[u1, c1], axis=-1)
        # print(f'u1_concat.shape :{u1_concat.shape.as_list()}')+

        u2 = upsample2d_block(u1_concat, 128, [3, 5], [1, 1], name_prefix='gen_up_u2')
        # print(f'u2.shape :{u2.shape.as_list()}')
        c2 = tf.tile(c_cast, [1, u2.shape[1], u2.shape[2], 1])
        # u2_concat = tf.concat([u2, c2], axis=-1)
        u2_concat = tf.keras.layers.concatenate(inputs=[u2, c2], axis=-1)

        u3 = upsample2d_block(u2_concat, 64, [4, 8], [2, 2], name_prefix='gen_up_u3')
        # print(f'u3.shape :{u3.shape.as_list()}')
        c3 = tf.tile(c_cast, [1, u3.shape[1], u3.shape[2], 1])
        # u3_concat = tf.concat([u3, c3], axis=-1)
        u3_concat = tf.keras.layers.concatenate(inputs=[u3, c3], axis=-1)

        u4 = upsample2d_block(u3_concat, 32, [4, 8], [2, 2], name_prefix='gen_up_u4')
        c4 = tf.tile(c_cast, [1, u4.shape[1], u4.shape[2], 1])
        u4_concat = tf.keras.layers.concatenate(inputs=[u4, c4], axis=-1)

        u5 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=[3, 9], strides=[1, 1], padding='same',
                                             name='generator_last_deconv')(u4_concat)
        # print(f'u5.shape :{u5.shape.as_list()}')

        self.generator_net = tf.keras.Model(inputs=[inputs, speaker_id], outputs=u5, name="generator_net")

    def call(self, inputs, speaker_id=None):
        out = self.generator_net(inputs=[inputs, speaker_id])

        return out


class discriminator(tf.keras.layers.Layer):
    def __init__(self, reuse=False, scope_name='discriminator'):
        super(discriminator, self).__init__()
        self.reuse = reuse  # def discriminator( inputs, speaker_id=None):
        inner = tf.keras.layers.Input(shape=[36, 512, 1])
        spk = tf.keras.layers.Input(shape=[1, 4])
        c_cast = tf.cast(tf.reshape(spk, [-1, 1, 1, spk.shape[-1]]), tf.float32)
        c = tf.tile(c_cast, [1, inner.shape[1], inner.shape[2], 1])

        concated = tf.concat([inner, c], axis=-1)
        d1 = downsample2d_block(
            inputs=concated, filters=32, kernel_size=[3, 9], strides=[1, 1], padding=[1, 4],
            name_prefix='downsample2d_dis_block1_')
        c1 = tf.tile(c_cast, [1, d1.shape[1], d1.shape[2], 1])
        d1_concat = tf.concat([d1, c1], axis=-1)

        d2 = downsample2d_block(
            inputs=d1_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3],
            name_prefix='downsample2d_dis_block2_')
        c2 = tf.tile(c_cast, [1, d2.shape[1], d2.shape[2], 1])
        d2_concat = tf.concat([d2, c2], axis=-1)
        #
        d3 = downsample2d_block(
            inputs=d2_concat, filters=32, kernel_size=[3, 8], strides=[1, 2], padding=[1, 3],
            name_prefix='downsample2d_dis_block3_')
        c3 = tf.tile(c_cast, [1, d3.shape[1], d3.shape[2], 1])
        d3_concat = tf.concat([d3, c3], axis=-1)

        d4 = downsample2d_block(
            inputs=d3_concat, filters=32, kernel_size=[3, 6], strides=[1, 2], padding=[1, 2],
            name_prefix='downsample2d_diss_block4_')
        c4 = tf.tile(c_cast, [1, d4.shape[1], d4.shape[2], 1])
        d4_concat = tf.concat([d4, c4], axis=-1)
        # print("d4_concat:",d4_concat.shape.as_list())

        c1 = conv2d_layer(inputs=d4_concat, filters=1, kernel_size=[36, 5], strides=[36, 1], padding=[0, 1],
                          activation="sigmoid", name='discriminator-last-conv')
        # print("c1_shape:",c1.shape.as_list())
        c1_red = tf.reduce_mean(c1, keepdims=True)
        # tf.keras.layers.Activation(activation

        self.discriminator_net = tf.keras.Model(inputs=[inner, spk], outputs=c1_red, name="discriminator_net")

    def call(self, inputs, speaker_id=None):
        # print("inputs:",inputs.shape.as_list())
        # print("speaker",speaker_id.shape.as_list())
        out_put = self.discriminator_net(inputs=[inputs, speaker_id])
        # print("c1_shape:", c1.shape.as_list())
        return out_put


class domain_classifier(tf.keras.layers.Layer):
    def __init__(self, batchsize=1, reuse=False, scope_name='classifier'):
        super(domain_classifier, self).__init__()
        self.scope_name = scope_name

        inner = tf.keras.layers.Input(shape=[8, 512, 1])
        d1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(2, 2), activation="relu",
                                    padding="same")(inner)
        d2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 4), strides=(2, 2), activation="relu",
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
        o_r = tf.keras.layers.Reshape([-1, 1, dim])(p)
        # _, _, dim, channels = inner.get_shape().as_list()
        # output_dim = dim * channels
        # inner = layers.Reshape((-1, output_dim))(inner)
        self.x_net = tf.keras.Model(inputs=inner, outputs=o_r, name="x_net")
        # print(self.x_net.summary())

    def call(self, inputs):
        x0 = self.x_net(inputs)

        return x0

if __name__ == '__main__':
    # starganvc = generator_gatedcnn()
    inner = tf.ones([1, 36, 512, 1], dtype="float32")
    print(inner.shape)
    # inner = inner.squezz([0])
    # inner =tf.squeeze(inner,[0])
    # generator = discriminator()
    target_label = tf.constant((1, 1, 1, 0, 0), dtype="float32")
    # target_label = tf.concat(tf.constant([1]),onehot)
    generated_forward = discriminator(inner, speaker_id=target_label)
    print(inner)