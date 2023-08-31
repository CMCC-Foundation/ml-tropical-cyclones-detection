import tensorflow as tf



def ConvBlock(filters, initializer, kernel_size=3, strides=1, apply_batchnorm=False, apply_dropout=False, apply_gaussian_noise=False):
    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", kernel_initializer=initializer, use_bias=False))
    if apply_gaussian_noise:
        layer.add(tf.keras.layers.GaussianNoise(stddev=1.0))
    if apply_batchnorm:
        layer.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        layer.add(tf.keras.layers.Dropout(0.5))
    layer.add(tf.keras.layers.LeakyReLU())
    return layer
