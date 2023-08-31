import tensorflow as tf


def coo_rot180(data):
    X,y = data
    patch_size = X.shape[0]
    X = tf.image.rot90(X, k=2)
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [-y[0] + patch_size -1, -y[1] + patch_size -1]
    return (X,y1)


def coo_left_right(data):
    X,y = data
    patch_size = X.shape[0]
    X = tf.image.flip_left_right(X)
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [y[0], - y[1] + patch_size -1]
    return (X,y1)


def coo_up_down(data):
    X,y = data
    patch_size = X.shape[0]
    X = tf.image.flip_up_down(X)
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [- y[0] + patch_size -1, y[1]]
    return (X,y1)
