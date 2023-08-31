import tensorflow as tf



def tensor_feature(value):
    """
    Returns a bytes_list from a tensor.
    
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(tf.convert_to_tensor(value)).numpy()]) )

