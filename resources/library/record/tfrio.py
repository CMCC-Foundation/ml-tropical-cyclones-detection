import tensorflow as tf
import numpy as np
import os

from .feature import tensor_feature





class DriverInfo():
    """
    Class storing information about driver variables, shape and datatype.

    """
    def __init__(self, vars, shape, dtype=tf.float32):
        self.vars = vars
        self.shape = shape
        self.dtype = dtype





class TensorCoder():
    """
    Tensor Encoder and Decoder class. It provides two functionalities:
    1) encodes variables tensors into a serialized version to be stored as TFRecord
    2) decodes TFRecords serialized data into tensors usable in ML pipelines

    """
    def __init__(self, drivers_info=[]):
        """
        Parameters
        ----------
        drivers_info : list(DriverInfo)
            A list of driver information for the decoder to be read from file.

        """
        self.drivers_info = drivers_info


    def encoding_fn(self, **kwargs):
        """
        Builds a serialized version of the dataset. kwargs must be tensors.
    
        """
        # feature dictionary
        feature = {}
        # for each keyword argument
        for key, value in kwargs.items():
            # add the serialized variable to feature dictionary
            feature.update({key:tensor_feature(value)})
        # define features using the feature dictionary
        features = tf.train.Features(feature=feature)
        # serialize data examples
        return tf.train.Example(features=features).SerializeToString()


    def decoding_fn(self, serialized_data):
        """
        Decoding function for a dataset written to disk as tensor_encoding_fn()
        
        """
        # define features dictionary
        features = {}
        # cycle with respect to driver info list
        for info in self.drivers_info:
            for var in info.vars:
                # add features for each variable
                features.update({var : tf.io.FixedLenFeature([], tf.string)})
        # parse the serialized data so we get a dict with our data.
        parsed_data = tf.io.parse_single_example(serialized_data, features=features)
        # accumulator for data elements
        data = []
        # for each output tensor driver
        for info in self.drivers_info:
            # parsed single examples to tensors and stack them
            if len(info.vars) == 1 and len(info.shape) == 1:
                data_tensor = tf.ensure_shape(tf.io.parse_tensor(serialized=parsed_data[info.vars[0]], out_type=info.dtype), shape=info.shape)
            else:
                data_tensor = tf.stack([tf.ensure_shape(tf.io.parse_tensor(serialized=parsed_data[var], out_type=info.dtype), shape=info.shape) for var in info.vars], axis=-1)
            data.append(data_tensor)
        return tuple(data)
