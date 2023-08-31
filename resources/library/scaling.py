import tensorflow as tf


class TFScaler():
    """
    Scaler class for Tensorflow datasets. It takes in input a 
    tf.data.Dataset and returns an object that can be used for scaling 

    """
    def __init__(self, ndims=2, feature_range=(0,1), dtype=tf.float32) -> None:
        """
        Parmeters
        ---------
        ndims : int
            Number of dimensions of the input data, excluding the batch (usually the first dim).
        feature_range : tuple | default : (0,1)
            output range of input features
        dtype : tf.dtype
            datatype of the computed scaler and output data.

        """
        self.ndims = ndims
        self.feature_range = feature_range
        self.dtype = dtype
        self.axis = tuple([i for i in range(self.ndims+1)])


    def feed_strategy(self, strategy):
        """
        It is used to replicate the scaling variables across the strategy replicas in sync

        """
        with strategy.scope():
            self.scale_ = tf.Variable(self.scale_)
            self.data_max_ = tf.Variable(self.data_max_)
            self.data_min_ = tf.Variable(self.data_min_)
            self.data_range_ = tf.Variable(self.data_range_)


    def fit_dataset(self, dataset:tf.data.Dataset, data_id=None):
        """
        Finds min and max from the provided tf.data.Dataset.

        Parameters
        ----------
        dataset : tf.data.Dataset
            The passed dataset to be fit.
        data_id : int (default=None)
            Index that indicates which of the data element will be scaled, if provided. Otherwise, it will be assumed that the dataset contains only one data element.
            Example:
            # dataset is composed of X, y and z.
            dataset = tf.data.Dataset.zip((X, y, z))
            # suppose we want to compute scaler on X
            scaler.fit_dataset(dataset=dataset, data_id=0)

        """
        # for each batch in the dataset, compute the min and max, then we will find min and max among them
        self.data_min_ = None
        self.data_max_ = None
        for i,data in enumerate(dataset):
            if not data_id:
                data = (data)
                data_id = 0
            # compute min and max on this batch
            dmax, dmin = self.__compute_min_and_max(data[data_id])
            # add one dimension on axis 1 
            dmax, dmin = tf.expand_dims(dmax, axis=1), tf.expand_dims(dmin, axis=1)  
            if not i:
                # during first iteration data_max_ and data_min_ must be created
                self.data_max_, self.data_min_ = dmax, dmin
            else:
                # during other iterations append the dmax and dmin to data_max_ and data_min_
                self.data_max_, self.data_min_ = tf.concat([self.data_max_, dmax], axis=1), tf.concat([self.data_min_, dmin], axis=1)
        self.data_max_, self.data_min_ = tf.cast(tf.reduce_max(self.data_max_, axis=1), dtype=self.dtype), tf.cast(tf.reduce_min(self.data_min_, axis=1), dtype=self.dtype)
        self.__fit()


    def fit(self, input_tensor):
        self.data_max_, self.data_min_ = self.__compute_min_and_max(input_tensor)
        self.__fit()


    def fit_transform(self, input_tensor):
        self.data_max_, self.data_min_ = self.__compute_min_and_max(input_tensor)
        self.__fit()
        output_tensor = self.transform(input_tensor=input_tensor)
        return output_tensor


    def transform(self, input_tensor):
        output_tensor = input_tensor * self.scale_
        output_tensor += self.min_
        return tf.cast(output_tensor, dtype=self.dtype)


    def __compute_min_and_max(self, input_tensor):
        """
        Computes min and max of the passed tf.Tensor, returning them as a tuple.

        """
        data_min_ = tf.cast(tf.reduce_min(input_tensor, axis=self.axis), dtype=self.dtype)
        data_max_ = tf.cast(tf.reduce_max(input_tensor, axis=self.axis), dtype=self.dtype)
        return (data_max_, data_min_)


    def __fit(self):
        """
        Computes the data_range_, scale_ and min_ from data_min_, data_max_ and feature_range
        
        """
        self.data_range_ = self.data_max_ - self.data_min_
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / self.data_range_
        self.min_ = self.feature_range[0] - self.data_min_ * self.scale_
