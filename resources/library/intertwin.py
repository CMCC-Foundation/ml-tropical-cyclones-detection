import tensorflow as tf
from .scaling import TFScaler
from .record.tfrio import TensorCoder



class DatasetBuilder():
    """
    Dataset Builder default class. It builds a dataset with the preferred characteristics. It is general purpose, so 
    that it can be used with every desired ML workflow.
    
    """
    def __init__(self):
        self.AUTOTUNE = tf.data.AUTOTUNE


    def batch(self, batch_size=None, drop_remainder=False):
        # check if dataset is defined
        self._check_dataset()
        # separate in batches
        if batch_size:
            self.dataset = self.dataset.batch(batch_size, drop_remainder=drop_remainder, num_parallel_calls=self.AUTOTUNE)
        else:
            self.dataset = self.dataset.batch(self.count, drop_remainder=drop_remainder, num_parallel_calls=self.AUTOTUNE)
        return self


    def shuffle(self, shuffle_buffer=None):
        # check if dataset is defined
        self._check_dataset()
        # shuffle if necessary
        if shuffle_buffer:
            self.dataset = self.dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        return self


    def scale(self, x_scaler:TFScaler=None, y_scaler:TFScaler=None):
        # check if dataset is defined
        self._check_dataset()
        def apply_scaling(data):
            X, y = data
            if x_scaler:
                X = x_scaler.transform(X)
            if y_scaler:
                y = y_scaler.transform(y)
            return (X, y)
        # scale the data
        self.dataset = self.dataset.map(lambda X,y: (apply_scaling((X,y))), num_parallel_calls=self.AUTOTUNE)
        return self


    def resize(self, shape:tuple=None):
        # check if dataset is defined
        self._check_dataset()
        def apply_resize(data):
            resized_data = []
            for x in data:
                resized_data.append(tf.image.resize(x,shape,tf.image.ResizeMethod.NEAREST_NEIGHBOR))
            return tuple(resized_data)
        if shape:
             self.dataset = self.dataset.map(lambda X,y: (apply_resize((X,y))), num_parallel_calls=self.AUTOTUNE)
        return self


    def mask(self, mask):
        # check if dataset is defined
        self._check_dataset()
        # apply mask function
        def apply_mask(data):
            X,y = data
            y_masked = tf.where(tf.math.is_nan(y), tf.ones_like(y) * mask, y)
            return (X, y_masked)
        # apply mask on target if label_no_cyclone is provided
        if mask:
            self.dataset = self.dataset.map(lambda X,y: (apply_mask((X,y))), num_parallel_calls=self.AUTOTUNE)
        return self


    def repeat(self):
        # check if dataset is defined
        self._check_dataset()
        # set number of epochs that can be repeated on this dataset
        self.dataset = self.dataset.repeat(count=self.epochs)
        return self


    def optimize(self):
        # check if dataset is defined
        self._check_dataset()
        # add parallelism option
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        options.experimental_threading.max_intra_op_parallelism = 1
        self.dataset = self.dataset.with_options(options)
        # prefetch
        self.dataset = self.dataset.prefetch(buffer_size=self.AUTOTUNE)
        return self


    # private function
    def _check_dataset(self):
        if not hasattr(self, 'dataset'):
            raise Exception('The dataset variable is not defined. Try calling assemble_dataset() first.')
        
    

    # protected function
    def _count_filenames_elems(self, filenames):
        """
        Counts the number of elements present in the passed dataset files.
        
        """
        return sum([int(fname.split('/')[-1].split('.tfrecord')[0].split('_')[-1]) for fname in filenames])
    



class InterTwinTFRDatasetBuilder(DatasetBuilder):
    """
    Builder class for EFlows dataset stored into TFRecords files. Building steps:
        1. source - configure the data sources to be retrieved
        2. augment (optional) - whether to online augment the TC occurrences
        3. interleave - to create the tf.data.Dataset
        4. get_dataset - returns the tf.data.Dataset

    """
    def __init__(self, epochs, tensor_coder:TensorCoder):
        """
        Parameters
        ----------
        tensor_coder : TensorCoder
            Coder function to help decode data contained in the dataset
        
        """
        super().__init__()
        # set the number of epochs
        self.epochs = epochs

        # coder for tfrecord decoding
        self.tensor_coder = tensor_coder

        # initialize important data structures
        self.cyc_dict = {
            'filenames' : [],
            'datasets' : [],
            'weights' : [],
            'patch_type' : []
        }
        self.nocyc_dict = {
            'filenames' : [],
            'datasets' : [],
            'weights' : [],
            'patch_type' : []
        }
        
        # counter for the number of elements of the dataset
        self.count = 0



    def source(self, filenames, is_cyc=False, weight=1, patch_type=None):
        """
        Adds a new source of filenames for the dataset. The source must contain 
        elements of identical type.

        Parameters
        ----------
        filenames : list(str)
            List of filenames used as source for the dataset
        is_cyc : bool
            Whether or not the source files contain cyclone or not
        weight : int
            Specifies the weight as a ratio of the number of elements contained in it

        """
        # determine the dictionary in which the data must be saved
        if is_cyc:
            data_dict = self.cyc_dict
        else:
            data_dict = self.nocyc_dict 

        # count the number of elements of the dataset and increment count variable
        self.count += self._count_filenames_elems(filenames=filenames)

        # save the filenames
        data_dict['filenames'] += filenames

        # create dataset
        data_dict['datasets'] += [tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTOTUNE).map(self.tensor_coder.decoding_fn, num_parallel_calls=self.AUTOTUNE)]

        # add the weight of the added dataset
        data_dict['weights'] += [weight]

        # add the type of patches added
        data_dict['patch_type'] += [patch_type]

        return self



    def augment(self, aug_fns, only_tcs=True):
        """
        Create and adds to the dataset the augmented versions of the data

        """
        # update the number of elements of this dataset
        self.count += self._count_filenames_elems(filenames=self.cyc_dict['filenames']) * len(aug_fns.keys())

        # define augmented datasets for each augmentation function
        aug_cyc_datasets = [(tf.data.TFRecordDataset(self.cyc_dict['filenames'], num_parallel_reads=self.AUTOTUNE).map(self.tensor_coder.decoding_fn, num_parallel_calls=self.AUTOTUNE).map(lambda x,y: (aug_fn((x,y))), num_parallel_calls=self.AUTOTUNE)) for aug_fn in aug_fns.values()]

        # add augmented datasets to dataset list
        self.cyc_dict['datasets'] += aug_cyc_datasets

        # add weights from the augmented datasets
        self.cyc_dict['weights'] += [1 for _ in range(len(aug_cyc_datasets))]

        # if augmentation involves nocyclone data, too
        if not only_tcs:
            for w, pt in zip(self.nocyc_dict['weights'], self.nocyc_dict['patch_type']):

                # get only filenames of this patch type
                filenames = [f for f in self.nocyc_dict['filenames'] if pt in f]

                # update the number of elements of this dataset
                self.count += self._count_filenames_elems(filenames=filenames) * len(aug_fns.keys())

                # define augmented datasets for each augmentation function
                aug_nocyc_datasets = [(tf.data.TFRecordDataset(filenames, num_parallel_reads=self.AUTOTUNE).map(self.tensor_coder.decoding_fn, num_parallel_calls=self.AUTOTUNE).map(lambda x,y: (aug_fn((x,y))), num_parallel_calls=self.AUTOTUNE)) for aug_fn in aug_fns.values()]

                # add augmented datasets to dataset list
                self.nocyc_dict['datasets'] += aug_nocyc_datasets

                # add weights from the augmented datasets
                self.nocyc_dict['weights'] += [w for _ in range(len(aug_nocyc_datasets))]

        return self



    def assemble_dataset(self, interleave=True):
        """
        Assemble a tf.data.Dataset that has been built from sources and augmentations.

        Parameters
        ----------
        interleave : bool | default : True
            Whether or not to equally interleave the data elements coming from different sources. Typically, 
            • interleave to True is needed only during training
            • interleave to False is needed only during inference
        
        """
        # get the number of repeats of choice dataset
        choice_count = self.count * self.epochs
        
        # create a list of all datasets to interleave on
        datasets = self.cyc_dict['datasets'] + self.nocyc_dict['datasets']

        if interleave:
            # get the interleave of all datasets
            data_interleave = self.__get_interleave(cyc_weights=self.cyc_dict['weights'], nocyc_weights=self.nocyc_dict['weights'])
            
            # compute the choice dataset with the interleave
            choice_dataset = tf.data.Dataset.from_tensor_slices(data_interleave).repeat(count=choice_count)
        
        else:
            # select the order in which the augmented samples must be interleaved
            choice_dataset = tf.data.Dataset.range(len(datasets)).repeat(count=choice_count)
        
        # statically interleave elements from all the datasets
        self.dataset = tf.data.experimental.choose_from_datasets(datasets=datasets, choice_dataset=choice_dataset)
        
        return self



    def mask(self, label_no_cyclone):
        # check if dataset is defined
        self._check_dataset()
        # apply mask function
        def apply_mask(data):
            X,y = data
            y_masked = tf.where(y < 0, label_no_cyclone, y)
            return (X, y_masked)
        # apply mask on target if label_no_cyclone is provided
        if label_no_cyclone:
            self.dataset = self.dataset.map(lambda X,y: (apply_mask((X,y))), num_parallel_calls=self.AUTOTUNE)
        return self
    


    # private function
    def __get_interleave(self, cyc_weights, nocyc_weights):
        """
        Returns the interleaved dataset indexes based on cyclone and nocyclone weights.

        """
        # define cyclone interleave
        cyc_interleave = [i for i,w in enumerate(cyc_weights) for _ in range(w)]
        
        # define nocyclone interleave
        nocyc_interleave = [i+len(cyc_interleave) for i,w in enumerate(nocyc_weights) for _ in range(w)]

        # compute the number of blocks + the remainder of the interleaves
        blocks, remainder = len(nocyc_interleave) // len(cyc_interleave), len(nocyc_interleave) % len(cyc_interleave)
        
        interleave = []
        for i in cyc_interleave:
            interleave += [i] + nocyc_interleave[i*blocks:(i+1)*blocks]
        if remainder:
            interleave += nocyc_interleave[-remainder:]
        
        return tf.cast(interleave, dtype=tf.int64)
