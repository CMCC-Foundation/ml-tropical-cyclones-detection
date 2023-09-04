# Machine Learning Tropical Cyclones Detection

## Credit
- Gabriele Accarino
- Davide Donno
- Francesco Immorlano
- Donatello Elia
- Giovanni Aloisio

## Overview
The repository provides a Machine Learning (ML) library to setup training and validation of a Tropical Cyclones (TCs) Detection model. ERA5 reanalysis and the International Best Track Archive for Climate Stewardship (IBTrACS) data are used as input and the target, respectively. Input-Output data pairs are provided as TFRecords.

Input drivers:
- 10m wind gust since previous post-processing [ms**{-1}]
- mean sea level pressure [Pa]
- temperature at 300 mb [K]
- temperature at 500 mb [K]

Target:
- TC center row-column coordinates within the 40 x 40 pixels patch 

## Code Structure

The _train.py_ script (located in _src_ folder) allows running both training and validation on input data. The _data_ folder should be located at the same level of _resources_ and _src_ folders. 

Here is an example of training.

```bash
cd src
python -u train.py --config config.toml
```

The _train.py_ script takes advantage of the Command Line Interface (CLI) to pass a training configuration file (_config.toml_ in the example above) to the script that is useful for both training and validation of the model.

The configuration file must be prepared in toml format. The configuration file is structured as follows:

- dirs : directories
    - main: relative path from training source file to repository folder (i.e., the one that contains the _src_, _data_, _resources_, etc folders).
    - data: relative path to _data_ folder.
    - model: the relative path to a trained model to be loaded (ONLY if we want to continue a training).
    - experiments: the relative path to the directory that will contain the output.
    - dataset: relative path to the direcotry that contains the dataset to be used during training
    - scaler: relative path to the destination directory that will contain the scaler
    - scaler_filename: output scaler filename

- run : run-specific arguments
    - name: name of the run (i.e., the name of the output folder that will contain training results)
    - seed: the seed of this training (to repeat experiments)

- data : information to data provided as input
    - patch_size: size of an input patch.
    - drivers: list of variables that will be used as input during training.
    - targets: list of variables that will be used as output during training.
    - drivers_shape: shape of the input drivers (it should always be equal to [patch_size, patch_size]).
    - targets_shape: shape of the output variables (since we have to predict row-col coordinates it should always be equal to [2,]).
    - patch_type: type of patches that will be used during training (it can only assume value _nearest_ or _alladjacent_)-
    - force_scaler_compute: boolean that indicates whether or not to force the scaler computation even if it has already been previously computed.

- model: parameters related to the model configuration
    - network: string embedding a Python code that loads a model from the library
    - loss: string embedding a Python code that loads a Tensorflow loss function
    - optimizer: string embedding a Python code that loads a Tensorflow optimizer function
    - metrics: list of strings embedding Python codes that load Tensorflow metrics functions
    - callbacks: list of strings embedding Python codes that load Tensorflow (or custom) callback functions

- training: parameters related to the training
    - epochs: number of epochs used to train the model
    - batch_size: size of a batch of data that will be fed to the network
    - shuffle_buffer: size of the buffer used for shuffling
    - label_no_cyclone: label that indicates the TC absence within the patch
    - only_tcs: whether to augment only TC-patches or all the patches.
    - augmentation: augmentation functions are expressed as a key-value pair. key is the name associated to the augmentation and the value is a string that embeds Python code that load a Tensorflow (or custom) augmentation function.


## Python3 Environment 
The code has been tested on Python 3.8.16 with the following dependencies:
- keras=2.12.0
- numpy=1.23.5
- pandas=1.5.3
- psutil=5.9.5
- scikit-learn=1.2.2
- tensorflow-macos=2.12.0
