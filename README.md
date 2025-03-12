# Machine Learning Tropical Cyclones Detection

## Overview
The repository provides a Machine Learning (ML) library to setup training and validation of a Tropical Cyclones (TCs) Detection model. ERA5 reanalysis and the International Best Track Archive for Climate Stewardship (IBTrACS) data are used as input and the target, respectively. Input-Output data pairs are provided as Zarr data stores.

Input drivers:
- 10m wind gust [ $\frac{m}{s}$]
- 10m wind gust since previous post-processing [ $\frac{m}{s}$]
- mean sea level pressure [Pa]
- relative vorticity at 850 mb [ $s^{-1}$]
- temperature at 300 mb [K]
- temperature at 500 mb [K]

Target:
- TC center row-column coordinates within the 40 x 40 pixels patch 

## Code Structure

The _train.py_ script (located in _src_ folder) allows running both training and validation on input data. The _data_ folder should be located at the same level of _resources_ and _src_ folders. 

Here is an example of training.

```bash
cd src
python -u train.py --config config.toml --devices 1 --num_nodes 1
```
The _train.py_ script takes advantage of the Command Line Interface (CLI) to pass additional arguments useful for both training and validation of the model. In particular:

- `--config` specifies the path to the *config.toml* file where the training configuration is stored. Pre-defined configuration file are located under `src/config/`
- `--devices` argument defines the number of GPU devices per node to run the training on.
- `--num_nodes` argument defines the total number of nodes that will be used.

The total number of GPUs used during the training can be evinced by simply multiplying `devices * num_nodes`.

With regards to the configuration file, it must be prepared in toml format. The configuration file is structured as follows:

- run : generic arguments
    - seed: the seed of this training (for reproducibility purposes)
    - use_case: it can be either `cnn` or `gnn` to select the proper type of ML model

- dir : directories
    - experiment: path to store the current experiment
    - train: path to the stored train patches
    - valid: path to the stored valid patches
    - run: name of the current experiment run
    - scaler
        - mean: path to mean netcdf
        - std: path to standard deviation netcdf

- torch: pytorch configuration arguments
    - matmul_precision: set the precision of the matix multiplications
    - dtype: data type that will be used for the data

- lightning: pytorch-lightning configuration arguments
    - accelerator: defines the hardware component that will be used during training (CPU, GPU, ecc)
    - precision: precision of the internal computations during model training

- model: defines arguments related to model initialization
    - monitor: name of the loss to be monitored (for model checkpoints)
    - cls: name of the class of the model (taken from `tropical_cyclone` library located in `resources/library/tropical_cyclone`)
    - metrics: list of additional metrics to be computed during training
    - args: dictionary of model arguments to be passed in `model.cls`

- data : information to data provided as input
    - drivers: list of variables that will be used as input during training.
    - targets: list of variables that will be used as output during training.
    - label_no_cyclone: label applied to indicate the absence of a TC in the patch (only for the cnn case)

- loss: informations about loss function
    - cls: class name of the loss function
    - args: arguments passed to the loss (can be none)
- oprimizer: informations about oprimizer function
    - cls: class name of the optimizer
    - args: arguments passed to the optimizer (can be none)
- scheduler: informations about scheduler function
    - cls: class name of the lr scheduler
    - args: arguments passed to the lr scheduler (can be none)

- train: parameters related to the training
    - epochs: number of epochs used to train the model
    - batch_size: size of a batch of data that will be fed to the network
    - drop_remainder: whether or not to drop the last batch if the number of dataset elements is not divisible by the batch size
    - accumulation_steps: number of gradient accumulation steps before calliing backward propagation
 
## How to

### Download IBTrACS

Since the TC Detection case study relies on IBTrACS dataset, it must be downloaded from the official repository. 

- Go to the IBTrACS page on the NOAA site at this link: https://www.ncei.noaa.gov/products/international-best-track-archive
- Go to the "_Access Methods_" paragraph
- On the central panel (named "_Comma Separated Values (CSV)_"), click on **CSV Data**
- From the page that opens (at https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/), any IBTrACS dataset can be downloaded (e.g., filtered by basin, global, ecc)

### Download ERA5 data

To download ERA5 data you must need a CDS account and the set of IBTrACS for which the reated ERA5 data is gathered. The script `era5_gathering.py` under `src/dataset` can be used for this purpose.

## Python3 Environment 
The code has been tested on Python 3.11.2 with the following dependencies:

- dask == 2023.7.0
- lightning == 2.0.9
- mpi4py == 3.1.4
- munch == 4.0.0
- netcdf4 == 1.6.0
- numpy == 1.24.1
- pandas == 1.5.3
- pytorch-lightning == 2.0.9
- scikit-learn == 1.2.2
- scipy == 1.10.1
- timm == 0.9.7
- toml == 0.10.2
- torch == 2.0.1+cu118
- torch-geometric == 2.5.0
- torchaudio == 2.0.2+cu118
- torchmetrics == 1.1.2
- torchvision == 0.15.2+cu118
- xarray == 2022.6.0
