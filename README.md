# Machine Learning Tropical Cyclones Detection

## Overview
The repository provides a Machine Learning (ML) library to setup training and validation of a Tropical Cyclones (TCs) Detection model. ERA5 reanalysis and the International Best Track Archive for Climate Stewardship (IBTrACS) data are used as input and the target, respectively. Input-Output data pairs are provided as TFRecords.

Input drivers:
- 10m wind gust since previous post-processing [ms^{-1}]
- mean sea level pressure [Pa]
- temperature at 300 mb [K]
- temperature at 500 mb [K]

Target:
- TC center row-column coordinates within the 40 x 40 pixels patch 

## Code Structure

_scripts_ folder is the main source to launch training configurations. It contains:

- trainval.py - python script that contains the training-validation procedure that makes use of the provided TFRecords data. 
- launcher.sh - bash script that launches a sample configuration of the training. It can be customized according to the training preferences.

Here is a proof-of-concept training example, that can be launched from _scripts/launcher.sh_:
```bash
python -u trainval.py --batch_size 512 --epochs 3
```


The CLI arguments taken in input by trainval.py script are:
- -bs, --batch_size : Global batch size of data.
- -e, --epochs : Number of epochs through which the model must be trained.

- -rn, --run_name [optional | Default: 'debug'] : Name to be assigned to the trained model. 
- -tm, --trained_model [Optional | Default: None]: The filepath to a trained model to be loaded (ONLY if we want to continue a training).
- -ks, --kernel_size [Optional | Default: None] : Kernel size (only for Model V5 architecture). Possible values: 3,5,7,8,11,13. 
- -s, --shuffle [Optional | Default: 'False'] : Whether to shuffle dataset TFRecords filenames.
- -a, --augmentation [Optional | Default: None] : Whether or not to perform data augmentation.
- -c, --cores [Optional | Default: None] : Number of cores (for multicore CPUs. NOT designed for GPUs).
- -sb, --shuffle_buffer [Optional | Default: None] :  Number of consecutive samples to be shuffled.
- -lr, --learning_rate [Optional | Default: 0.0001] : Learning rate at which the model is trained.
- -ts, --target_scale [Optional | Default: 'False'] : Whether or not to scale the target.
- -l, --loss [Optional | Defualt: 'mae'] : Loss function to be applied. Possible values: mae, mse.
- -n, --network [Optional | Default: 'vgg_v1'] : Neural network used to train the model. Possible values: vgg_v1, vgg_v2, vgg_v3, model_v5.
- -ac, --activation [Optional | Default: 'linear'] : Last layer activation function.
- -at, --aug_type [Optional | Default: 'only_tcs'] : Type of augmentation. Possible values : only_tcs, all_patches.
- -pt, --patch_type [Optional | Default: 'nearest'] : Type of patches used during training. Possible values: alladjacent, nearest.
- -lc, --label_no_cyclone [Optional | Default: None] : The coordinate value assigned to indicate cyclone absence.
- -rg, --regularization_strength [Optional | Default: 'none'] : Regularization strength. Possible values : weak, medium, strong, very_strong, none.




