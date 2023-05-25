# Machine Learning Tropical Cyclones Detection

The repository contains a Machine Learning (ML) library to configure and train a model 
with the use of TFRecords Dataset.

_scripts_ folder is the main source to launch training configurations. It contains:

- trainval.py - python script that contains the training-validation procedure that makes use of the provided TFRecords data. 
- launcher.sh - bash script that launches a sample configuration of the training. It can be customized according to the training preferences.

To run a proof-of-concept training go to \textit{scripts} folder and launch launcher.sh bash script. 

The CLI arguments taken in input by trainval.py script are:
- -bs, --batch_size : Global batch size of data.
- -e, --epochs : Number of epochs through which the model must be trained.
- -rn, --run_name : Name to be assigned to the trained model.
- -tm, --trained_model: The filepath to a trained model to be loaded (ONLY if we want 
to continue a training).
- -ks, --kernel_size : Kernel size (only for Model V5 architecture). Possible values: 3,5,7,8,11,13.
- -s, --shuffle : Whether to shuffle dataset TFRecords filenames.
- -a, --augmentation : Whether or not to perform data augmentation.
- -c, --cores : Number of cores (for multicore CPUs. NOT designed for GPUs).
- -sb, --shuffle_buffer :  Number of consecutive samples to be shuffled.
- -lr, --learning_rate : Learning rate at which the model is trained.
- -ts, --target_scale : Whether or not to scale the target.
- -l, --loss : Loss function to be applied. Possible values: mae, mse.
- -n, --network : Neural network used to train the model. Possible values: vgg_v1, vgg_v2, vgg_v3, model_v5.
- -ac, --activation : Last layer activation function.
- -at, --aug_type : Type of augmentation. Possible values : only_tcs, all_patches.
- -pt, --patch_type : Type of patches used during training. Possible values: alladjacent, nearest.
- -lc, --label_no_cyclone : The coordinate value assigned to indicate cyclone absence.
- -rg, --regularization_strength : Regularization strength. Possible values : weak, medium, strong, very_strong, none.
