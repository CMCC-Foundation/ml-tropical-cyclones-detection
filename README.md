# Machine Learning Tropical Cyclones Detection

The repository contains a Machine Learning (ML) library to configure and train a model 
with the use of TFRecords Dataset.

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
- -rn, --run_name [optional] : Name to be assigned to the trained model. Default: 'debug'
- -tm, --trained_model [optional]: The filepath to a trained model to be loaded (ONLY if we want to continue a training). Default: None
- -ks, --kernel_size [optional] : Kernel size (only for Model V5 architecture). Possible values: 3,5,7,8,11,13. Default: None
- -s, --shuffle [optional] : Whether to shuffle dataset TFRecords filenames. Default: 'False'
- -a, --augmentation [optional] : Whether or not to perform data augmentation.
- -c, --cores [optional] : Number of cores (for multicore CPUs. NOT designed for GPUs).
- -sb, --shuffle_buffer [optional] :  Number of consecutive samples to be shuffled.
- -lr, --learning_rate [optional] : Learning rate at which the model is trained.
- -ts, --target_scale [optional] : Whether or not to scale the target.
- -l, --loss [optional] : Loss function to be applied. Possible values: mae, mse.
- -n, --network [optional] : Neural network used to train the model. Possible values: vgg_v1, vgg_v2, vgg_v3, model_v5.
- -ac, --activation [optional] : Last layer activation function.
- -at, --aug_type [optional] : Type of augmentation. Possible values : only_tcs, all_patches.
- -pt, --patch_type [optional] : Type of patches used during training. Possible values: alladjacent, nearest.
- -lc, --label_no_cyclone [optional] : The coordinate value assigned to indicate cyclone absence.
- -rg, --regularization_strength [optional] : Regularization strength. Possible values : weak, medium, strong, very_strong, none.

parser.add_argument( "-rn", "--run_name", default='debug', help="Name to be assigned to the run", required=False)
    parser.add_argument( "-mb", "--model_backup", default=None, help="The filepath to a trained model to be loaded", required=False)
    parser.add_argument( "-ks", "--kernel_size", default=None, type=int, help="Kernel size (only for Model V5)", required=False)
    parser.add_argument( "-s", "--shuffle", default='False', help="Number of consecutive samples to be shuffled", required=False)
    parser.add_argument( "-a", "--augmentation", default='True', help="Whether or not to perform data augmentation", required=False)
    parser.add_argument( "-c", "--cores", default=None, type=int, help="Number of cores (for local mirrored strategy)", required=False)
    parser.add_argument( "-sb", "--shuffle_buffer", default=None, type=int, help="Number of consecutive samples to be shuffled", required=False)
    parser.add_argument( "-lr", "--learning_rate", default=0.0001, type=float, help="Learning rate at which the model is trained", required=False)
    parser.add_argument( "-ts", "--target_scale", default='False', choices=['True','False'], help="Whether or not to scale the target", required=False)
    parser.add_argument( "-l", "--loss", default=Losses.MAE.value[0], choices=[l.value[0] for l in Losses], help="Loss function to be applied", required=False)
    parser.add_argument( "-n", "--network", default=Network.VGG_V1.value, choices=[n.value for n in Network], help="Neural network used to train the model", required=False)
    parser.add_argument( "-ac", "--activation", default=Activation.LINEAR.value, choices=[a.value for a in Activation], help="Last layer activation function", required=False)
    parser.add_argument( "-at", "--aug_type", default=AugmentationType.ONLY_TCS.value, choices=[at.value for at in AugmentationType], help="Type of augmentation", required=False)
    parser.add_argument( "-pt", "--patch_type", default=PatchType.NEAREST.value, choices=[pt.value for pt in PatchType], help="Type of patches used during training", required=False)
    parser.add_argument( "-lc", "--label_no_cyclone", default=str(LabelNoCyclone.NONE.value), choices=[str(lnc.value) for lnc in LabelNoCyclone], help="The label assigned to the cyclone", required=False)
    parser.add_argument( "-rg", "--regularization_strength", default=RegularizationStrength.NONE.value[0], choices=[r.value[0] for r in RegularizationStrength], help="Regularization strength", required=False)




