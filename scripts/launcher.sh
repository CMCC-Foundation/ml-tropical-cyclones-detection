#!/bin/bash

# example of training
python -u trainval.py --batch_size 512 --epochs 3 --experiment exp_3 --run_name test_model --shuffle True --shuffle_buffer 2048 --loss mae --network model_v5 --kernel_size 7 --activation linear --aug_type only_tcs --patch_type nearest --label_no_cyclone -1.0