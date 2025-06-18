#!/bin/bash

echo Start the program " "
echo ""

set -e

CONFIG_FILE=config/cnn.toml
n_nodes=1
n_devices=1

echo "Performing experiments"

export MPLCONFIGDIR=~/.matplotlib

TRAINING_FILE=training.py
#mpirun -n $n_devices -- python $TRAINING_FILE -c $CONFIG_FILE -d $n_devices -n $n_nodes
#OMP_NUM_THREADS=1 torchrun --nnodes $n_devices --nproc-per-node gpu --standalone $TRAINING_FILE -c $CONFIG_FILE -d $n_devices -n $n_nodes
python $TRAINING_FILE -c $CONFIG_FILE -d 1 -n 1

# End of script
echo "Program ended"