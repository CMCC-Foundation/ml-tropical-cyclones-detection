# STANDARD LIBRARY IMPORTS
from sklearn.metrics import mean_absolute_error
from os import makedirs, listdir, environ
from os.path import join, exists
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import shutil
# import gdown
import munch
import toml
import sys

# parse the CLI arguments
parser = argparse.ArgumentParser(prog='train.py', description="Training pipeline for eFlows4HPC workflow")
parser.add_argument( "-c", "--config", type=str, help="Configuration file for this training", required=True)
args = parser.parse_args()

# read TOML configuration file
config = munch.munchify(toml.load(args.config))

# CUSTOM LIBRARY IMPORTS
sys.path.append('../resources/')
import library as lb

# SEED THE RUN
lb.utils.seed_everything(config.run.seed)

# # DIRECTORY MACROS
EXPERIMENTS_DIR = config.dirs.experiments
SCALER_DIR = config.dirs.scaler
RUN_DIR = join(EXPERIMENTS_DIR, config.run.name)
CHECKPOINTS_DIR = join(RUN_DIR, 'checkpoints')
TENSORBOARD_DIR = join(RUN_DIR, 'tensorboard')

# FILE MACROS
X_SCALER_FILE = join(SCALER_DIR, config.dirs.scaler_filename)
LOSS_METRICS_HISTORY_CSV = join(RUN_DIR, 'loss_metrics_history.csv')
CHECKPOINT_FNAME = join(CHECKPOINTS_DIR, 'model_{epoch:02d}')
BENCHMARK_HISTORY_CSV = join(RUN_DIR, 'benchmark_history.csv')
TRAINVAL_TIME_CSV = join(RUN_DIR, 'trainval_time.csv')
LAST_MODEL = join(RUN_DIR, 'last_model')
LOG_FILE = join(RUN_DIR, 'run.log')

# CREATE DIRECTORIES
makedirs(RUN_DIR, exist_ok=True)
makedirs(SCALER_DIR, exist_ok=True)
makedirs(EXPERIMENTS_DIR, exist_ok=True)
makedirs(TENSORBOARD_DIR, exist_ok=True)
makedirs(CHECKPOINTS_DIR, exist_ok=True)


# UTILITY FUNCTIONS
def compute_tf_scaler(cyc_fnames, rnd_fnames, adj_fnames, tensor_coder, scaler_file, config):
    # compute if we are forcing the computation or the scaler does not exist
    if config.data.force_scaler_compute or not exists(scaler_file):
        # get basic training dataset
        scaler_dataset_builder = (lb.intertwin.InterTwinTFRDatasetBuilder(epochs=1, tensor_coder=tensor_coder).source(filenames=cyc_fnames, is_cyc=True, weight=1, patch_type=lb.enums.PatchType.CYCLONE.value).source(filenames=adj_fnames, is_cyc=False, weight=3, patch_type=lb.enums.PatchType.NEAREST.value).source(filenames=rnd_fnames, is_cyc=False, weight=1, patch_type=lb.enums.PatchType.RANDOM.value).assemble_dataset(interleave=True).batch(batch_size=config.training.batch_size).optimize())
        # get the dataset from the builder
        dataset = scaler_dataset_builder.dataset
        # create the scaler
        x_scaler = lb.scaling.TFScaler()
        # fit the dataset with the scaler
        x_scaler.fit_dataset(dataset=dataset, data_id=0)
        # save the scaler
        lb.utils.joblib_save(dst=scaler_file, data=x_scaler)
    # load scaler from disk
    x_scaler = lb.utils.joblib_load(src=scaler_file)
    return x_scaler



def load_dataset(cyc_fnames, rnd_fnames, adj_fnames, x_scaler, tensor_coder, config):
    # get label no cyclone
    label_no_cyclone = None
    if hasattr(config.training, 'label_no_cyclone'):
        label_no_cyclone = config.training.label_no_cyclone
    # setup augmentation functions
    aug_fns = {}
    for key, value in config.training.augmentation.items():
        aug_fns.update({key:eval(value)})
    # create dataset builder for eFlows use case
    tf_builder = (lb.intertwin.InterTwinTFRDatasetBuilder(epochs=config.training.epochs, tensor_coder=tensor_coder).source(filenames=cyc_fnames, is_cyc=True, weight=1, patch_type=lb.enums.PatchType.CYCLONE.value).source(filenames=adj_fnames, is_cyc=False, weight=3, patch_type=lb.enums.PatchType.NEAREST.value).source(filenames=rnd_fnames, is_cyc=False, weight=1, patch_type=lb.enums.PatchType.RANDOM.value).augment(aug_fns=aug_fns, only_tcs=config.training.only_tcs).assemble_dataset(interleave=True).shuffle(shuffle_buffer=config.training.shuffle_buffer).batch(batch_size=config.training.batch_size).mask(label_no_cyclone=label_no_cyclone).scale(x_scaler=x_scaler).repeat().optimize())
    # get the number of steps per epoch
    steps_per_epoch = tf_builder.count // config.training.batch_size
    # get the dataset
    dataset = tf_builder.dataset
    return dataset, steps_per_epoch



def load_compiled_model(config, mirrored_strategy):
    # parse model configurations
    loss = eval(config.model.loss)
    callbacks = [eval(cllb) for cllb in config.model.callbacks]
    with mirrored_strategy.scope():
        try:
            optimizer = dict(config.model.optimizer)
            for key, value in optimizer.items():
                optimizer[key] = eval(value)
        except:
            optimizer = eval(config.model.optimizer)
        metrics = [eval(mtr) for mtr in config.model.metrics]
        if hasattr(config.run, 'imported'):
            model:tf.keras.Model = tf.keras.models.load_model(config.run.imported)
        else:
            model:tf.keras.Model = eval(config.model.network)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model, loss, optimizer, metrics, callbacks


def get_data_filenames(dataset_dir, years, patch_type=lb.enums.PatchType.NEAREST.value, runtype='training'):
    """
    Returns the filenames for eFlows TC detection workflow. Cyclone, Random and Adjacent patches filenames are returned.

    Parameters
    ----------
    dataset_dir: pathlike
        Directory containing the data files.
    years: list[int]
        Data years to be considered for the dataset.
    patch_type: str
        Type of adjacent patches to consider.
    
    """
    cyc_fnames = sorted([join(dataset_dir,f) for f in listdir(dataset_dir) if f.endswith('.tfrecord') and f.startswith(lb.enums.PatchType.CYCLONE.value) and int(f.split('_')[1]) in years])
    if runtype=='training':
        rnd_fnames = sorted([join(dataset_dir,f) for f in listdir(dataset_dir) if f.endswith('.tfrecord') and f.startswith(lb.enums.PatchType.RANDOM.value) and int(f.split('_')[1]) in years])
        if patch_type == lb.enums.PatchType.NEAREST.value:
            adj_fnames = sorted([join(dataset_dir,f) for f in listdir(dataset_dir) if f.endswith('.tfrecord') and f.startswith(lb.enums.PatchType.NEAREST.value) and int(f.split('_')[1]) in years])
        elif patch_type == lb.enums.PatchType.ALLADJACENT.value:
            adj_fnames = sorted([join(dataset_dir,f) for f in listdir(dataset_dir) if f.endswith('.tfrecord') and f.startswith(lb.enums.PatchType.ALLADJACENT.value) and int(f.split('_')[1]) in years])
        return cyc_fnames, rnd_fnames, adj_fnames
    elif runtype=='test':
        no_cyc_fnames = sorted([join(dataset_dir,f) for f in listdir(dataset_dir) if f.endswith('.tfrecord') and f.startswith(lb.enums.PatchType.NOCYCLONE.value) and int(f.split('_')[1]) in years])
        return cyc_fnames, no_cyc_fnames
