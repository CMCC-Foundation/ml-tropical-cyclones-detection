# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy
from lightning.pytorch.strategies.fsdp import FSDPStrategy
from lightning.pytorch.strategies.ddp import DDPStrategy

from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from lightning import Trainer
import torch.nn as nn
import torch

import argparse
import logging
import shutil
import munch
import toml
import os

import warnings
warnings.simplefilter("ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Custom path imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import sys
sys.path.append('../resources/library')
from tropical_cyclone.models import *
from tropical_cyclone.scaling import StandardScaler
from tropical_cyclone.dataset import TCPatchDataset
from tropical_cyclone.callbacks import DiscordLog, BenchmarkCSV
from tropical_cyclone.sampler import DistributedWeightedSampler

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program Info
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

PROGRAM_NAME = r'''
  ____________   ____       __            __  _           
 /_  __/ ____/  / __ \___  / /____  _____/ /_(_)___  ____ 
  / / / /      / / / / _ \/ __/ _ \/ ___/ __/ / __ \/ __ \
 / / / /___   / /_/ /  __/ /_/  __/ /__/ /_/ / /_/ / / / /
/_/  \____/  /_____/\___/\__/\___/\___/\__/_/\____/_/ /_/ 
                                                          
'''
PROGRAM_DESCRIPTION = 'Training program for Tropical Cyclone Detection'
PROGRAM_ARGUMENTS = [
    [('-c', '--config'), {'type':str, 'help':'Configuration file for this program', 'required':True}],
    [('-n', '--num_nodes'), {'type':int, 'help':'Number of cluster GPU nodes', 'required':True}],
    [('-d', '--devices'), {'type':int, 'help':'Number of GPU devices per node', 'required':True}],
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Parse arguments
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# parse CLI arguments
parser = argparse.ArgumentParser(prog=PROGRAM_NAME, description=PROGRAM_DESCRIPTION)
for arg in PROGRAM_ARGUMENTS:
    names, kwargs = arg
    sname, lname = names
    parser.add_argument(sname, lname, **kwargs)
args = parser.parse_args()

# read configuration file for this execution
config = munch.munchify(toml.load(args.config))

# get number of nodes and devices for this run
num_nodes = args.num_nodes
devices = args.devices

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Parse Configuration file
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# run
seed = config.run.seed

# directories
run_dir = config.dir.run
experiment_dir = config.dir.experiment
webhook_url = config.dir.webhook if hasattr(config.dir, 'webhook') else None
checkpoint = config.dir.checkpoint if hasattr(config.dir, 'checkpoint') else None
train_src = config.dir.train
valid_src = config.dir.valid
mean_src = config.dir.scaler.mean
std_src = config.dir.scaler.std

# pytorch
dtype = eval(config.torch.dtype)
matmul_precision = config.torch.matmul_precision

# lightning
accelerator = config.lightning.accelerator
precision = config.lightning.precision
strategy_name = config.lightning.strategy if hasattr(config.lightning, 'strategy') else 'ddp'

# model
model_cls = eval(config.model.cls)
model_args = dict(config.model.args)

# metrics
metrics_list = [eval(mts) for mts in config.model.metrics]

# loss
loss_cls = eval(config.loss.cls)
loss_args = dict(config.loss.args)

# optimizer
optimizer_cls = eval(config.optimizer.cls)
optimizer_args = dict(config.optimizer.args)

# scheduler
scheduler_warmup_args = config.scheduler.warmup
scheduler_annealing_args = config.scheduler.annealing

# data
drivers = config.data.drivers
targets = config.data.targets
label_no_cyclone = config.data.label_no_cyclone if hasattr(config.data, 'label_no_cyclone') else -1.0
only_one_coo = config.data.only_one_coo if hasattr(config.data, 'only_one_coo') else None

# train
epochs = config.train.epochs
batch_size = config.train.batch_size
drop_remainder = config.train.drop_remainder
accumulation_steps = config.train.accumulation_steps
n_samples = config.train.n_samples if hasattr(config.train,'n_samples') else None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Environment setup
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

device = 'cpu'
# set the device
if torch.cuda.is_available():
    device = 'cuda'
torch.set_float32_matmul_precision(matmul_precision)

# define important directories
log_dir = os.path.join(run_dir, 'logs')
logging_dir = os.path.join(log_dir, 'logging')
checkpoint_dir = os.path.join(run_dir, 'checkpoints')

# define important filenames
benchmark_csv = os.path.join(run_dir, 'benchmark.csv')

# create experiment directory
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# save training hyperparameters
shutil.copy(src=args.config, dst=os.path.join(run_dir, 'configuration.toml'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program variables
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# load scaler
x_scaler = StandardScaler(mean_src=mean_src, std_src=std_src, drivers=drivers)

# define user callbacks
callbacks = [
    ModelCheckpoint(checkpoint_dir, "epoch-{epoch:04d}-val_loss-{val_loss:.2f}", monitor='val_loss', save_last=True, save_top_k=5, auto_insert_metric_name=False), 
    BenchmarkCSV(filename=benchmark_csv), 
    DiscordLog(model_name=os.path.basename(run_dir), webhook_url=webhook_url, benchmark_csv=benchmark_csv, msg_every_n_epochs=1, plot_every_n_epochs=10), 
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Setup Trainer
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init distribution strategy
if accelerator == 'cuda':
    if strategy_name == 'deepspeed':
        strategy = DeepSpeedStrategy(zero_optimization=True, stage=1, load_full_weights=True)
    elif strategy_name == 'ddp':
        strategy = DDPStrategy()
    elif strategy_name == 'fsdp':
        strategy = FSDPStrategy(sharding_strategy="NO_SHARD")
else:
    strategy = 'auto'

# initialize trainer
trainer = Trainer(
    accelerator=accelerator, 
    strategy=strategy, 
    devices=devices, 
    num_nodes=num_nodes, 
    precision=precision, 
    callbacks=callbacks, 
    max_epochs=epochs, 
    logger=None, 
    enable_checkpointing=True, 
    enable_progress_bar=True, 
    accumulate_grad_batches=accumulation_steps, 
    use_distributed_sampler=False, 
    default_root_dir=run_dir, 
    num_sanity_val_steps=0, 
    enable_model_summary=False, 
)

# store parallel execution variables
world_size = trainer.world_size
node_rank = trainer.node_rank
global_rank = trainer.global_rank
local_rank = trainer.local_rank

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Init Logger
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f'{logging_dir}/dev-{global_rank}.log', filemode='w', level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# log
logging.info(f"Logger initialized. Starting the execution")
logging.info(f"   World size  : {world_size}")
logging.info(f"   Node rank   : {node_rank}")
logging.info(f"   Global rank : {global_rank}")
logging.info(f"   Local rank  : {local_rank}")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Initialize ML Model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init model
model = model_cls(**model_args)
# init optimizer
optimizer = optimizer_cls(model.parameters(), **optimizer_args)
# init scheduler
verbose = True if global_rank == 0 else False
warmup = lr_scheduler.LinearLR(optimizer=optimizer, **scheduler_warmup_args, verbose=verbose)
annealing = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=epochs+1, **scheduler_annealing_args, verbose=verbose)
lr_scheduler = lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup, annealing], milestones=[scheduler_warmup_args['total_iters']], verbose=verbose)
# add attributes to the model
model.optimizer = optimizer
model.lr_scheduler = lr_scheduler
model.loss = loss_cls(**loss_args)
model.metrics = [mts() for mts in metrics_list]

# log summary
if not global_rank:
    print(ModelSummary(model=model, max_depth=1))
    logging.info(ModelSummary(model=model, max_depth=1))

# log
logging.info(f'Model and Trainer inizialized')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Load and distribute dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init train and validation datasets
train_dataset = TCPatchDataset(src=train_src, drivers=drivers, targets=targets, scaler=x_scaler, label_no_cyclone=label_no_cyclone, augmentation=True, only_one_coo=only_one_coo, dtype=dtype)
valid_dataset = TCPatchDataset(src=valid_src, drivers=drivers, targets=targets, scaler=x_scaler, label_no_cyclone=label_no_cyclone, augmentation=True, only_one_coo=only_one_coo, dtype=dtype)

# log
logging.info(f'Train and valid datasets inizialized')

# init train and val samplers
train_sampler = DistributedWeightedSampler(dataset=train_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
valid_sampler = DistributedWeightedSampler(dataset=valid_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)

# log
logging.info(f'Dataset samplers initialized')

# load dataloader
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, drop_last=drop_remainder)
valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=batch_size, drop_last=drop_remainder)

# log
logging.info(f'Dataloaders created')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Train and Validate the model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# log
logging.info(f'Training the model')

# fit the model
trainer.fit(model, train_loader, valid_loader, ckpt_path=checkpoint)

# log
logging.info(f'Model trained')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program End
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# log
logging.info(f'Program completed')

# close program
exit(1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
