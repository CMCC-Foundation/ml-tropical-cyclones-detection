# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from lightning.fabric.plugins.environments import MPIEnvironment
from torch.utils.data.distributed import DistributedSampler
from lightning.fabric.strategies.fsdp import FSDPStrategy
from torch.distributed.fsdp import ShardingStrategy
from lightning.fabric.loggers import CSVLogger
from torch.utils.data import DataLoader
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
sys.path.append('../resources/library/')
import tropical_cyclone as tc
from tropical_cyclone.callbacks import FabricBenchmark, FabricCheckpoint
from tropical_cyclone.dataset import InterTwinTrainvalCycloneDataset
from tropical_cyclone.scaling import StandardScaler
from tropical_cyclone import FabricTrainer

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
experiment_dir = config.dir.experiment
run_dir = os.path.join(experiment_dir, config.dir.run)
mean_src = config.dir.scaler.mean
std_src = config.dir.scaler.std
checkpoint = config.dir.checkpoint if hasattr(config.dir, 'checkpoint') else None
train_src = config.dir.train
valid_src = config.dir.valid

# pytorch
dtype = eval(config.torch.dtype)
matmul_precision = config.torch.matmul_precision

# lightning
accelerator = config.lightning.accelerator
precision = config.lightning.precision

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
scheduler_cls = eval(config.scheduler.cls) if hasattr(config.scheduler, 'cls') else None
scheduler_args = dict(config.scheduler.args) if hasattr(config.scheduler, 'cls') else None

# data
drivers = config.data.drivers
targets = config.data.targets

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
last_model = os.path.join(run_dir, 'last_model.pt')

# create experiment directory
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# logger and profiler
logger = CSVLogger(root_dir=log_dir, name='csv-logs')

# save training hyperparameters
shutil.copy(src=args.config, dst=os.path.join(run_dir, 'configuration.toml'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program variables
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# load scaler
scaler = StandardScaler(mean_src=mean_src, std_src=std_src, drivers=drivers)

# define user callbacks
callbacks = [
    FabricCheckpoint(dst=checkpoint_dir, monitor='val_loss', verbose=False), 
    FabricBenchmark(filename=benchmark_csv), 
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Setup Trainer
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init distribution strategy
if accelerator == 'cuda':
    strategy = FSDPStrategy(sharding_strategy=ShardingStrategy.NO_SHARD)
else:
    strategy = 'auto'

# initialize trainer
trainer = FabricTrainer(
    accelerator=accelerator, 
    strategy=strategy, 
    devices=devices, 
    num_nodes=num_nodes, 
    precision=precision, 
    loggers=logger, 
    plugins=[MPIEnvironment()], 
    callbacks=callbacks, 
    max_epochs=epochs, 
    grad_accum_steps=accumulation_steps, 
    use_distributed_sampler=False, 
    seed=seed, 
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
model:nn.Module = model_cls(**model_args)
model.loss = loss_cls(**loss_args)
model.metrics = [mts() for mts in metrics_list]

# setup the model and the optimizer
trainer.setup(model=model, optimizer_cls=optimizer_cls, optimizer_args=optimizer_args, scheduler_cls=scheduler_cls, scheduler_args=scheduler_args, checkpoint=checkpoint)

# log
logging.info(f'Model and Fabric Trainer inizialized')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Load and distribute dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init train and validation datasets
train_dataset = InterTwinTrainvalCycloneDataset(src=train_src, drivers=drivers, targets=targets, scaler=scaler, augmentation=True, dtype=dtype)
valid_dataset = InterTwinTrainvalCycloneDataset(src=valid_src, drivers=drivers, targets=targets, scaler=scaler, augmentation=True, dtype=dtype)

# log
logging.info(f'Train and valid datasets inizialized')

# init train and val samplers
train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=drop_remainder)
valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=global_rank, shuffle=False, drop_last=drop_remainder)

# log
logging.info(f'Dataset samplers initialized')

# load dataloader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_remainder)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_remainder)

# log
logging.info(f'Dataloaders created')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Train and Validate the model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# log
logging.info(f'Training the model')

# fit the model
trainer.fit(train_loader=train_loader, val_loader=valid_loader)

# log
logging.info(f'Model trained')

# save the model to disk
trainer.fabric.save(path=last_model, state={'model':trainer.model, 'optimizer':trainer.optimizer, 'scheduler': trainer.scheduler_cfg})

# log
logging.info(f'Last model saved to {last_model}')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program End
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# log
logging.info(f'Program completed')

# close program
exit(1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
