# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from lightning import Trainer
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
from torch.distributed.fsdp import ShardingStrategy
from torch.optim import lr_scheduler
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import argparse
import logging
import joblib
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
import tropical_cyclone as tc
from tropical_cyclone.callbacks import DiscordLog
from tropical_cyclone.dataset import TCGraphDataset

# Provenance logger
try:
    import sys
    sys.path.append('../../yProvML')
    import prov4ml
except ImportError:
    print('Library prov4ml not found, halting execution...')
    exit(0)

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

# run - TODO not in use anymore
seed = config.run.seed

# directories
run_dir = config.dir.run
experiment_dir = config.dir.experiment
scaler_fpath = config.dir.scaler
webhook_url = config.dir.webhook if hasattr(config.dir, 'webhook') else None
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
scheduler_warmup_args = config.scheduler.warmup
scheduler_annealing_args = config.scheduler.annealing

# data
drivers = config.data.drivers
targets = config.data.targets

# train
epochs = config.train.epochs
batch_size = config.train.batch_size
augmentation = config.train.augmentation
drop_remainder = config.train.drop_remainder
accumulation_steps = config.train.accumulation_steps
n_samples = config.train.n_samples if hasattr(config.train,'n_samples') else None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Environment setup
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# set the device
device = torch.device('cuda')
torch.set_float32_matmul_precision(matmul_precision)

# define important directories
log_dir = os.path.join(run_dir, 'logs_training')
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

# save training hyperparameters
shutil.copy(src=args.config, dst=os.path.join(run_dir, 'configuration.toml'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program variables
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# load scaler
scaler = joblib.load(scaler_fpath)

# define user callbacks
callbacks = [
    ModelCheckpoint(checkpoint_dir, "epoch-{epoch:04d}-val_loss-{val_loss:.2f}", monitor='val_loss', save_last=True, save_top_k=5, auto_insert_metric_name=False),
    DiscordLog(webhook_url=webhook_url, benchmark_csv=benchmark_csv, msg_every_n_epochs=10, plot_every_n_epochs=10), 
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Setup Trainer
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init distribution strategy
if accelerator == 'cuda':
    strategy = DDPStrategy()    #FSDPStrategy(sharding_strategy=ShardingStrategy.NO_SHARD)
else:
    strategy = 'auto'

# initialize trainer
trainer = Trainer(
    accelerator=accelerator, 
    strategy=strategy, 
    devices=devices, 
    num_nodes=num_nodes, 
    precision=precision, 
    logger=None,
    callbacks=callbacks, 
    max_epochs=epochs, 
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

# Initialize general logger
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
#  Initialize provenance logger
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

prov_path = os.path.join(run_dir, "prov_path")
os.makedirs(prov_path, exist_ok=True)

prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="default", 
    provenance_save_dir=prov_path, 
    collect_all_processes=False,
    save_after_n_logs=100,
)

logging.info(f"Prov4ML logger started running")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Load and distribute dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init train and validation datasets
train_dataset = TCGraphDataset(src=train_src, drivers=drivers, targets=targets, scaler=scaler, augmentation=augmentation, dtype=dtype)
valid_dataset = TCGraphDataset(src=valid_src, drivers=drivers, targets=targets, scaler=scaler, augmentation=augmentation, dtype=dtype)

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
#  Initialize ML Model
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# init model
prov4ml.log_param("model arguments", model_args)
model:nn.Module = model_cls(**model_args)

# init scheduler
optimizer = optimizer_cls(model.parameters(), **optimizer_args)
warmup = lr_scheduler.LinearLR(optimizer=optimizer, **scheduler_warmup_args, verbose=True)
annealing = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=epochs+1, **scheduler_annealing_args, verbose=True)
lr_scheduler = lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=[warmup, annealing], milestones=[scheduler_warmup_args['total_iters']], verbose=True)

# init model attributes
model.loss = loss_cls(**loss_args)
model.lr_scheduler = lr_scheduler
model.metrics = [mts() for mts in metrics_list]
model.optimizer = optimizer

model = model.to(device)

print(model)


# log
logging.info(f'Model and Fabric Trainer inizialized')

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

# log model in provenance graph
model_name = str(config.model.cls.split('.')[-1])
prov4ml.log_model(model, model_name)

# terminate prov4ml
prov4ml.end_run(create_graph=True, create_svg=False)

# close program
exit(1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
