# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Program imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies.fsdp import FSDPStrategy
from sklearn.model_selection import ParameterGrid
import torch
from torch.distributed.fsdp import ShardingStrategy
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import argparse
import joblib
import logging
import munch
import os
import pickle
import re
import shutil
import toml

import warnings

warnings.simplefilter("ignore")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Custom path imports
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys

sys.path.append("../resources/library")
import tropical_cyclone as tc
from tropical_cyclone.trainer import FabricTrainer
from tropical_cyclone.tester import GraphTester
from tropical_cyclone.callbacks import DiscordLog
from tropical_cyclone.dataset import TCGraphDataset

# Provenance logger
sys.path.append("../../ProvML")
import prov4ml

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Program Info
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

PROGRAM_NAME = r"""
  ____________   ____       __            __  _           
 /_  __/ ____/  / __ \___  / /____  _____/ /_(_)___  ____ 
  / / / /      / / / / _ \/ __/ _ \/ ___/ __/ / __ \/ __ \
 / / / /___   / /_/ /  __/ /_/  __/ /__/ /_/ / /_/ / / / /
/_/  \____/  /_____/\___/\__/\___/\___/\__/_/\____/_/ /_/ 
                                                          
"""
PROGRAM_DESCRIPTION = "Training program for Tropical Cyclone Detection"
PROGRAM_ARGUMENTS = [
    [
        ("-c", "--config"),
        {"type": str, "help": "Configuration file for this program", "required": True},
    ],
    [
        ("-n", "--num_nodes"),
        {"type": int, "help": "Number of cluster GPU nodes", "required": True},
    ],
    [
        ("-d", "--devices"),
        {"type": int, "help": "Number of GPU devices per node", "required": True},
    ],
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
scaler_fpath = config.dir.scaler
webhook_url = config.dir.webhook if hasattr(config.dir, "webhook") else None
checkpoint = config.dir.checkpoint if hasattr(config.dir, "checkpoint") else None
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
optimizer_params = {}
for optim_cls in config.optimizer.cls:
    optimizer_params[optim_cls.split(".")[2]] = dict(
        config.optimizer[optim_cls.split(".")[2]]
    )

# scheduler
scheduler_cls = eval(config.scheduler.cls) if hasattr(config.scheduler, "cls") else None
scheduler_args = (
    dict(config.scheduler.args) if hasattr(config.scheduler, "cls") else None
)

# data
drivers = config.data.drivers
targets = config.data.targets

# train
epochs = config.train.epochs
batch_size = config.train.batch_size
augmentation = config.train.augmentation
drop_remainder = config.train.drop_remainder
accumulation_steps = config.train.accumulation_steps
n_samples = config.train.n_samples if hasattr(config.train, "n_samples") else None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Environment setup
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# set the device
device = torch.device("cuda")
torch.set_float32_matmul_precision(matmul_precision)

# define important directories
log_dir = os.path.join(run_dir, "logs_training")
logging_dir = os.path.join(log_dir, "logging")
checkpoint_dir = os.path.join(run_dir, "checkpoints")

# define important filenames
benchmark_csv = os.path.join(run_dir, "benchmark.csv")
last_model = os.path.join(run_dir, "last_model.pt")

# create experiment directory
os.makedirs(experiment_dir, exist_ok=True)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# logger and profiler
logger = CSVLogger(root_dir=log_dir, name="csv-logs")

# save training hyperparameters
shutil.copy(src=args.config, dst=os.path.join(run_dir, "configuration.toml"))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Program variables
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# load scaler
scaler = joblib.load(scaler_fpath)

# define user callbacks
callbacks = [
    DiscordLog(
        webhook_url=webhook_url,
        benchmark_csv=benchmark_csv,
        msg_every_n_epochs=10,
        plot_every_n_epochs=10,
    ),
]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Initialize dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# init train and validation datasets
train_dataset = TCGraphDataset(
    src=train_src,
    drivers=drivers,
    targets=targets,
    scaler=scaler,
    augmentation=augmentation,
    dtype=dtype,
)
valid_dataset = TCGraphDataset(
    src=valid_src,
    drivers=drivers,
    targets=targets,
    scaler=scaler,
    augmentation=augmentation,
    dtype=dtype,
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Hyperparameter Optimization Loop
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# initialize logger
logging_level = logging.INFO

# prepare the parameters to be considered in the grid search
search_params = model_args.copy()

search_params["batch_size"] = batch_size
search_params["epochs"] = epochs
search_params["optimizer_cls"] = config.optimizer.cls
search_params["optimizer_params"] = optimizer_params

# convert to list the arguments that are not lists yet
search_params.update(
    (idx, val) if isinstance(val, list) else (idx, [val])
    for idx, val in search_params.items()
)

# optimization results
results = {}

for params in ParameterGrid(search_params):
    # post-permutation operations
    # Extract the optimizer's args for the optimizer
    optimizer_args = params["optimizer_params"][params["optimizer_cls"].split(".")[2]]
    # Remove the list of all the arguments (We replace it by the exact set of args for the optimizer)
    params.pop("optimizer_params")
    params["optimizer_args"] = optimizer_args
    # Extract the name of the version of the optimizer configuration
    postfix = re.match(r"^[^_]+(_(\w+.*))?$", params["optimizer_cls"])

    # If there is a versioan name after "_" in the name of the optimizer class, add the name to the params dict.
    if postfix[2] != None:
        params["optimizer_cls_version"] = postfix[2]
    else:
        params["optimizer_cls_version"] = "__"

    # saving the torch.optim.* without versions
    optimizer_cls = params["optimizer_cls"].split("_")[0]

    # removing the "torch.optim." string from the optimizer name for more clearity in the resulting plots
    params["optimizer_cls"] = params["optimizer_cls"].split(".")[-1].split("_")[0]

    str_params = str(params)
    print("Hyperparamers in this section:\n", str_params, "\n")

    for key in model_args:
        if key in params:
            model_args[key] = params[key]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  Setup Trainer
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # init distribution strategy
    if accelerator == "cuda":
        strategy = FSDPStrategy(sharding_strategy=ShardingStrategy.NO_SHARD)
    else:
        strategy = "auto"

    # initialize trainer
    trainer = FabricTrainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=devices,
        num_nodes=num_nodes,
        precision=precision,
        loggers=None,
        plugins=[],
        callbacks=callbacks,
        max_epochs=params["epochs"],
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
    #  Provenance logger
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

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  Distribute dataset and DataLoader creation
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # init train and val samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=drop_remainder,
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=drop_remainder,
    )

    # load dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        drop_last=drop_remainder,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=512, shuffle=False, drop_last=drop_remainder
    )  # fixed batch size to keep the testing even

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  Initialize ML Model
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    model: nn.Module = model_cls(**model_args)
    model.loss = loss_cls(**loss_args)
    model.metrics = [mts() for mts in metrics_list]
    model = model.to(device)

    # setup the model and the optimizer
    trainer.setup(
        model=model,
        optimizer_cls=eval(optimizer_cls),
        optimizer_args=optimizer_args,
        scheduler_cls=scheduler_cls,
        scheduler_args=scheduler_args,
        checkpoint=checkpoint,
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #  Train and Validate the model
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # fit the model - using the valid_loader for testing
    trainer.fit(train_loader=train_loader, val_loader=None)

    # test the model on the validation set
    tester = GraphTester(
        device=device,
        loader=valid_loader,
        model=model,
        n_cyclones=valid_dataset.n_cy,
        nodes_per_graph=model_args["nodes_per_graph"],
    )
    metrics = tester.get_metrics()
    results[str_params] = metrics

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Program End
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# save the model to disk
trainer.fabric.save(
    path=last_model,
    state={
        "model": trainer.model,
        "optimizer": trainer.optimizer,
        "scheduler": trainer.scheduler_cfg,
    },
)

results_file = os.path.join(logging_dir, config.dir.result)
# using Pickle to save the result
with open(results_file, "wb") as file:
    pickle.dump(results, file)

# terminate prov4ml
prov4ml.end_run(create_graph=False, create_svg=False)

# close program
exit(1)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
