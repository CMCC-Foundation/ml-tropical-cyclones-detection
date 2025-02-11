import logging
from mpi4py import MPI
import munch
import os
import pandas as pd
import sys
import toml
import torch
from torch_geometric.loader import DataLoader as GraphDataLoader

import warnings
warnings.filterwarnings('ignore')

sys.path.append('../resources/library')
from info import test_years
import tropical_cyclone as tc
from tropical_cyclone.inference import SingleModelInference
from tropical_cyclone.models import GraphUNet
from tropical_cyclone.scaling import StandardScaler
from tropical_cyclone.tester import GraphTester


# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program Parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# main directories
experiment_dir = "../experiments"
data_dir = "../data"

experiment = sys.argv[1]
eps = float(sys.argv[2])

# select model
model_dir = os.path.join(experiment_dir, f'{experiment}')
dataset_dir = os.path.join(data_dir, 'dataset')

# get config file
config_file = os.path.join(model_dir, 'configuration.toml')

# parse config parameters
config = munch.munchify(toml.load(config_file))

# define inference folder
inference_dir = os.path.join(data_dir, 'inference')
inference_model_dir = os.path.join(inference_dir, os.path.basename(model_dir))

# define logs directory
log_dir = f'logs_{experiment}'

# create directory if not exist
if not rank:
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(inference_model_dir, exist_ok=True)
# wait for all the processes to arrive here
comm.Barrier()

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f"{log_dir}/proc-{rank}.log", 
                    filemode="w", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Inference on the Dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

logging.info(f'Starting inference')
logging.info(f'   selected model at {model_dir}')

# select use case (CNN/GNN)
if config.run.use_case == 'cnn':
    inference = SingleModelInference(model_dir=model_dir, device='cuda')
    drivers = inference.drivers

    logging.info(f'Evaluating test set years')

    for i in range(rank, len(test_years), size):
        year = test_years[i]
        logging.info(f'Year {year}')
        # get detection destination
        detection_dst = os.path.join(inference_model_dir, f'{year}.csv')
        logging.info(f'  Predicting...')
        # predict with the model
        ds, dates = inference.load_dataset(dataset_dir=dataset_dir, drivers=drivers, year=year, is_cmip6=False)
        detections = inference.predict(ds, patch_size=40, eps=eps)
        logging.info(f'   ...done')
        # store detections on disk
        inference.store_detections(detections, detection_dst)
        logging.info(f'   predictions stored')

elif config.run.use_case == 'gnn':
    # load scaler
    scaler = StandardScaler(src=config.dir.scaler, drivers=config.data.drivers)
    # define device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # get model details
    model_cls = eval(config.model.cls)
    model_args = dict(config.model.args)
    # get the model
    model:torch.nn.Module = model_cls(**model_args)
    model = model.to(device)
     
    for i in range(rank, len(test_years), size):
        year = test_years[i]
        logging.info(f'Year {year}')
        # get detection destination
        detection_dst = os.path.join(inference_model_dir, f'{year}.csv')
        # creating graph dataset and dataloader for the current year
        logging.info(f'  Graph dataset preparation...')
        dataset = tc.dataset.TCGraphDatasetInference(src=dataset_dir, year=year, drivers=config.data.drivers, targets=config.data.targets, scaler=scaler)
        loader = GraphDataLoader(dataset, batch_size=config.train.batch_size, shuffle=False, drop_last=False)
        # getting a graph tester to obtain the y predictions with shape [B, 2]
        logging.info(f'  Predicting...')
        tester = GraphTester(device=device, loader=loader, model=model, nodes_per_graph=model_args['nodes_per_graph'])
        tot_pred = tester.get_inference_y(threshold=0.4)
        # post-process operations
        logging.info(f'  ...post-processing...')
        dataset.post_process(tot_pred)
        logging.info(f'   ...done')
        # save .csv with coordinates and times to disk
        dataset.store_detections(dst=detection_dst)
        logging.info(f'   predictions stored')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()