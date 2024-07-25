# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
#                         Author : Massimiliano Fronza
#                  Create dataset scalers and store to disk
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from enum import Enum
from tqdm import tqdm
import logging
import joblib
import torch
import os

import sys
sys.path.append('../resources/library')
import tropical_cyclone as tc


class UseCase(Enum):
    CONV = 0
    GRAPH = 1

class ScalerType(Enum):
    STANDARD = 'standard'
    MINMAX = 'minmax'



# PROGRAM PARAMETERS
use_case = UseCase.GRAPH
scaler_type = ScalerType.STANDARD
batch_size = 64
drivers = ['fg10', 'i10fg', 'msl', 't_500', 't_300', 'vo_850']
targets = ['density_map_tc']
scaler_filename = f'x_{scaler_type.value}_scaler_{len(drivers)}_{"-".join(drivers)}_drivers.dump'

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f'Program started')

# define directories
data_dir = '../data'
scaler_dir = os.path.join(data_dir, 'scalers')
patches_dir = os.path.join(data_dir, 'patches', 'train')

# define scaler filename
scaler_fpath = os.path.join(scaler_dir, scaler_filename)

# create scaler dir if not exists
os.makedirs(scaler_dir, exist_ok=True)

logging.info(f'Reading the dataset')

if use_case == UseCase.GRAPH:
    from torch_geometric.loader import DataLoader
    
    # Train Dataset and Dataloader(much faster for .partial_fit() due to batching)
    dataset_train = tc.dataset.TCGraphDataset(src=patches_dir, drivers=drivers, targets=targets, augmentation=True)
    loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=False)
    
elif use_case == UseCase.CONV:
    from torch.utils.data import DataLoader
    
    # create the dataset
    dataset = tc.dataset.InterTwinTrainvalCycloneDataset(src=patches_dir, drivers=drivers, targets=targets)
    # using a DataLoader to make .partial_fit() is much faster than with Dataset
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

logging.info(f'Dataset read')

# select the scaler
if scaler_type == ScalerType.STANDARD:
    scaler = StandardScaler()
elif scaler_type == ScalerType.MINMAX:
    scaler = MinMaxScaler()

logging.info(f'Computing the dataset scaler')

# apply the partial fit to the scaler
for _, batch in zip(tqdm(loader), loader):
    if use_case == UseCase.CONV:
        x,_ = batch
        # if the patch is shaped as B x C x H x W (convolution use case)
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            # permute x dimensions to B x H x W x C
            x = torch.permute(x, dims=(0,2,3,1))
            # collapse B x H x W channels to N x C
            x = torch.reshape(x, shape=(B * H * W, C))
            scaler.partial_fit(x)
    elif use_case == UseCase.GRAPH:
        # if the patch is shaped as B x C x N (graph use case) TODO remove this
        scaler.partial_fit(batch.x)

# save the scaler to disk
joblib.dump(value=scaler, filename=scaler_fpath)

logging.info(f'Program completed')
