# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
#                         Author : Davide Donno
#               Process the dataset into patches and store 
#                them to disk in the preferred data format
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from mpi4py import MPI
import pandas as pd
import numpy as np
import logging
import os
import sys
sys.path.append('../../resources/library')
import tropical_cyclone as tc


# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# dataset years
train_years = sorted(tc.macros.TRAIN_YEARS)
valid_years = sorted(tc.macros.VALID_YEARS)
# trainval_years = sorted(train_years + valid_years)
test_years = sorted(tc.macros.TEST_YEARS)

# directories setup
main_dir = '../../'
data_dir = os.path.join(main_dir, 'data')
src_data_dir = os.path.join(data_dir, 'dataset')
dst_train_data_dir = os.path.join(data_dir, 'patches', 'train')
dst_valid_data_dir = os.path.join(data_dir, 'patches', 'valid')
dst_test_data_dir = os.path.join(data_dir, 'patches', 'test')
georef_src = os.path.join(data_dir, 'ibtracs', 'georef.csv')
patch_vars = ['fg10', 'i10fg', 'msl', 'sst', 't_500', 't_300', 'vo_850', 'density_map_tc']
coo_vars = ['real_cyclone', 'rounded_cyclone', 'global_cyclone', 'patch_cyclone']
log_dir = 'logs'

# hyperparameters
dtype = np.float32
sigma = 5
grid_res = 0.25
patch_size = 40
label_no_cyclone = -1.0

# create directory if not exists
if not rank:
    os.makedirs(dst_train_data_dir, exist_ok=True)
    os.makedirs(dst_valid_data_dir, exist_ok=True)
    os.makedirs(dst_test_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

# wait for all the processes to arrive here
comm.Barrier()

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f"{log_dir}/proc-{rank}.log", 
                    filemode="w", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# load dataframes
georef = pd.read_csv(georef_src, index_col=0)

# init the dataset writer
dataset_writer = tc.io.InterTwinMPIDatasetWriter(
    src_dir=src_data_dir, 
    georef=georef, 
    patch_vars=patch_vars, 
    coo_vars=coo_vars, 
    patch_size=patch_size, 
    label_no_cyclone=label_no_cyclone, 
    sigma=sigma, grid_res=grid_res, dtype=dtype)

# process the test dataset
dataset_writer.process(dst_dir=dst_test_data_dir, years=test_years, is_test=True)
logging.info(f'Test dataset completed')

# process the train dataset
dataset_writer.process(dst_dir=dst_train_data_dir, years=train_years, is_test=False)
logging.info(f'Train dataset completed')

# process the valid dataset
dataset_writer.process(dst_dir=dst_valid_data_dir, years=valid_years, is_test=False)
logging.info(f'Valid dataset completed')

# wait for all the processes to finish
comm.Barrier()
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()