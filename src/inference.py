from mpi4py import MPI
import pandas as pd
import logging
import os
import sys

import warnings
warnings.filterwarnings('ignore')

sys.path.append('../resources/library')
from info import test_years
from tropical_cyclone.inference import SingleModelInference


# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program Parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# main directories
experiment_dir = '/work/cmcc/machine_learning/dd26322/tropical_cyclone_tracking/experiments'
data_dir = '/work/cmcc/machine_learning/dd26322/tropical_cyclone_tracking/data'

experiment = sys.argv[1]
eps = float(sys.argv[2])

# select model
model_dir = os.path.join(experiment_dir, f'{experiment}')
dataset_dir = os.path.join(data_dir, 'datasets/north_pacific')

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
    detections = inference.predict(ds, patch_size=40, eps=eps, roll=False)
    # rdetections = inference.predict(ds, patch_size=40, eps=eps, roll=True) # rolled detections
    # detections = pd.concat((detections, rdetections))
    logging.info(f'   ...done')
    # store detections on disk
    inference.store_detections(detections, detection_dst)
    logging.info(f'   predictions stored')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()