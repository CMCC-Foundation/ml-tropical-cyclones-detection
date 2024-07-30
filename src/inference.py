from mpi4py import MPI
import logging
import os
import sys

import warnings
warnings.filterwarnings('ignore')

sys.path.append('../resources/library')
from tropical_cyclone.macros import TEST_YEARS as test_years
from tropical_cyclone.inference import SingleModelInference


# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program Parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# select model
run_dir = f'../backup/{sys.argv[1]}'
dataset_dir = '../data/dataset'

# define inference folder
inference_dir = '../data/inference'
inference_model_dir = os.path.join(inference_dir, os.path.basename(run_dir))

# define logs directory
log_dir = f'../logs/inference_{sys.argv[1]}'

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

inference = SingleModelInference(model_dir=run_dir, device='mps')
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
    detections = inference.predict(ds, patch_size=40)
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