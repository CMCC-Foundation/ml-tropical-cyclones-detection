# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#                         Author : Massimiliano Fronza
#                  Create dataset scalers and store to disk
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from enum import Enum
import joblib
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append("../resources/library")
import tropical_cyclone as tc


class ScalerType(Enum):
    STANDARD = "standard"
    MINMAX = "minmax"


# PROGRAM PARAMETERS
scaler_type = ScalerType.STANDARD
batch_size = 512
drivers = ["fg10", "i10fg", "msl", "t_500", "t_300", "vo_850"]
targets = ["density_map_tc"]
scaler_filename = f"{scaler_type.value}.dump"

# define directories
data_dir = "../data"
scaler_dir = os.path.join(data_dir, "scalers")
patches_dir = os.path.join(data_dir, "patches", "train")
scaler_fpath = os.path.join(scaler_dir, scaler_filename)

# create scaler dir if it doesn't exist
os.makedirs(scaler_dir, exist_ok=True)

# train Dataset and Dataloader(much faster for .partial_fit() due to batching)
dataset_train = tc.dataset.TCGraphDataset(
    src=patches_dir, drivers=drivers, targets=targets, augmentation=True
)
loader = DataLoader(
    dataset_train, batch_size=batch_size, shuffle=False, drop_last=False
)

# select the scaler
if scaler_type == ScalerType.STANDARD:
    scaler = StandardScaler()
elif scaler_type == ScalerType.MINMAX:
    scaler = MinMaxScaler()

# apply the partial fit to the scaler
for _, batch in zip(tqdm(loader), loader):
    scaler.partial_fit(batch.x)

# save the scaler to disk
joblib.dump(value=scaler, filename=scaler_fpath)
