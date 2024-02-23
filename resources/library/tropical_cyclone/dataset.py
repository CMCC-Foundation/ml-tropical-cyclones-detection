from torch.utils.data import Dataset
from typing import Any, List
import xarray as xr
import numpy as np
import torch
import glob
import os

from tropical_cyclone.utils import coo_rot180, coo_left_right, coo_up_down



def read_xarray_dataset(filenames):
    return [xr.load_dataset(filename) for filename in filenames]


def read_data_as_torch_tensor(dss: List[str], variables: List[str], dtype = torch.float32):
    return torch.stack([torch.as_tensor(ds[variables].to_array().load().data, dtype=dtype) for ds in dss], dim=0)


def read_zarrs_as_torch_tensor(zarrs: List[xr.Dataset], variables: List[str], dtype = torch.float32):
    data = []
    for zarr in zarrs:
        x = torch.as_tensor(zarr[variables].to_array().load().data, dtype=dtype)
        if len(x.shape) == 4:
            x = torch.permute(x, dims=(1,0,2,3))
        elif len(x.shape) == 3:
            x = torch.permute(x, dims=(1,0,2))
        data.append(x)
    return torch.concat(data, dim=0)


class InterTwinTrainvalCycloneDataset(Dataset):
    def __init__(self, src: str, drivers: List[str], targets: List[str], scaler = None, augmentation: bool = False, dtype = torch.float32) -> None:
        super().__init__()
        # store params
        self.augmentation = augmentation
        self.scaler = scaler
        self.dtype = dtype
        # get dataset filenames
        cy_files = sorted(glob.glob(os.path.join(src,'cyclone*.zarr')))
        nr_files = sorted(glob.glob(os.path.join(src,'nearest*.zarr')))
        rn_files = sorted(glob.glob(os.path.join(src,'random*.zarr')))
        # open zarr datasets
        cy_zarrs = [xr.open_zarr(file) for file in cy_files]
        nr_zarrs = [xr.open_zarr(file) for file in nr_files]
        rn_zarrs = [xr.open_zarr(file) for file in rn_files]
        # get the total number of elements from each dataset
        cy_n = sum([ds.pid.shape[0] for ds in cy_zarrs])
        nr_n = sum([ds.pid.shape[0] for ds in nr_zarrs])
        rn_n = sum([ds.pid.shape[0] for ds in rn_zarrs])
        # get the total number of elements of the entire dataset
        if self.augmentation: mul = 4
        else: mul = 1
        self.n = cy_n * mul + nr_n + rn_n
        # save cy_n for augmentation purposes
        self.cy_n = cy_n
        # get dataset from the zarr files
        self.__prepare_dataset(cy_zarrs, nr_zarrs, rn_zarrs, drivers, targets)
        # prepare for the scaling
        if self.scaler:
            _, self.C, self.H, self.W = self.x_data.shape

    def __len__(self):
        return self.n

    def __getitem__(
            self, 
            index: int) -> Any:
        # get the data from dataset
        x, y = self.x_data[index], self.y_data[index]
        # scale the features
        x = self.__scale(x)
        # augment data
        x, y = self.__augment(x, y, index)
        # cast the tensor to desired dtype
        x, y = x.type(torch.float32), y.type(self.dtype)
        return x, y

    def __prepare_dataset(self, 
            cy_zarrs: List[xr.Dataset], 
            nr_zarrs: List[xr.Dataset], 
            rn_zarrs: List[xr.Dataset], 
            drivers: List[str], 
            targets: List[str]):
        # cyclone data
        x_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, drivers, self.dtype)
        y_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, targets, self.dtype)
        # eventually augment data
        if self.augmentation:
            # rot 180 data
            x_rot180_cy_data = torch.clone(x_cy_data)
            y_rot180_cy_data = torch.clone(y_cy_data)
            # flip up down data
            x_flipud_cy_data = torch.clone(x_cy_data)
            y_flipud_cy_data = torch.clone(y_cy_data)
            # flip left right data
            x_fliplr_cy_data = torch.clone(x_cy_data)
            y_fliplr_cy_data = torch.clone(y_cy_data)
            # concatenate the datasets
            x_cy_data = torch.concat([x_cy_data, x_rot180_cy_data, x_flipud_cy_data, x_fliplr_cy_data], dim=0)
            y_cy_data = torch.concat([y_cy_data, y_rot180_cy_data, y_flipud_cy_data, y_fliplr_cy_data], dim=0)
        # nearest data
        x_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, drivers, self.dtype)
        y_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, targets, self.dtype)
        # random data
        x_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, drivers, self.dtype)
        y_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, targets, self.dtype)
        # store the data
        self.x_data = torch.concat([x_cy_data, x_nr_data, x_rn_data])
        self.y_data = torch.concat([y_cy_data, y_nr_data, y_rn_data])

    def __augment(self, x, y, index):
        if self.augmentation:
            if index in range(self.cy_n * 1, self.cy_n * 2):
                # flip up down
                x, y = coo_up_down(data=(x,y))
                return x, y
            if index in range(self.cy_n * 2, self.cy_n * 3):
                # flip left right
                x, y = coo_left_right(data=(x,y))
                return x, y
            if index in range(self.cy_n * 3, self.cy_n * 4):
                # rot 180
                x, y = coo_rot180(data=(x,y))
                return x, y
        return x, y

    def __scale(self, x):
        if self.scaler:
            # permute x dimensions to H x W x C
            x = torch.permute(x, dims=(1,2,0))
            # collapse H x W channels to N x C
            x = torch.reshape(x, shape=(self.H * self.W, self.C))
            # scale the data
            x = torch.as_tensor(self.scaler.transform(x))
            # reverse from N x C to H x W x C
            x = torch.reshape(x, shape=(self.H, self.W, self.C))
            # permute x dimensions to C x H x W
            x = torch.permute(x, dims=(2,0,1))
        return x

