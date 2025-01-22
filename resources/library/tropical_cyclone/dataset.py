from torch_geometric.data import Data, InMemoryDataset as Dataset_PyG
from torch.utils.data import Dataset as Dataset_torch
import torch

from typing import Any, List, Tuple, Union
import xarray as xr
import logging
import shutil
import glob
import os

from tropical_cyclone.utils import coo_rot180, coo_left_right, coo_up_down
from tropical_cyclone.scaling import Scaler



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



class TCPatchDataset(Dataset_torch):
    def __init__(self, src: str, drivers: List[str], targets: List[str], scaler: Scaler = None, label_no_cyclone: float = -1.0, augmentation: bool = False, dtype = torch.float32) -> None:
        super().__init__()
        # store params
        self.label_no_cyclone = label_no_cyclone
        self.augmentation = augmentation
        self.scaler: Scaler = scaler
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
        self._prepare_dataset(cy_zarrs, nr_zarrs, rn_zarrs, drivers, targets)
        # prepare for the scaling
        if self.scaler:
            _, self.C, self.H, self.W = self.x_data[0].shape

    def __len__(self):
        return self.n

    def __getitem__(self, index: int) -> Any:
        bucket, idx = index
        # get data from the correct bucket
        x, y = self.x_data[bucket][idx], self.y_data[bucket][idx]
        # scale the features
        x = self._scale(x)
        # augment data
        x, y = self._augment(x, y, bucket)
        # apply no cyclone label
        y = self._apply_no_cyclone_label(y)
        # cast the tensor to desired dtype
        x, y = x.type(torch.float32), y.type(self.dtype)
        return x, y[0]

    def _apply_no_cyclone_label(self, y: torch.Tensor):
        return torch.where(y < 0, self.label_no_cyclone, y)

    def _prepare_dataset(self, 
            cy_zarrs: List[xr.Dataset], 
            nr_zarrs: List[xr.Dataset], 
            rn_zarrs: List[xr.Dataset], 
            drivers: List[str], 
            targets: List[str]):
        # log
        logging.info(f'reading cyclone data')
        # cyclone data
        x_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, drivers, self.dtype)
        y_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, targets, self.dtype)
        # eventually augment the dataset
        if self.augmentation:
            # log
            logging.info(f'  adding augmentations')
            # rot 180 data
            x_rot180_cy_data = torch.clone(x_cy_data)
            y_rot180_cy_data = torch.clone(y_cy_data)
            # flip up down data
            x_flipud_cy_data = torch.clone(x_cy_data)
            y_flipud_cy_data = torch.clone(y_cy_data)
            # flip left right data
            x_fliplr_cy_data = torch.clone(x_cy_data)
            y_fliplr_cy_data = torch.clone(y_cy_data)
        # log
        logging.info(f'reading nearest data')
        # nearest data
        x_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, drivers, self.dtype)
        y_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, targets, self.dtype)
        # log
        logging.info(f'reading random data')
        # random data
        x_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, drivers, self.dtype)
        y_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, targets, self.dtype)
        # if augmentation is added, add TC copies to the data
        if self.augmentation:
            self.x_data = [x_cy_data, x_rot180_cy_data, x_flipud_cy_data, x_fliplr_cy_data, x_nr_data, x_rn_data]
            self.y_data = [y_cy_data, y_rot180_cy_data, y_flipud_cy_data, y_fliplr_cy_data, y_nr_data, y_rn_data]
        else:
            self.x_data = [x_cy_data, x_nr_data, x_rn_data]
            self.y_data = [y_cy_data, y_nr_data, y_rn_data]
        self.total_num_samples = [x.shape[0] for x in self.x_data]

    def _augment(self, x, y, index):
        if self.augmentation:
            if index == 1:
                x, y = coo_rot180(data=(x,y))
                return x, y
            elif index == 2:
                x, y = coo_up_down(data=(x,y))
                return x, y
            elif index == 3:
                x, y = coo_left_right(data=(x,y))
                return x, y
        return x, y

    def _scale(self, x: torch.Tensor):
        if self.scaler:
            x = torch.permute(x, dims=(1,2,0)) # C x H x W -> H x W x C
            x = self.scaler.transform(x)
            x = torch.permute(x, dims=(2,0,1)) # H x W x C -> C x H x W
        return x


class TCGraphDataset(Dataset_PyG):
    def __init__(self,
                 src: str,
                 drivers: List[str],
                 targets: List[str],
                 scaler = None,
                 augmentation: bool = False,
                 dtype = torch.float32):
        self.src = src
        self.drivers = drivers
        self.targets = targets
        self.scaler = scaler
        self.augmentation = augmentation
        self.dtype = dtype
        
        self.split = self.src.split("/")[-1]
        self.data_list = None
        self.n_cy = -1
        
        # Trigger self.process()
        super(TCGraphDataset, self).__init__(src)
        
        # Remove dummy folders
        try:
            shutil.rmtree(self.processed_dir)
        except OSError as e:
            print(f"Error in cleaning dummy folder: {e.strerror}")
        
        # Loads the stored dataset in self.data, or self._data(with this method of loading you should make the get retrieve the value from this)
        #self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return os.path.join(self.src)
    
    @property
    def processed_dir(self) -> str:
        return "dummy"
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return os.listdir(self.raw_dir)
    
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        return ['dummy']
    
    # Process zarr data into graphs and save it into the processed_dir folder
    def process(self) -> None:
        x_data, y_data = self.__prepare_zarr()
        #print("\t", x_data.shape)
        #print("\t", y_data.shape)
        
        # Adjacency structure is the same for all 40x40 grids
        edge_index = self.__get_adjacency_info(x_data[0, 0])
        
        data_list = []
        
        for i in range(x_data.shape[0]):
            # x[i] is reshaped from [6, 40, 40], so [C, H, W], to [W, H, C], to [W*H, C], in the same order as self__get_adjacency_info() does
            nodes_feats = x_data[i].permute(2, 1, 0).contiguous().view(-1, 6)
            nodes_labels = y_data[i].permute(2, 1, 0).contiguous().view(-1, 1)
            
            # Create and append the Data object
            data = Data(
                x=nodes_feats,
                edge_index=edge_index,
                y=nodes_labels
            )
            
            # If scaler exists, transform the data now for faster training later
            if self.scaler != None:
                data.x = torch.tensor(self.scaler.transform(data.x), dtype=self.dtype)
            data_list.append(data)
        
        self.data_list = data_list
        print(f"\t{self.split} dataset created with {self.len()} elements!")
        print(f"\tshape of elements:")
        print(f"\t\tx: {self.data_list[0].x.shape}")
        print(f"\t\tedge_index: {self.data_list[0].edge_index.shape}")
        print(f"\t\ty: {self.data_list[0].y.shape}")
        
    def __get_adjacency_info(self, x_data) -> torch.Tensor:
        width = x_data.shape[1]
        height = x_data.shape[0]
        coo_links = [[], []]
        this_node = 0
        
        # The order of nodes is the same in how .permute(2, 1, 0).contiguous().view(-1, 6) is done in self.process()
        for w in range(width):
            for h in range(height):
                # Cell above exists, add the link
                if (h-1)>=0:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node-1)
                # Cell right exists, add the link
                if (w+1)<width:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node+height)
                # Cell below exists, add the link
                if (h+1)<height:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node+1)
                # Cell left exists, add the link
                if (w-1)>=0:
                    coo_links[0].append(this_node)
                    coo_links[1].append(this_node-height)
                
                this_node += 1
        
        return torch.tensor(coo_links, dtype=torch.long)
    
    def __prepare_zarr(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get dataset filenames
        cy_files = sorted(glob.glob(os.path.join(self.raw_dir, 'cyclone*.zarr')))
        if (self.split == 'test'):
            no_cy_files = sorted(glob.glob(os.path.join(self.raw_dir, 'no_cyclone*.zarr')))
        else:
            nr_files = sorted(glob.glob(os.path.join(self.raw_dir, 'nearest*.zarr')))
            rn_files = sorted(glob.glob(os.path.join(self.raw_dir, 'random*.zarr')))
        
        # Open zarr datasets
        cy_zarrs = [xr.open_zarr(file) for file in cy_files]
        if (self.split == 'test'):
            no_cy_zarrs = [xr.open_zarr(file) for file in no_cy_files]
        else:
            nr_zarrs = [xr.open_zarr(file) for file in nr_files]
            rn_zarrs = [xr.open_zarr(file) for file in rn_files]
        
        # Get cyclone data
        x_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, self.drivers, self.dtype)
        y_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, self.targets, self.dtype)
        
        # Eventually augment cyclone data
        if self.augmentation:
            # Rot 180, flip up-down and flip left-right the x features
            x_rot180_cy_data = torch.rot90(x_cy_data, k=2, dims=(2, 3))
            x_flipud_cy_data = torch.flip(x_cy_data, dims=(2,))
            x_fliplr_cy_data = torch.flip(x_cy_data, dims=(3,))
            # Concatenate the datasets
            x_cy_data = torch.concat([x_cy_data, x_rot180_cy_data, x_flipud_cy_data, x_fliplr_cy_data], dim=0)

            # Same for the y
            if 'density_map_tc' in self.targets:
                y_cy_data = self.__augment_y_density_map_tc(y_cy_data)
            elif 'patch_cyclone' in self.targets:
                patch_size = x_cy_data.shape[2]
                y_cy_data = self.__augment_y_patch_cyclone(y_cy_data, patch_size)
            
            print(f"\taugmentation of {self.split} dataset: done!")
        
        print(f"\tN of cyclone patches: {x_cy_data.shape[0]}")
        
        if (self.split == 'test'):
            # Get no cyclone data
            x_no_cy_data = read_zarrs_as_torch_tensor(no_cy_zarrs, self.drivers, self.dtype)
            y_no_cy_data = read_zarrs_as_torch_tensor(no_cy_zarrs, self.targets, self.dtype)
            print(f"\t{self.split} set - N of no cyclone patches: {x_no_cy_data.shape[0]}")
        else:
            # Get nearest data
            x_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, self.drivers, self.dtype)
            y_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, self.targets, self.dtype)
            # Get random data
            x_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, self.drivers, self.dtype)
            y_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, self.targets, self.dtype)
            print(f"\t{self.split} set - N of nearest patches: {x_nr_data.shape[0]}")
            print(f"\t{self.split} set - N of random patches: {x_rn_data.shape[0]}")
        
        # save number of cyclone patches for testing purposes
        self.n_cy = x_cy_data.shape[0]
        
        # Group the data
        if (self.split == 'test'):
            x_data = torch.concat([x_cy_data, x_no_cy_data])
            y_data = torch.concat([y_cy_data, y_no_cy_data])
        else:
            x_data = torch.concat([x_cy_data, x_nr_data, x_rn_data])
            y_data = torch.concat([y_cy_data, y_nr_data, y_rn_data])
        
        return x_data, y_data
   
    def __augment_y_density_map_tc(self, y_data) -> torch.Tensor:
        y_rot180 = torch.rot90(y_data, k=2, dims=(2, 3))
        y_flipud = torch.flip(y_data, dims=(2,))
        y_fliplr = torch.flip(y_data, dims=(3,))
        return torch.concat([y_data, y_rot180, y_flipud, y_fliplr], dim=0)
    
    def __augment_y_patch_cyclone(self, y_data, patch_size) -> torch.Tensor:
        y_rot180 = torch.clone(y_data)
        y_flipud = torch.clone(y_data)
        y_fliplr = torch.clone(y_data)

        # y rot 180
        for i in range(y_rot180.shape[0]):
            if y_rot180[i, 0, 0] != -1:
                y_rot180[i] = torch.as_tensor([-y_rot180[i, 0, 0] + patch_size -1, 
                                               -y_rot180[i, 0, 1] + patch_size -1])
        # y flip up-down
        for i in range(y_flipud.shape[0]):
            if y_flipud[i, 0, 0] != -1:
                y_flipud[i] = torch.as_tensor([-y_flipud[i, 0, 0] + patch_size -1,
                                               y_flipud[i, 0, 1]])
        # y flip left-right
        for i in range(y_fliplr.shape[0]):
            if y_fliplr[i, 0, 0] != -1:
                y_fliplr[i] = torch.as_tensor([y_fliplr[i, 0, 0],
                                               -y_fliplr[i, 0, 1] + patch_size -1])
        
        return torch.concat([y_data, y_rot180, y_flipud, y_fliplr], dim=0)
    
    def get(self, idx: int) -> Data:
        return self.data_list[idx]
    
    def len(self) -> int:
        return len(self.data_list)
    
## TODO: NON CANCELLARE
# class InterTwinTrainvalCycloneDataset(Dataset):
#     def __init__(self, src: str, drivers: List[str], targets: List[str], scaler = None, augmentation: bool = False) -> None:
#         super().__init__()
#         # store augmentation
#         self.augmentation = augmentation
#         # store scaler, if provided
#         self.scaler = scaler
#         # get dataset filenames
#         cy_files = sorted(glob.glob(os.path.join(src,'cyclone_2004*.nc')))
#         nr_files = sorted(glob.glob(os.path.join(src,'nearest_2004*.nc')))
#         rn_files = sorted(glob.glob(os.path.join(src,'random_2004*.nc')))
#         # get the total number of elements from each dataset
#         cy_n, nr_n, rn_n = len(cy_files), len(nr_files), len(rn_files)
#         # get the total number of elements of the entire dataset
#         if self.augmentation: mul = 4
#         else: mul = 1
#         self.n = cy_n * mul + nr_n + rn_n
#         # save cy_n for augmentation purposes
#         self.cy_n = cy_n
#         # load dataset
#         cy_dss = read_xarray_dataset(cy_files)
#         nr_dss = read_xarray_dataset(nr_files)
#         rn_dss = read_xarray_dataset(rn_files)
#         # get dataset from the zarr files
#         self.__prepare_dataset(cy_dss, nr_dss, rn_dss, drivers, targets)
#         # prepare for the scaling
#         if self.scaler:
#             _, self.C, self.H, self.W = self.x_data.shape

#     def __len__(self):
#         return self.n

#     def __getitem__(
#             self, 
#             index: int) -> Any:
#         # get the data from dataset
#         x, y = self.x_data[index], self.y_data[index]
#         # scale the features
#         x = self.__scale(x)
#         # augment data
#         x, y = self.__augment(x, y, index)
#         return x, y

#     def __prepare_dataset(self, 
#             cy_dss: List[xr.Dataset], 
#             nr_dss: List[xr.Dataset], 
#             rn_dss: List[xr.Dataset], 
#             drivers: List[str], 
#             targets: List[str]):
#         # cyclone data
#         x_cy_data = read_data_as_torch_tensor(cy_dss, drivers)
#         y_cy_data = read_data_as_torch_tensor(cy_dss, targets)
#         # eventually augment data
#         if self.augmentation:
#             # rot 180 data
#             x_rot180_cy_data = torch.clone(x_cy_data)
#             y_rot180_cy_data = torch.clone(y_cy_data)
#             # flip up down data
#             x_flipud_cy_data = torch.clone(x_cy_data)
#             y_flipud_cy_data = torch.clone(y_cy_data)
#             # flip left right data
#             x_fliplr_cy_data = torch.clone(x_cy_data)
#             y_fliplr_cy_data = torch.clone(y_cy_data)
#             # concatenate the datasets
#             x_cy_data = torch.concat([x_cy_data, x_rot180_cy_data, x_flipud_cy_data, x_fliplr_cy_data], dim=0)
#             y_cy_data = torch.concat([y_cy_data, y_rot180_cy_data, y_flipud_cy_data, y_fliplr_cy_data], dim=0)
#         # nearest data
#         x_nr_data = read_data_as_torch_tensor(nr_dss, drivers)
#         y_nr_data = read_data_as_torch_tensor(nr_dss, targets)
#         # random data
#         x_rn_data = read_data_as_torch_tensor(rn_dss, drivers)
#         y_rn_data = read_data_as_torch_tensor(rn_dss, targets)
#         # store the data
#         self.x_data = torch.concat([x_cy_data, x_nr_data, x_rn_data])
#         self.y_data = torch.concat([y_cy_data, y_nr_data, y_rn_data])

#     def __augment(self, x, y, index):
#         if self.augmentation:
#             if index in range(self.cy_n * 1, self.cy_n * 2):
#                 # flip up down
#                 x, y = coo_up_down(data=(x,y))
#                 return x, y
#             if index in range(self.cy_n * 2, self.cy_n * 3):
#                 # flip left right
#                 x, y = coo_left_right(data=(x,y))
#                 return x, y
#             if index in range(self.cy_n * 3, self.cy_n * 4):
#                 # rot 180
#                 x, y = coo_rot180(data=(x,y))
#                 return x, y
#         return x, y

#     def __scale(self, x):
#         if self.scaler:
#             # permute x dimensions to H x W x C
#             x = torch.permute(x, dims=(1,2,0))
#             # collapse H x W channels to N x C
#             x = torch.reshape(x, shape=(self.H * self.W, self.C))
#             # scale the data
#             x = torch.as_tensor(self.scaler.transform(x))
#             # reverse from N x C to H x W x C
#             x = torch.reshape(x, shape=(self.H, self.W, self.C))
#             # permute x dimensions to C x H x W
#             x = torch.permute(x, dims=(2,0,1))
#         return x
