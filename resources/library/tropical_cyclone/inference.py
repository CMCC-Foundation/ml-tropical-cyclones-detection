from tropical_cyclone.cyclone import retrieve_predicted_tc, init_track_dataframe, tracking_track_algorithm
from tropical_cyclone.macros import ERA5_CMIP6_DRIVERS_MAPPING
from tropical_cyclone.georeferencing import round_to_grid
from tropical_cyclone.scaling import StandardScaler
import tropical_cyclone as tc

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np
import logging
import joblib
import torch
import munch
import glob
import toml
import os



class Inference():
    def __init__(self, model_dir, dataset_dir, ibtracs_src, lat_range, lon_range, is_cmip6, device='cpu') -> None:
        # store arguments
        self.model_dir = model_dir
        self.dataset_dir = dataset_dir
        self.ibtracs_src = ibtracs_src
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.is_cmip6 = is_cmip6
        self.device = device
        # load config file
        self.config = munch.munchify(toml.load(os.path.join(self.model_dir, 'configuration.toml')))
        # parse the configuration
        self.scaler, self.drivers, self.targets = self._parse_config_file()

    def _parse_config_file(self):
        drivers = self.config.data.drivers
        targets = self.config.data.targets
        scaler = StandardScaler(mean_src=self.config.dir.scaler.mean, std_src=self.config.dir.scaler.std, drivers=drivers)
        return scaler, drivers, targets

    def _load_trained_model(self):
        # get model weights file
        model_weights_file = sorted(glob.glob(os.path.join(self.model_dir, 'checkpoints','*.ckpt')))[-1]
        # init model class from config
        model_cls = eval(self.config.model.cls)
        # get model arguments from config
        model_args = self.config.model.args
        # create the model
        model:nn.Module = model_cls(**model_args)
        model_state = torch.load(f=model_weights_file, map_location='cpu')
        # load weights into model
        model.load_state_dict(model_state['model'])
        # put the model to device
        model.to(self.device)
        return model, model_weights_file

    def _load_dataset(self, year=None):
        if self.is_cmip6:
            ds = xr.open_zarr(os.path.join(self.dataset_dir, 'dataset.zarr'))
        else:
            if year is not None: pattern = f'{year}*.nc'
            else: pattern = f'*.nc'
            files = sorted(glob.glob(os.path.join(self.dataset_dir, pattern)))
            if len(files) == 0: return None
            ds = xr.open_mfdataset(files)
        return ds

    def _load_ibtracs(self):
        if self.ibtracs_src is None:
            return None
        ibtracs = pd.read_csv(self.ibtracs_src)[['ISO_TIME','LAT','LON','SID']]
        # round lats and lons to the grid
        ibtracs['LAT'] = round_to_grid(ibtracs['LAT'])
        ibtracs['LON'] = round_to_grid(ibtracs['LON'])
        # shift coordinates from [-180,180] to [0,360]
        ibtracs['LON'] = (ibtracs['LON'] + 360) % 360
        # remove out of bound detections
        ibtracs = ibtracs[(ibtracs['LAT'] >= self.lat_range[0]) & (ibtracs['LAT'] <= self.lat_range[1]) & (ibtracs['LON'] >= self.lon_range[0]) & (ibtracs['LON'] <= self.lon_range[1])]
        return ibtracs

    def store_detections(self, detections, dst):
        if detections is None:
            logging.info(f'No detections found')
            return
        if os.path.exists(dst):
            logging.info(f'File already existing at {dst}')
            return
        # store detections to disk
        detections.to_csv(dst)

    def tracking(self, max_distance=400.0, min_track_count=12, min_wind_speed=None):
        # initialize tracking dataframe
        det_tracks = init_track_dataframe(self.detections)
        # apply tracking scheme to detections
        # det_tracks = tracking_track_algorithm(track_df=det_tracks, lats=self.latitudes, lons=self.longitudes, max_distance=max_distance, min_track_count=min_track_count, min_wind_speed=min_wind_speed)
        det_tracks = tracking_track_algorithm(track_df=det_tracks, max_distance=max_distance, min_track_count=min_track_count, min_wind_speed=min_wind_speed)
        return det_tracks

    def get_observed_tracks(self):
        # rename the sid with the track id
        obs_tracks = self.observations.rename(columns={'SID':'TRACK_ID'})
        # sort tracks
        obs_tracks = obs_tracks.sort_values('TRACK_ID').sort_values('ISO_TIME').reset_index(drop=True)
        return obs_tracks

    def predict(self, year=None):
        if self.is_cmip6:
            dvs = []
            for d in self.drivers:
                dvs.append(ERA5_CMIP6_DRIVERS_MAPPING[d])
            self.drivers = dvs
        # load model
        self.model, _ = self._load_trained_model()
        # load dataset
        self.ds = self._load_dataset(year)
        if self.ds is None:
            logging.info(f'No dataset found')
            self.detections = None
            return
        # load ibtracs
        ibtracs = self._load_ibtracs()
        patch_size = 40
        lons = self.ds['lon'].shape[0]
        lats = self.ds['lat'].shape[0]
        latitudes, longitudes = self.ds['lat'].data, self.ds['lon'].data
        rows = lats // patch_size
        cols = lons // patch_size
        time, channels = self.ds['time'].shape[0], len(self.drivers)
        time_instants = pd.to_datetime(self.ds['time'].astype(str))
        # divide dataset in patches
        patch_ds = self.ds.coarsen({'lat':patch_size, 'lon':patch_size}, boundary="trim").construct({'lon':("cols", "lon_range"), 'lat':("rows", "lat_range")})

        # load dataset to numpy
        x = patch_ds[self.drivers].to_array().load().data

        # transpose drivers to last channel
        x = np.transpose(x, axes=(1,2,3,4,5,0))

        # put rows and cols channels near time dimension
        x = np.transpose(x, axes=(0,1,3,2,4,5))

        # reshape aggregating time, rows and cols
        x = np.reshape(x, newshape=(time*rows*cols, patch_size, patch_size, channels))

        # transform the data with the scaler
        x = self.scaler.transform(torch.as_tensor(x))

        # move drivers to second dimension
        x = np.transpose(x, axes=(0,3,1,2))

        # convert to dataset
        dataset = TensorDataset(x)

        # create a dataloader
        data_loader = DataLoader(dataset=dataset, batch_size=4096)

        # predict with the trained model
        y_pred = np.empty(shape=(0,2))
        for data in tqdm(data_loader):
            x = data[0].to(self.device)
            y_pred = np.concatenate([y_pred, self.model(x).cpu().detach().numpy()])

        # reshape disgregating time, rows and cols
        y_pred = np.reshape(y_pred, newshape=(time, rows, cols, 2))

        # get predicted cyclone coordinates
        self.patch_ds = retrieve_predicted_tc(y_pred, self.ds, patch_ds, patch_size)

        # convert cyclone coordinates to pandas dataframe
        df = self.patch_ds['patch_cyclone_pred'].to_dataframe().reset_index()

        # merge dataframe to get coordinates
        detections = pd.merge(left=df[df['coordinate']==0], right=df[df['coordinate']==1], on=['time','rows','cols'])

        # take only time and coordinates
        detections = detections[['time', 'patch_cyclone_pred_x', 'patch_cyclone_pred_y']]

        # rename coordinates to LAT and LON
        detections = detections.rename(columns={'patch_cyclone_pred_x':'LAT', 'patch_cyclone_pred_y':'LON'})

        # remove NaN rows
        detections = detections[~(np.isnan(detections['LAT']) & np.isnan(detections['LON']))]

        # convert time axis to datetime
        detections['time'] = pd.to_datetime(detections['time'].astype(str))

        # get ibtracs detections for all era5 timesteps
        if ibtracs is not None:
            ibtracs['ISO_TIME'] = pd.to_datetime(ibtracs['ISO_TIME'])
            observations = ibtracs[(ibtracs['ISO_TIME'].isin(time_instants))]
            observations[observations['ISO_TIME'].isin(detections['time'])].reset_index(drop=True)
            observations = observations[['ISO_TIME','SID','LAT','LON']]
            observations['WS'] = np.inf
            observations = observations.reset_index(drop=True)

        # convert proj and era5 `time` col to `ISO_TIME`
        detections = detections.rename(columns={'time':'ISO_TIME'})

        # add infinite wind speed on each row
        detections['WS'] = np.inf

        # reset index
        detections = detections.reset_index(drop=True)

        self.detections = detections
        self.observations = observations if ibtracs is not None else None
        self.latitudes = latitudes
        self.longitudes = longitudes
