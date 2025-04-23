from tropical_cyclone.cyclone import (
    retrieve_predicted_tc,
    init_track_dataframe,
    tracking_algorithm,
)
# from tropical_cyclone.macros import ERA5_CMIP6_DRIVERS_MAPPING
from tropical_cyclone.georeferencing import round_to_grid
from tropical_cyclone.scaling import StandardScaler
from tropical_cyclone.models import *
import tropical_cyclone as tc

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np
import logging
import torch
import munch
import glob
import toml
import os


def load_ibtracs(ibtracs_src, lat_range=None, lon_range=None):
    if ibtracs_src is None:
        return None
    # load ibtracs from src
    ibtracs = pd.read_csv(ibtracs_src)[["ISO_TIME", "LAT", "LON", "SID"]]
    # round lats and lons to the grid
    ibtracs["LAT"] = round_to_grid(ibtracs["LAT"])
    ibtracs["LON"] = round_to_grid(ibtracs["LON"])
    # shift coordinates from [-180,180] to [0,360]
    ibtracs["LON"] = (ibtracs["LON"] + 360) % 360
    # remove out of bound detections
    ibtracs = ibtracs[
        (ibtracs["LAT"] >= lat_range[0])
        & (ibtracs["LAT"] <= lat_range[1])
        & (ibtracs["LON"] >= lon_range[0])
        & (ibtracs["LON"] <= lon_range[1])
    ]
    # convert ibtracs dates to pandas dates
    ibtracs["ISO_TIME"] = pd.to_datetime(ibtracs["ISO_TIME"])
    return ibtracs


def get_observations(ibtracs_src, dates, lat_range, lon_range):
    ibtracs = load_ibtracs(ibtracs_src, lat_range, lon_range)
    # get only observations within the dates
    observations = ibtracs[(ibtracs["ISO_TIME"].isin(dates))]
    observations = observations[["ISO_TIME", "SID", "LAT", "LON"]]
    observations["WS"] = np.inf
    observations = observations.reset_index(drop=True)
    return observations


def get_observed_tracks(observations):
    # rename the sid with the track id
    observed_tracks = observations.rename(columns={"SID": "TRACK_ID"})
    # sort tracks
    observed_tracks = (
        observed_tracks.sort_values("TRACK_ID")
        .sort_values("ISO_TIME")
        .reset_index(drop=True)
    )
    return observed_tracks


def iqr_localization(y_pred, n_models):
    """
    This method is based on the Inter Quartile Range technique.

    """
    # compute third and first quartiles
    percentile_75 = np.nanpercentile(y_pred, 75, axis=1)
    percentile_25 = np.nanpercentile(y_pred, 25, axis=1)
    # compute iqr
    iqr = percentile_75 - percentile_25
    # get upper and lower bounds
    upper_bound = percentile_75 + 1.5 * iqr
    lower_bound = percentile_25 - 1.5 * iqr
    # replicate lower and upper bound to first axis
    upper_bound = np.repeat(upper_bound[:, np.newaxis, :], repeats=n_models, axis=1)
    lower_bound = np.repeat(lower_bound[:, np.newaxis, :], repeats=n_models, axis=1)
    # filter out all predictions that are not contained between lower bound and upper bound
    y_pred_filtered = np.where(
        (y_pred > lower_bound) & (y_pred < upper_bound), y_pred, np.nan
    )
    # compute mean over the entire dataset
    mu = np.round(
        np.nanmean(
            np.where(
                np.repeat(
                    (~np.isnan(y_pred_filtered))
                    .astype(np.int8)
                    .min(axis=2)[:, :, np.newaxis],
                    repeats=2,
                    axis=2,
                )
                > 0,
                y_pred_filtered,
                np.nan,
            ),
            axis=1,
        ),
        0,
    )
    # compute mean over the entire dataset
    std = np.nanstd(
        np.where(
            np.repeat(
                (~np.isnan(y_pred_filtered))
                .astype(np.int8)
                .min(axis=2)[:, :, np.newaxis],
                repeats=2,
                axis=2,
            )
            > 0,
            y_pred_filtered,
            np.nan,
        ),
        axis=1,
    )
    return mu, std


def load_trained_model(model_dir, device="cpu"):
    # read config file
    config = munch.munchify(toml.load(os.path.join(model_dir, "configuration.toml")))
    # get model weights file
    model_weights_file = sorted(
        glob.glob(os.path.join(model_dir, "checkpoints", "*.ckpt"))
    )[-1]
    # init model class from config
    model_cls = eval(config.model.cls)
    # get model arguments from config
    model_args = config.model.args
    # create the model
    model: nn.Module = model_cls(**model_args)
    # load state dict
    model_state_dict = torch.load(f=model_weights_file, map_location=device)
    # load weights into model
    model.load_state_dict(model_state_dict["state_dict"])
    # put the model to device
    model.to(device)
    return model, config, model_weights_file


def prepare_dataloader(
    patch_ds, patch_size, scaler, drivers, time, rows, cols, channels, batch_size=4096
):
    # load dataset to numpy
    x = patch_ds[drivers].to_array().load().data
    # transpose drivers to last channel
    x = np.transpose(x, axes=(1, 2, 3, 4, 5, 0))
    # put rows and cols channels near time dimension
    x = np.transpose(x, axes=(0, 1, 3, 2, 4, 5))
    # reshape aggregating time, rows and cols
    x = np.reshape(x, newshape=(time * rows * cols, patch_size, patch_size, channels))
    # transform the data with the scaler
    x = scaler.transform(torch.as_tensor(x))
    # remove all nan values (if present)
    x = np.nan_to_num(x, nan=0.0)
    # move drivers to second dimension
    x = np.transpose(x, axes=(0, 3, 1, 2))
    # convert to dataset
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32))
    # create a dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    return data_loader


def predict_with_models(models, data_loader, device="cpu"):
    if type(models) != list:
        models = [models]
    models_predictions = []
    for model in models:
        # predict with the trained model
        y_pred = np.empty(shape=(0, 2))
        for data in tqdm(data_loader):
            x = data[0].to(device)
            y_pred = np.concatenate([y_pred, model(x).cpu().detach().numpy()])
        # reshape disgregating time, rows and cols
        models_predictions.append(y_pred)
    # stack together each model's predictions
    models_predictions = np.stack(models_predictions, axis=1)
    if models_predictions.shape[1] == 1:
        models_predictions = models_predictions[:, 0]
    return models_predictions


def get_detections(patch_ds):
    # convert cyclone coordinates to pandas dataframe
    df = patch_ds["patch_cyclone_pred"].to_dataframe().reset_index()
    # merge dataframe to get coordinates
    detections = pd.merge(
        left=df[df["coordinate"] == 0],
        right=df[df["coordinate"] == 1],
        on=["time", "rows", "cols"],
    )
    # take only time and coordinates
    detections = detections[["time", "patch_cyclone_pred_x", "patch_cyclone_pred_y"]]
    # rename coordinates to LAT and LON
    detections = detections.rename(
        columns={"patch_cyclone_pred_x": "LAT", "patch_cyclone_pred_y": "LON"}
    )
    # remove NaN rows
    detections = detections[
        ~(np.isnan(detections["LAT"]) & np.isnan(detections["LON"]))
    ]
    # convert time axis to datetime
    detections["time"] = pd.to_datetime(detections["time"].astype(str))
    # convert proj and era5 `time` col to `ISO_TIME`
    detections = detections.rename(columns={"time": "ISO_TIME"})
    # add infinite wind speed on each row
    detections["WS"] = np.inf
    # reset index
    detections = detections.reset_index(drop=True)
    return detections


class Inference:
    def __init__(self, device="cpu") -> None:
        self.device = device

    def predict(self):
        raise NotImplementedError

    def store_detections(self, detections, dst):
        if detections is None:
            logging.info(f"No detections found")
            return
        if os.path.exists(dst):
            logging.info(f"File already existing at {dst}")
            return
        # store detections to disk
        detections.to_csv(dst)
        logging.info(f"Detections stored at {dst}")

    def _parse_config_file(self, config):
        drivers = config.data.drivers
        targets = config.data.targets
        scaler = StandardScaler(
            mean_src=config.dir.scaler.mean, 
            std_src=config.dir.scaler.std, 
            drivers=drivers)
        return scaler, drivers, targets

    def load_dataset(self, dataset_dir, year=None):
        if year is not None:
            pattern = f"{year}*.nc"
        else:
            pattern = f"*.nc"
        files = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
        logging.info(f"Opening dataset containing {len(files)} files")
        if len(files) == 0:
            return None
        ds = xr.open_mfdataset(files)
        logging.info(f"Dataset opened")
        if ds is None:
            logging.info(f"No dataset found")
            return None
        dates = pd.to_datetime(ds["time"].astype(str))
        return ds, dates

    def tracking(
        self, detections, max_distance=400.0, min_track_count=12
    ):
        # initialize tracking dataframe
        detected_tracks = init_track_dataframe(detections)
        # apply tracking scheme to detections
        detected_tracks = tracking_algorithm(
            detected_tracks=detected_tracks,
            max_distance=max_distance,
            min_track_count=min_track_count,
        )
        return detected_tracks


class SingleModelInference(Inference):
    def __init__(self, model_dir, device="cpu") -> None:
        super().__init__(device)
        self.model_dir = model_dir
        self.model, self.config, _ = load_trained_model(model_dir, device)
        self.scaler, self.drivers, self.targets = self._parse_config_file(self.config)

    def predict(
        self,
        ds: xr.Dataset,  # xarray dataset containing input data
        patch_size: int = 40,  # dimension of a patch
        eps: float = 0.1,  # tolerance value for negative label
    ):
        lons = ds["lon"].shape[0]
        lats = ds["lat"].shape[0]
        rows = lats // patch_size
        cols = lons // patch_size
        time, channels = ds["time"].shape[0], len(self.drivers)
        # divide dataset in patches
        patch_ds = ds.coarsen(
            {"lat": patch_size, "lon": patch_size}, boundary="trim"
        ).construct({"lon": ("cols", "lon_range"), "lat": ("rows", "lat_range")})
        # get dataloader
        data_loader = prepare_dataloader(
            patch_ds=patch_ds,
            patch_size=patch_size,
            scaler=self.scaler,
            drivers=self.drivers,
            time=time,
            rows=rows,
            cols=cols,
            channels=channels,
            batch_size=4096,
        )
        # predict
        y_pred = predict_with_models(
            models=[self.model], data_loader=data_loader, device=self.device
        )
        y_pred = np.reshape(y_pred, newshape=(time, rows, cols, 2))
        # get predicted cyclone coordinates
        patch_ds = retrieve_predicted_tc(y_pred, ds, patch_ds, patch_size, eps=eps)
        # get detections
        detections = get_detections(patch_ds)
        return detections


class EnsembleModelInference(Inference):
    """Class for the ensemble of models.
    It expects that all the models share the same drivers and same scaler
    """

    def __init__(self, models_dir, device="cpu") -> None:
        super().__init__(device)
        self.models_dir = models_dir
        self._load_models(models_dir, device)

    def _load_models(self, src, device):
        self.models, self.configs = [], []
        logging.info(f"Loading models")
        model_dirs = sorted(glob.glob(os.path.join(src, "*")))
        logging.info(f"   found {len(model_dirs)} models")
        # self.model, self.config, _ = load_trained_model(model_dir, device)
        # self.scaler, self.drivers, self.targets = self._parse_config_file(self.config)
        for i, model_dir in enumerate(model_dirs):
            model, config, _ = load_trained_model(model_dir, device)
            scaler, drivers, _ = self._parse_config_file(config)
            self.models.append(model)
            self.configs.append(config)
            if i == 0:
                self.drivers = drivers
                self.scaler = scaler
            logging.info(f"   model {os.path.basename(model_dir)} loaded")
        logging.info(f"All models successfully loaded")

    def _retrieve_ensemble_predictions(
        self,
        models_predictions,
        epsilon,
        patch_size,
        n_consensus,
        label_no_cyclone,
        time,
        rows,
        cols,
    ):
        # set to nan all negative coordinates
        y_pred = np.where(models_predictions < -epsilon, np.nan, models_predictions)
        # round to patch_size-1 all predictions greater than patch_size-1
        y_pred = np.where(y_pred > patch_size - 1, patch_size - 1, y_pred)
        # create a vector containing the number of models that predicted a TC for each sample
        n_tc_pred = np.repeat(
            (y_pred > 0).astype(np.int8).min(axis=2).sum(axis=1)[:, np.newaxis],
            repeats=2,
            axis=1,
        )
        # localize points
        mu, std = iqr_localization(y_pred, len(self.models))
        # select only predictions where most models agree and round predictions to the nearest integer
        y_pred = np.where(n_tc_pred >= n_consensus, np.round(mu, 2), label_no_cyclone)
        # select only standard deviations where most models agree
        std = np.where(n_tc_pred >= n_consensus, std, np.nan)
        # replace remaining nan values
        y_pred = np.where(np.isnan(y_pred), label_no_cyclone, y_pred)
        # reshape to adapt to the patch ds
        y_pred = np.reshape(y_pred, newshape=(time, rows, cols, 2))
        mu = np.reshape(mu, newshape=(time, rows, cols, 2))
        std = np.reshape(std, newshape=(time, rows, cols, 2))
        return y_pred, mu, std

    def predict(self, ds, n_consensus, label_no_cyclone=-1.0, patch_size=40, epsilon=0):
        lons = ds["lon"].shape[0]
        lats = ds["lat"].shape[0]
        rows = lats // patch_size
        cols = lons // patch_size
        time, channels = ds["time"].shape[0], len(self.drivers)
        # divide dataset in patches
        patch_ds = ds.coarsen(
            {"lat": patch_size, "lon": patch_size}, boundary="trim"
        ).construct({"lon": ("cols", "lon_range"), "lat": ("rows", "lat_range")})
        # get dataloader
        data_loader = prepare_dataloader(
            patch_ds=patch_ds,
            patch_size=patch_size,
            scaler=self.scaler,
            drivers=self.drivers,
            time=time,
            rows=rows,
            cols=cols,
            channels=channels,
            batch_size=4096,
        )
        # predict with the models
        models_predictions = predict_with_models(
            models=self.models, data_loader=data_loader, device=self.device
        )
        # get the patch ds filled with predicted tc coordinates
        y_pred, _, _ = self._retrieve_ensemble_predictions(
            models_predictions,
            epsilon,
            patch_size,
            n_consensus,
            label_no_cyclone,
            time,
            rows,
            cols,
        )
        # get predicted cyclone coordinates
        patch_ds = retrieve_predicted_tc(y_pred, ds, patch_ds, patch_size)
        # get detections
        detections = get_detections(patch_ds)
        return detections
