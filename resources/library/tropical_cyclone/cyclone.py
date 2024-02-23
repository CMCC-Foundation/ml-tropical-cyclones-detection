from tropical_cyclone.macros import DENSITY_MAP_TC, SQUARE_MAP_TC, LABEL_MAP_TC
from tropical_cyclone.georeferencing import from_local_to_global

from scipy.ndimage import gaussian_filter
import pandas as pd
import xarray as xr
import numpy as np
import os



def get_tc_map_labels(georef:pd.DataFrame, max_ts_tcs:int=20) -> pd.DataFrame:
    """
    Create a label pd.DataFrame in which, for each unique `ISO_TIME` in `georef`,
    it is associated the same label to each TC `SID` contained in georef.

    """
    # create a label pd.DataFrame
    label = pd.DataFrame(data={'ISO_TIME':georef['ISO_TIME'].unique()})
    for l in range(max_ts_tcs):
        # create a label to the label dataframe
        label[f'{l+1}'] = [np.nan] * len(label)
    # for each of the TCs in the dataset
    for sid in sorted(georef['SID'].unique()):
        # get the iso times in which it has developed
        iso_times = georef[georef['SID']==sid]['ISO_TIME'].sort_values().to_numpy()
        # get the first iso time in the label pd.DataFrame
        labels = label[label['ISO_TIME']==iso_times[0]].iloc[0,1:]
        # get the first nan index within the labels
        first_nan_idx = pd.isna(labels).argmax(axis=0)
        # add the TC SID to the corresponding label `first_nan_idx`
        label.loc[label['ISO_TIME'].isin(iso_times), f'{first_nan_idx+1}'] = sid
    return label



def get_tropical_cyclone_positions(ds:xr.Dataset, georef:pd.DataFrame, sigma:int=10) -> xr.Dataset:
    """
    Add Tropical Cyclone maps into the passed xr.Dataset. The maps built from `georef` and `label`
    pd.DataFrames are 3:
    1. `dm_tc` : map that, for every TC position has a gaussian filter of dimension 
        `sigma` pixels around its center.
    2. `sq_tc` - map that, for every TC position has a square of `radius` pixels around
        its center.
    3. `lb_tc` - map that, for every TC position has a square of `radius` pixels around
        its center, whose value is the same across each cyclone SID. This means that the 
        label correspondence between each TC occurrence is preserved among different timesteps.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset in which will be added the maps
    georef : pd.DataFrame
        Georeferencing dataset that contains the positions of each TC

    Returns
    -------
    ds : xr.Dataset
        The updated dataset

    Examples
    --------
    >>> ds = xr.open_dataset('path/to/netcdf/dataset')
    >>> georef = pd.DataFrame(data={'ISO_TIME':[], 'SID':[], 'LAT':[], 'LON':[], 'RLAT':[], 'RLON':[], 'YLAT':[], 'XLON':[]})
    >>> label = tc.cyclone.get_tc_map_labels(georef)
    >>> ds = tc.cyclone.get_tropical_cyclone_positions(ds, georef, label)

    """
    # make a copy of georeferencing pd.DataFrame
    gr = georef.copy()
    # convert gr column into datetime
    gr['ISO_TIME'] = pd.to_datetime(gr['ISO_TIME'])
    # get the shape of the 2D cyclone maps to be added to the xr.Dataset
    shape = (ds['time'].shape[0], ds['lat'].shape[0], ds['lon'].shape[0])
    # create a gaussian map that will contain the density map of each TC
    gaussian_map = np.zeros(shape=shape)
    # select only iso times that are compatible with our dataset
    gr = gr[gr['ISO_TIME'].isin(pd.to_datetime(ds['time']))]
    # loop over each time-step
    for t,iso_time in enumerate(gr['ISO_TIME'].unique()):
        rows = gr[gr['ISO_TIME']==iso_time]
        for _,row in rows.iterrows():
            j,i = row['XLON'], row['YLAT']
            # place 1 on the TC center
            gaussian_map[t,i,j] = 1
        # apply gaussian filtering over timestep t
        gaussian_map[t,:,:] = gaussian_filter(gaussian_map[t,:,:], sigma=sigma)
        # rescale the density map in [0,1]
        gaussian_map[t,:,:] = (gaussian_map[t,:,:] - gaussian_map[t,:,:].min()) / (gaussian_map[t,:,:].max() - gaussian_map[t,:,:].min())
    # create a density map in the xr.Dataset
    ds[DENSITY_MAP_TC] = (('time','lat','lon'), gaussian_map)
    # return the updated dataset
    return ds



def retrieve_predicted_tc(y_pred, ds, patch_ds, patch_size):
    """
    Retrieves the latitude-longitude coordinates from the passed predicted TCs
    
    """
    # create a latlons matrix with the same shape of y_pred_reshaped filled with nan
    cyclone_latlon_coords = np.full_like(y_pred, fill_value=np.nan)
    cyclone_rowcol_coords = np.full_like(y_pred, fill_value=np.nan)
    # for each timestep
    for t in range(y_pred.shape[0]):
        # for each row
        for i in range(y_pred.shape[1]):
            # for each column
            for j in range(y_pred.shape[2]):
                # if the model prediction is valid
                if y_pred[t,i,j,0] >= 0.0 and y_pred[t,i,j,1] >= 0.0:
                    try:
                        # retrieve global row-col coordinates of the TC
                        global_rowcol = from_local_to_global((i,j), y_pred[t,i,j,:], patch_size)
                        # retrieve global lat-lon coordinates of the TC
                        cyclone_latlon_coords[t,i,j,:] = (ds['lat'].data[global_rowcol[0]], ds['lon'].data[global_rowcol[1]])
                        cyclone_rowcol_coords[t,i,j,:] = (global_rowcol[0], global_rowcol[1])
                    except:
                        continue
    # update patch_ds cyclone_information
    patch_ds['patch_cyclone_pred'] = (('time','rows','cols','coordinate'), cyclone_latlon_coords)
    patch_ds['patch_rowcol_pred'] = (('time','rows','cols','rowcol'), cyclone_rowcol_coords)
    return patch_ds
