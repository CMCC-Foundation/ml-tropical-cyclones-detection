#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Utilitary functions to use inside the module

import numpy as np

### ==================================================== ###
###          VARIABLES GEOGRAPHIC MANIPULATIONS          ###
### ==================================================== ###


def apply_mask_axis(var, mask, axis=-1):
    """Apply a mask on a field on the given axis.

    Parameters
    ----------
    var : np.ndarray
        The field on which to apply the mask
    mask : 1D np.ndarray
        The boolean mask (dimension must fit the dimension of var for the axis)
    axis : int, optional
        axis on which to perform the operation, by default -1

    Returns
    -------
    np.ndarray
        The field but apply for the desired region.
    """
    # Transposition of var so that the lat dimension is the first one
    L = len(np.shape(var))
    if axis < 0:
        axis = L + axis
    axes = np.arange(L)
    axes[0] = axis
    axes[1 : axis + 1] -= 1
    var_t = np.transpose(var, axes)

    # Apply the mask
    var_t_cut = var_t[mask]

    # Reverse the transposition
    axes = np.arange(L)
    axes[0:axis] = axes[1 : axis + 1]
    axes[axis] = 0
    var_cut = np.transpose(var_t_cut, axes)

    return var_cut


# ------------- North/South division -------------#


def get_south(var, lat, axis=-2, flip=False):
    """Returns the southern part of a given field.

    Parameters
    ----------
    var : np.ndarray
        The field whose southern part we want
    lat : np.ndarray
        latitude coordinate
    axis : int, optional
        latitude axis in var, by default -2
    flip : bool, option
        if True, reverse the _data along the latitude index (so that abs(lat) is increasing)

    Returns
    -------
    list of np.ndarray
        Southern part of the field and the corrsponding latitude coordinate array
    """

    mask = lat <= 0.0
    lat_south = lat[mask]

    var_south = apply_mask_axis(var, mask, axis)

    if flip:
        var_south = np.flip(var_south, axis)
        lat_south = np.flip(lat_south)

    return lat_south, var_south


def get_north(var, lat, axis=-2, flip=False):
    """Returns the northern part of a given field.

    Parameters
    ----------
    var : np.ndarray
        The field whose northern part we want
    lat : np.ndarray
        latitude coordinate
    axis : int, optional
        latitude axis in var, by default -2
    flip : bool, option
        if True, reverse the _data along the latitude index (so that abs(lat) is increasing)


    Returns
    -------
    list of np.ndarray
        northern part of the field and the corrsponding latitude coordinate array
    """

    mask = lat >= 0.0
    lat_north = lat[mask]

    var_north = apply_mask_axis(var, mask, axis)

    if flip:
        var_north = np.flip(var_north, axis)
        lat_north = np.flip(lat_north)

    return lat_north, var_north


# -------       Basin selection     -------- #


def select_box_indices(
    var,
    idx_lat_min,
    idx_lat_max,
    idx_lon_min=None,
    idx_lon_max=None,
    lat_axis=-2,
    lon_axis=-1,
):
    """Extract _data in a box from a field. The box is defined by its indices.

    Parameters
    ----------
    var : np.ndarray
        The field in which we exract the box
    idx_lat_min : int >= 0
        index of the minimum latitude
    idx_lat_max : int >= 0
        index of the maximum latitude
    idx_lon_min : int >= 0, optional
        index of the minimum longitude, by default None
    idx_lon_max : int >= 0, optional
        index of the maximum longitude, by default None
    lat_axis : int, optional
        latitude axis in var, by default -2
    lon_axis : int, optional
        longitude axis in var, by default -1

    Returns
    -------
    np.ndarray
        The field in the box.
    """

    # TODO : Traiter le cas où idx_lon_min>idx_lon_max

    lat_mask = np.array([False] * np.shape(var)[lat_axis])
    lat_mask[idx_lat_min:idx_lat_max] = True
    var_cut_lat = apply_mask_axis(var, lat_mask, lat_axis)

    lon_mask = np.array([False] * np.shape(var)[lon_axis])
    lon_mask[idx_lon_min:idx_lon_max] = True
    var_box = apply_mask_axis(var_cut_lat, lon_mask, lon_axis)

    return var_box


def select_box_lonlat(
    lon,
    lat,
    var,
    lat_min,
    lat_max,
    lon_min=None,
    lon_max=None,
    lon_axis=-1,
    lat_axis=-2,
):
    """Extract _data in a box from a field. The box is defined by its lon/lat coordinates

    Parameters
    ----------
    lon : 1D np.ndarray
        The longitude coordinate
    lat : 1D np.ndarray
        The latitude coordinate
    var : np.ndarray
        The field from which to extract the box
    lat_min : float
        The minimum latitude of the box
    lat_max : float
        The maximum latitude of the box
    lon_min : float, optional
        The minimum longitude of the box, by default None
    lon_max : float, optional
        The maximum longitude of the box, by default None
    lat_axis : int, optional
        latitude axis in var, by default -2
    lon_axis : int, optional
        longitude axis in var, by default -1

    Returns
    -------
    3 np.ndarrays
        longitude coordinate of the box, latitude coordinate of the box, field in the box.
    """

    lat_mask = (lat > lat_min) & (lat < lat_max)
    lat_box = lat[lat_mask]
    var_cut_lat = apply_mask_axis(var, lat_mask, lat_axis)

    if not ((lon_min == None) | (lon_max == None)):
        lon_mask = (lon > lon_min) & (lon < lon_max)
        lon_box = lon[lon_mask]
        var_box = apply_mask_axis(var_cut_lat, lon_mask, lon_axis)
    else:
        lon_box = lon
        var_box = var_cut_lat

    return lon_box, lat_box, var_box


def select_basin(lon, lat, var, basin, lon_axis=-1, lat_axis=-2):
    """Extract basin from a global field, from basin defined as in _basins.py.

    Parameters
    ----------
    lon : 1D np.ndarray
        Longitude coordinate
    lat : 1D np.ndarray
        Latitude coordinate
    var : np.ndarray
        The field from which we extract the basin
    basin : 1D list
        Definition of the basin as [[lon_min, lon_max], [lat_min, lat_max]]
    lat_axis : int, optional
        latitude axis in var, by default -2
    lon_axis : int, optional
        longitude axis in var, by default -1

    Returns
    -------
    3 np.ndarrays
        longitude coordinate of the basin, latitude coordinate of the basin, field in the basin.
    """
    [[lon_min, lon_max], [lat_min, lat_max]] = basin
    lon_basin, lat_basin, var_basin = select_box_lonlat(
        lon, lat, var, lat_min, lat_max, lon_min, lon_max, lon_axis, lat_axis
    )
    return lon_basin, lat_basin, var_basin


# -------       Application of land_mask     -------- #


def remove_land(var, land_sea_mask, land_value=0.0):
    """Remove land from a field.

    Parameters
    ----------
    var : np.ndarray
        The field from which to remove land, two last dimension must correspond to the mask.
    land_sea_mask : np.ndarray
        The land-sea mask corresponding to the two last dimensions of var.
    land_value : float, optional
        value of land in the land-sea mask, by default 0.0

    Returns
    -------
    np.ndarray
        The field with land points set as NaNs
    """
    land_sea_mask[land_sea_mask == land_value] = np.nan
    land_sea_mask[~np.isnan(land_sea_mask)] = 1.0

    return var * land_sea_mask


if __name__ == "__main__":

    import dynamicopy.ncload as ncl

    sp = ncl.var_load("sp", "data_tests/sp.nc")
    lat = ncl.var_load("latitude", "data_tests/sp.nc")
    lon = ncl.var_load("longitude", "data_tests/sp.nc")
    lsm = ncl.var_load("lsm", "data_tests/lsm.nc")[0]

    print(np.shape(sp))
