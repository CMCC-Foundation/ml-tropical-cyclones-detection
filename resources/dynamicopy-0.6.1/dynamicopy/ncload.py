# -*- coding: utf-8 -*-

# Tools to load variables from a NetCDF File.
# Note : I strongly advice using nco and cdo to pre-process file instead of python. (e.g. concatenation)

from netCDF4 import Dataset
import numpy as np

# TODO: Revoir avec xarray
def var_load(varname, file_path, group=None, subgroup=None, silent=True):
    """Loads a variable from a netCDF file.

    Parameters
    ----------
    varname : str
        The name of the variable in the file;
    file_path : str
        The path (relative or absolute) to the netCDF file;
    group : str, optional
        Group in which the variable is in the netCDF file if necessary (default None);
    subgroup : str, optional
        Subgroup in which the variable is in the netCDF file if necessary (default None);
    silent : bool, optional
        Indicates whether to print infos about the loaded variable (default True);

    Returns
    -------
    np.ndarray
        A numpy array corresponding to the variable in the file;
    """

    if not silent:
        print("Loading " + varname + "...")

    f_in = Dataset(file_path)

    if group == None:
        var = f_in.variables[varname][:]
    else:
        if subgroup == None:
            var = f_in.groups[group].variables[varname][:]
        else:
            var = f_in.groups[group].groups[subgroup].variables[varname][:]
    f_in.close()

    if not (silent):
        print("Variable dimensions: " + str(np.shape(var)))
    return np.array(var)


def get_lon_lat(file_path, lon_name="lon", lat_name="lat"):
    """Loads longitude and latitude coordinates from a netCDF file

    Parameters
    ----------
    file_path : str
        The path (relative or absolute) to the netCDF file;
    lon_name : str, optional
        name of the longitude coordinate in the file (default 'lon');
    lat_name : str, optional
        name of the latitude coordinate in the file (default 'lat');

    Returns
    -------
    list of np.ndarray
        Arrays of longitude and latitude in the netCDF file;
    """

    f_in = Dataset(file_path)
    lat = f_in[lat_name][:]
    lon = f_in[lon_name][:]
    f_in.close()

    return np.array(lon), np.array(lat)


def var_load_from_limit(
    varname, limit_file="limit.nc", lon_name="longitude", lat_name="latitude"
):
    """Loads a field from a limit.nc file (LMDZ standard structure)

    Parameters
    ----------
    varname : str
        Name of the variable to load in the file
    limit_file : str
        The path (relative or absolute) to the limit.nc file;
    lon_name : str, optional
        name of the longitude coordinate in the file (default 'longitude');
    lat_name : str, optional
        name of the latitude coordinate in the file (default 'latitute');

    Returns
    -------
    list of np.ndarray (lon, lat, var)
        The field reshaped in lon/lat coordinates and the corresponding coordinate arrays;
    """

    lon = var_load(lon_name, limit_file)
    lat = var_load(lat_name, limit_file)
    var = var_load(varname, limit_file)

    lat_dim_no_pole = len(np.unique(lat)) - 2
    lon_dim = len(np.unique(lon))

    if len(np.shape(var)) == 1:
        var_north_pole = var[0]
        var_south_pole = var[-1]
        var_no_pole = var[1:-1]

        var_no_pole_reshape = np.reshape(var_no_pole, [lat_dim_no_pole, lon_dim])
        var_north_pole_reshape = [[var_north_pole] * lon_dim]
        var_south_pole_reshape = [[var_south_pole] * lon_dim]
        var_reshape = np.concatenate(
            [var_north_pole_reshape, var_no_pole_reshape, var_south_pole_reshape], 0
        )

    elif len(np.shape(var)) == 2:
        var_north_pole = var[:, 0]
        var_south_pole = var[:, -1]
        var_no_pole = var[:, 1:-1]

        var_no_pole_reshape = [
            np.reshape(var_no_pole[i], [lat_dim_no_pole, lon_dim])
            for i in range(len(var))
        ]
        var_north_pole_reshape = [
            [[var_north_pole[i]] * lon_dim] for i in range(len(var))
        ]
        var_south_pole_reshape = [
            [[var_south_pole[i]] * lon_dim] for i in range(len(var))
        ]
        var_reshape = np.concatenate(
            [var_north_pole_reshape, var_no_pole_reshape, var_south_pole_reshape], 1
        )

    else:
        print("Problem with the variable dimensions")
        return None

    lat_reshape = np.flip(np.unique(lat))
    lon_reshape = np.unique(lon)

    return lon_reshape, lat_reshape, var_reshape


def change_limit(newfield, fieldname, limit_file="limit.nc"):
    """Function to change a field in a limit.nc file (LMDZ standard structure)

    Parameters
    ----------
    newfield : np.ndarray
        2D or 3D field corresponding to the new values for a limit variable.
    fieldname : np.ndarray
        name of the field to change.
    limit_file : str
        The path (relative or absolute) to the limit.nc file;

    Returns
    -------
    """
    north_pole = np.transpose([np.mean(newfield[:, 0], -1)])
    south_pole = np.transpose([np.mean(newfield[:, -1], -1)])
    if len(np.shape(newfield)) == 3:  # If time dimension
        newfield_flat = [newfield[t, 1:-1].flatten() for t in range(len(newfield))]
        newfield_flat = np.concatenate([north_pole, newfield_flat, south_pole], 1)
    else:
        newfield_flat = newfield[1:-1].flatten()
        newfield_flat = np.concatenate([north_pole, newfield_flat, south_pole])

    f_in = Dataset(limit_file, "a")
    f_in.variables[fieldname][:] = newfield_flat
    f_in.close()

    return None


def change_start(newfield, fieldname, start_file="start.nc"):
    """Function to change a field in a start.nc file (LMDZ standard structure)

    Parameters
    ----------
    newfield : np.ndarray
        2D field corresponding to the new values for a limit variable.
    fieldname : np.ndarray
        name of the field to change.
    start_file : str
        The path (relative or absolute) to the start.nc file;

    Returns
    -------
    """

    f_in = Dataset(start_file, "a")
    f_in.variables[fieldname][:, :] = newfield
    f_in.close()

    return None


if __name__ == "__main__":
    pass
