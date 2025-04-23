# -*- coding: utf-8 -*-

# Tools for ploting the _data from NetCDF file, using only numpy and matplotlib libs.

import numpy as np
import matplotlib.pyplot as plt


def _var2d(var, lon_axis=-1, lat_axis=-2):
    """Returns the 2D field average over dimensions that are not latitude or longitude

    Parameters
    ----------
    var : np.ndarray
        field
    lon_axis : int, optional
        axis of longitude in var, by default -1
    lat_axis : int, optional
        axis of latitude in var, by default -2

    Returns
    -------
    2D np.ndarray
        2D averaged field
    """
    L = len(np.shape(var))
    if lon_axis < 0:
        lon_axis = L + lon_axis
    if lat_axis < 0:
        lat_axis = L + lat_axis
    axes_avg = list(range(L))
    axes_avg.remove(lon_axis)
    axes_avg.remove(lat_axis)

    return np.mean(var, axis=tuple(axes_avg))


def lon_lat_plot(
    lon,
    lat,
    var,
    lon_axis=-1,
    lat_axis=-2,
    fig_ax=None,
    title="",
    cmap="bwr",
    colorbar_label="",
    norm=None,
    smooth=False,
    savefig=False,
    filename="saved_fig.png",
):
    """Plot a 2D map of the _data.

    Parameters
    ----------
    lon : 1D np.ndarray
        longitude coordinate
    lat : 1D np.ndarray
        latitude coordinate
    var : np.ndarray
        field to plot
    lon_axis : int, optional
        axis of longitude in var, by default -1
    lat_axis : int, optional
        axis of latitude in var, by default -2
    fig_ax : (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot), optional
        2-tuple of fig and ax for the plot, by default plt.subplots()
    title : str, optional
        title of the plot, by default ''
    cmap : str, optional
        color palette, by default "bwr"
    colorbar_label : str, optional
        label of the colorbar, by default ''
    norm : matplotlib.colors.<any>Norm, optional
        normalization for the colormap, by default None
        Example : with `import matplotlib.colors as c` in header of your script
            * For centering diverging colormap : c.DivergingNorm(vcenter = 0.0)
    smooth : bool, optional
        if True, contourf is used instead of pcolormesh, by default False
    savefig : bool, optional
        if True, prints the figure in the file specified with filename, by default False
    filename : str, optional
        File in which the figure will be saved if savefig is True, by default 'saved_fig.png'

    Returns
    -------
    None
        Plots the map in ax
    """
    # Obtain 2D variable to plot
    var2D = _var2d(var, lon_axis, lat_axis)

    # Plotting
    if fig_ax == None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    if not smooth:
        C = ax.pcolormesh(lon, lat, var2D, cmap=cmap, norm=norm, shading="nearest")
    else:
        C = ax.contourf(lon, lat, var2D, cmap=cmap, norm=norm)
    fig.colorbar(
        C,
        ax=ax,
        label=colorbar_label,
    )
    ax.set_ylabel("Latitude (°)")
    ax.set_xlabel("Longitude (°)")
    ax.set_title(title)

    if savefig:
        print("Saving figure as" + filename)
        fig.savefig(filename)

    return None


def zonal_plot(
    lat,
    lev,
    var,
    lat_axis=-1,
    lev_axis=-2,
    fig_ax=None,
    title="",
    cmap="bwr",
    colorbar_label="",
    norm=None,
    smooth=False,
    savefig=False,
    filename="saved_fig.png",
):
    """Plot a 2D map of the _data.

    Parameters
    ----------
    lat : 1D np.ndarray
        latitude coordinate
    lev : 1D np.ndarray
        pressure coordinate in Pa
    var : np.ndarray
        field to plot
    lon_axis : int, optional
        axis of longitude in var, by default -1
    lat_axis : int, optional
        axis of latitude in var, by default -2
    fig_ax : (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot), optional
        2-tuple of fig and ax for the plot, by default plt.subplots()
    title : str, optional
        title of the plot, by default ''
    cmap : str, optional
        color palette, by default "bwr"
    colorbar_label : str, optional
        label of the colorbar, by default ''
    norm : matplotlib.colors.<any>Norm, optional
        normalization for the colormap, by default None
        Example : with `import matplotlib.colors as c` in header of your script
            * For centering diverging colormap : c.DivergingNorm(vcenter = 0.0)
    smooth : bool, optional
        if True, contourf is used instead of pcolormesh, by default False
    savefig : bool, optional
        if True, prints the figure in the file specified with filename, by default False
    filename : str, optional
        File in which the figure will be saved if savefig is True, by default 'saved_fig.png'

    Returns
    -------
    None
        Plots the map in ax
    """

    # Obtain 2D variable to plot
    var2D = _var2d(var, lat_axis, lev_axis)

    # Plotting
    if fig_ax == None:
        fig, ax = plt.subplots(figsize=[10, 5])
    else:
        fig, ax = fig_ax

    if not smooth:
        C = ax.pcolormesh(
            lat, lev / 100, var2D, cmap=cmap, norm=norm, shading="nearest"
        )
    else:
        C = ax.contourf(lat, lev / 100, var2D, cmap=cmap, norm=norm)
    fig.colorbar(
        C,
        ax=ax,
        label=colorbar_label,
    )
    ax.set_ylim(np.max(lev / 100), np.min(lev / 100))
    ax.set_ylabel("Pressure / hPa")
    ax.set_xlabel("Latitude / °")
    ax.set_title(title)

    if savefig:
        print("Saving figure as" + filename)
        fig.savefig(filename)

    return None


if __name__ == "__main__":
    pass
