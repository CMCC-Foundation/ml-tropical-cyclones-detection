# -*- coding: utf-8 -*-

# Tools for ploting the _data from NetCDF file, using only numpy and matplotlib libs.

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import seaborn as sns
import numpy as np

from .plot import _var2d
from .utils_geo import apply_mask_axis
from .cyclones import load_ibtracs

from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes

GeoAxes._pcolormesh_patched = Axes.pcolormesh

# Ne peut pas fonctionner avec des subplots de la même façon que lon_lat_plot car on ne peut pas changer la projection d'un ax existant. Voire pour faire une fonction supplémentaire pour gérer ça ?
# Actuellement, pour utiliser des subplots, les créer avec plt.subplots(a, b, subplot_kw = {'projection':ccrs.Robinson()})


def lon_lat_plot_map(
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
    projection=ccrs.Robinson(),
    set_global=False,
    coastlines=True,
    grid=False,
    savefig=False,
    filename="saved_fig.png",
):
    """Plot a 2D map of the _data with cartopy.

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
        axis on latitude in var, by default -2
    fig_ax : [type], optional
        [description], by default None
    title : str, optional
        title of the plot, by default ''
    cmap : str, optional
        color palette, by default "bwr"
    colorbar_label : str, optional
        label to be shown beside the colorbar, by default ''
    norm : matplotlib.colors.<any>Norm, optional
        normalization for the colormap, by default None
        Example : with `import matplotlib.colors as c` in header of your script
            * For centering diverging colormap : c.DivergingNorm(vcenter = 0.0)
    smooth : bool, optional
        if True, contourf is used instead of pcolormesh, by default False
    projection : cartopy.crs projection, optional
        projection for the display, by default ccrs.Robinson()
    set_global : bool, optional
        if True, forces to plot the full globe, by default False
    coastlines : bool, optional
        if True, displays the coastlines, by default True
    grid : bool, optional
        if True, displays gridlines, by default False
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

    # Truncate if latitude coordinates array go too far
    mask = ~((lat > 88) | (lat < -88))
    var2D = apply_mask_axis(var2D, mask, axis=lat_axis)
    lat = lat[mask]
    if any(mask == False):
        print("Warning, some values too close to the pole(s) were not displayed.")

    # Plotting
    if fig_ax == None:
        fig = plt.figure()
        ax = plt.axes(projection=projection)
    else:
        fig, ax = fig_ax
    if set_global:
        ax.set_global()
    if coastlines:
        ax.coastlines()
    if grid:
        ax.gridlines(crs=projection)

    if not smooth:
        C = ax.pcolormesh(
            lon,
            lat,
            var2D,
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            shading="nearest",
        )
    else:
        C = ax.contourf(
            lon, lat, var2D, cmap=cmap, norm=norm, transform=ccrs.PlateCarree()
        )
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


def scatterplot_map(
    lons,
    lats,
    fig_ax=None,
    color="k",
    size=1,
    title="",
    projection=ccrs.Robinson(),
    set_global=False,
    coastlines=True,
    grid=False,
    link=False,
    linecolor="k",
    linewidth=1,
):
    """Plot points on a map.

    Parameters
    ----------
    lons : 1D np.ndarray
        longitudes of the points
    lats : 1D np.ndarray
        latitudes of the points
    fig_ax : [type], optional
        [description], by default None
    color : str, optional
        color of the points, can also be a 1D list of the same length than the _data, by default 'k'
    size : float, optional
        size of the points, by default 1
    title : str, optional
        title of the plot, by default ''
    projection : cartopy.crs projection, optional
        projection for the display, by default ccrs.Robinson()
    set_global : bool, optional
        if True, forces to plot the full globe, by default False
    coastlines : bool, optional
        if True, displays the coastlines, by default True
    grid : bool, optional
        if True, displays gridlines, by default False
    link : bool, optional
        if True, displays lines (or trajectories) instead of points, by default False
    linecolor : str, optional
        color of the line, by default 'k'
    linewidth : float, optional
        width of the line, by default 1

    Returns
    -------
    None
        Plots the points in ax
    """
    if fig_ax == None:
        fig = plt.figure()
        ax = plt.axes(projection=projection)
    else:
        ax = fig_ax[1]
    if set_global:
        ax.set_global()
    if coastlines:
        ax.coastlines()
    if grid:
        ax.gridlines()
    ax.scatter(lons, lats, transform=ccrs.PlateCarree(), s=size, c=color)
    if link:
        ax.plot(
            lons, lats, transform=ccrs.PlateCarree(), c=linecolor, linewidth=linewidth
        )

    ax.set_title(title)

    return None


def plot_hist2d(hist, axis, bounds, shrink_cbar=0.8):
    norm = BoundaryNorm(bounds, len(bounds) - 1, clip=True)
    cmap_reds = plt.cm.get_cmap("magma_r", len(bounds))
    colors = list(cmap_reds(np.arange(len(bounds))))
    colors[0] = "white"
    cmap = ListedColormap(colors[:-1], "")
    cmap.set_over(colors[-1])

    p = hist.plot(
        ax=axis,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        extend="max",
        cbar_kwargs={"shrink": shrink_cbar},
    )
    axis.coastlines()
    axis.set_global()


def plot_diff(diff, axis, range=2.0, mid=0.125, delta=0.25, shrink_cbar=0.8):
    bounds = (
        list(np.arange(-range, 0, delta))
        + [-mid, mid]
        + list(np.arange(delta, range + delta, delta))
    )
    norm = BoundaryNorm(bounds, len(bounds) - 1)
    p = diff.plot(
        ax=axis,
        transform=ccrs.PlateCarree(),
        cmap="bwr",
        norm=norm,
        cbar_kwargs={"shrink": shrink_cbar},
        extend="both",
    )
    axis.coastlines()
    axis.set_global()


zooms = {
    "NI": ccrs.NearsidePerspective(70, 20, 35785831 / 2),
    "SI": ccrs.NearsidePerspective(70, -20, 35785831 / 2),
    "AUS": ccrs.NearsidePerspective(135, -20, 35785831 / 2),
    "WNP": ccrs.NearsidePerspective(150, 20, 35785831 / 2),
    "SP": ccrs.NearsidePerspective(200, -20, 35785831 / 2),
    "ENP": ccrs.NearsidePerspective(240, 20, 35785831 / 2),
    "ATL": ccrs.NearsidePerspective(280, 20, 35785831 / 2),
}


if __name__ == "__main__":
    plot_tracks()
