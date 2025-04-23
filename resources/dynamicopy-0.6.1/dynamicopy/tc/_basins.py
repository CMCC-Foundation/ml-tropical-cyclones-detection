# -*- coding: utf-8 -*-

"""
Oceanic basins as defined by Knutson et al. 2020 appendix.
"""

from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    from .cartoplot import *
except ImportError:
    pass
import numpy as np
import matplotlib.ticker as mticker

NATL = Polygon(((260, 90), (360, 90), (360, 0), (295, 0), (260, 20)))
ENP = Polygon(((180, 90), (260, 90), (260, 20), (295, 0), (180, 0)))
WNP = Polygon(((100, 0), (100, 90), (180, 90), (180, 0)))
NI = Polygon(((30, 0), (30, 90), (100, 90), (100, 0)))
MED = Polygon(((0, 0), (0, 90), (30, 90), (30, 0)))
NH = {"NATL": NATL, "ENP": ENP, "WNP": WNP, "NI": NI, "MED": MED}

SI = Polygon(((20, -90), (20, 0), (135, 0), (135, -90)))
SP = Polygon(((135, 0), (135, -90), (295, -90), (295, 0)))
SA1 = Polygon(((295, -90), (295, 0), (360, 0), (360, -90)))
SA2 = Polygon(((20, -90), (20, 0), (0, 0), (0, -90)))
SA = MultiPolygon([SA1, SA2])
SA_plot = Polygon(((295, -90), (295, 0), (380, 0), (380, -90)))
SH = {"SI": SI, "SP": SP, "SA": SA}

basins = dict(SH, **NH)


def _save_basins_shapefile():
    """
    Code from Stackoverflow :
    https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
    """
    from osgeo import ogr
    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName("Esri Shapefile")
    ds = driver.CreateDataSource("dynamicopy/_data/basins.shp")
    layer = ds.CreateLayer("", None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    for b in basins:
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField("id", 123)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(basins[b].wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)
        feat = geom = None  # destroy these

    # Save and close everything
    ds = layer = feat = geom = None


def plot_basins(show=True, save=None):
    """
    Plot the basins according to the Knutson definition

    Parameters
    ----------
    show: Use plt.show in the end to display the figure
    save: Name of the file to save. If None, no file is saved.

    Returns
    -------
    A plot.
    """

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)}
    )
    ax.coastlines()
    ax.set_global()
    gl = ax.gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator([20, 30, 100, 135, 180, -100, -65])
    gl.ylocator = mticker.FixedLocator([-90, 0, 20, 90])
    gl.xlines = False
    gl.ylines = False

    for basin, name in zip(
        [NATL, ENP, WNP, NI, MED, SI, SP, SA_plot],
        ["NATL", "ENP", "WNP", "NI", "MED", "SI", "SP", "SATL"],
    ):
        plt.plot(
            basin.exterior.xy[0],
            basin.exterior.xy[1],
            transform=ccrs.PlateCarree(),
            color="k",
        )
        plt.text(
            basin.centroid.x - 10,
            np.sign(basin.centroid.y) * 25,
            name,
            transform=ccrs.PlateCarree(),
            fontweight="bold",
        )
    if show:
        plt.show()
    if save != None:
        plt.savefig(save)


# TODO (?) : Définir aussi les régions WMO
