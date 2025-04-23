import numpy as np
from ._basins import *
from shapely.geometry import Point
import pandas as pd
import pickle as pkl


def add_season(tracks):
    """
    Add (or changes) the season columns in a track dataframe.

    Parameters
    ----------
    tracks (pd.DataFrame): The track dataframe

    Returns
    -------
    pd.DataFrame
        The track dataframe with the new season column
    """
    if "season" in tracks.columns:
        tracks = tracks.drop(columns="season")
    group = (
        tracks.groupby(["track_id"])[["year", "month"]]
        .mean()
        .astype(int)
        .join(
            tracks.groupby("track_id")[
                ["hemisphere"]
            ].last(),  # .agg(pd.Series.mode), Assumption: last point of the track is in the right hemisphere
            on="track_id",
        )
    )
    hemi, yr, mth = group.hemisphere.values, group.year.values, group.month.values
    season = np.where(hemi == "N", yr, None)
    season = np.where((hemi == "S") & (mth >= 7), yr + 1, season)
    season = np.where((hemi == "S") & (mth <= 6), yr, season)
    group["season"] = season.astype(int)
    tracks = tracks.join(group[["season"]], on="track_id")
    return tracks


def get_time(year, month, day, hour):
    """
    Get np.datetime64 array corresponding to year, month, day and hour arrays

    Parameters
    ----------
    year (np.array or pd.Series)
    month (np.array or pd.Series)
    day (np.array or pd.Series)
    hour (np.array or pd.Series)

    Returns
    -------
    np.array or pd.Series
        The corresponding np.datetime64
    """
    time = (
        year.astype(str)
        + "-"
        + month.astype(str)
        + "-"
        + day.astype(str)
        + " "
        + hour.astype(str)
        + ":00"
    ).astype('datetime64[ns]') # updated by Davide Donno
    return time


# TODO : Optimiser cette fonction


def get_basin_gpd(tracks):
    """
    Get the basins corresponding to given lon and lat (More elegant but less fast than get_basin)

    Parameters
    ----------
    lon (np.array or pd.Series)
    lat (np.array or pd.Series)

    Returns
    -------
    list
        basins list
    """

    import geopandas as gpd
    with open("dynamicopy/_data/basins_geopandas.pkl", "rb") as handle:
        gdf = pkl.load(handle)

    df = tracks[["lon", "lat"]]
    df = df.assign(coords=list(zip(df["lon"], df["lat"])))
    df["coords"] = df["coords"].apply(Point)
    points = gpd.GeoDataFrame(df, geometry="coords", crs=gdf.crs)
    pointInPolys = gpd.tools.sjoin(points, gdf, how="left")

    return pointInPolys.basin


def get_basin(lon, lat):
    """
    Get the basins corresponding to given lon and lat

    Parameters
    ----------
    lon (np.array or pd.Series)
    lat (np.array or pd.Series)

    Returns
    -------
    list
        basins list
    """
    basin = []
    for x, y in zip(lon, lat):
        if y < 0:
            if 20 < x <= 135:
                basin.append("SI")
            elif 135 < x <= 295:
                basin.append("SP")
            else:
                basin.append("SA")
        else:
            if 0 <= x <= 30:
                basin.append("MED")
            elif 30 < x <= 100:
                basin.append("NI")
            elif 100 < x <= 180:
                basin.append("WNP")
            elif 180 < x <= 260:
                basin.append("ENP")
            elif 290 < x <= 360:
                basin.append("NATL")
            elif y >= 20:
                basin.append("NATL")
            else:
                if y >= 20 / 35 * (295 - x):
                    basin.append("NATL")
                else:
                    basin.append("ENP")

                # if Point(x,y).within(NH["ENP"]):
                #    basin.append("ENP")
                # else :
                #    basin.append("NATL")
    return basin


_Simpson_pres_thresholds = [990, 980, 970, 965, 945, 920]
_Klotzbach_pres_thresholds = [1005, 990, 975, 960, 945, 925]


def sshs_from_pres(p, classification="Klotzbach"):
    """
    Get the SSHS corresponding to the pressure

    Parameters
    ----------
    p (float or np.array or pd.Series): Pressure in hPa

    Returns
    -------
    float or np.array
        SSHS category
    """
    if classification == "Klotzbach":
        p0, p1, p2, p3, p4, p5 = _Klotzbach_pres_thresholds
    else:
        p0, p1, p2, p3, p4, p5 = _Simpson_pres_thresholds
    sshs = np.where(p > p0, -1, None)
    sshs = np.where((sshs == None) & (p >= p1), 0, sshs)
    sshs = np.where((sshs == None) & (p >= p2), 1, sshs)
    sshs = np.where((sshs == None) & (p >= p3), 2, sshs)
    sshs = np.where((sshs == None) & (p >= p4), 3, sshs)
    sshs = np.where((sshs == None) & (p >= p5), 4, sshs)
    sshs = np.where((sshs == None) & (~np.isnan(p)), 5, sshs)
    sshs = np.where(sshs == None, np.nan, sshs)
    return sshs


def sshs_from_wind(wind):
    """
    Get the SSHS corresponding to the wind

    Parameters
    ----------
    wind (float or np.array or pd.Series): wind in m/s

    Returns
    -------
    float or np.array
        SSHS category
    """
    sshs = np.where(wind <= 16, -1, None)
    sshs = np.where((sshs == None) & (wind < 29), 0, sshs)
    sshs = np.where((sshs == None) & (wind < 38), 1, sshs)
    sshs = np.where((sshs == None) & (wind < 44), 2, sshs)
    sshs = np.where((sshs == None) & (wind < 52), 3, sshs)
    sshs = np.where((sshs == None) & (wind < 63), 4, sshs)
    sshs = np.where((sshs == None) & (~np.isnan(wind)), 5, sshs)
    sshs = np.where(sshs == None, np.nan, sshs)
    return sshs
