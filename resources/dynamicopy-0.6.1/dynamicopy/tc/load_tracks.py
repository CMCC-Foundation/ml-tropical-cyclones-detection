import numpy as np

from .utils import *
import pandas as pd
import pickle as pkl
import xarray as xr
import numpy as np

"""
Format for loading the tracks data : 
track_id    time                lon     lat     hemisphere  basin   season  sshs    slp     wind10  year    month   day    hour 
str         np.datetime64[ns]   float   float   str         str     str     int     float   float   int     int     int    int

0 <= lon <= 360
"""


def load_TEtracks(
    file="tests/tracks_ERA5.csv",
    NH_seasons=None,
    SH_seasons=None,
    surf_wind_col="wind10",
    slp_col="slp",
    get_basins=True,
    get_seasons=True,
):
    """
    Parameters
    ----------
    file (str): csv file from TempestExtremes StitchNodes output
    NH_season (list of 2 ints): first and last season in the northern hemisphere
    SH_season (list of 2 ints): first and last season in the southern hemisphere
    surf_wind_col (str): Name of the column with the surface wind to output.
    slp_col (str): Name of the column with the sea-level pressure. If None, no sshs computation.
    get_basins (bool): Set to False if you don't need to get the basins.
    get_seasons (bool): Set to False if you don't need to the the seasons.

    Returns
    -------
    pd.DataFrame
        Columns as described in the module header
    """

    ## Read file
    if isinstance(file, str):
        tracks = pd.read_csv(file, index_col=0)
    else:
        tracks = file
    if tracks.columns.str[0][1] == " ":
        tracks = tracks.rename(columns={c: c[1:] for c in tracks.columns[1:]})
    tracks = tracks.rename(columns={surf_wind_col: "wind10", slp_col: "slp"})

    ## Geographical attributes
    tracks.loc[tracks.lon < 0, "lon"] += 360
    tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    if get_basins:
        tracks["basin"] = get_basin(tracks.lon.values, tracks.lat.values)
    else:
        tracks["basin"] = np.nan

    ## Temporal attributes
    tracks["time"] = get_time(tracks.year, tracks.month, tracks.day, tracks.hour)
    if get_seasons:
        tracks = add_season(tracks)
        if not ((NH_seasons == None) & (SH_seasons == None)):
            tracks = tracks[
                ((tracks.season >= NH_seasons[0]) & (tracks.season <= NH_seasons[1]))
                | (tracks.hemisphere == "S")
            ]
            tracks = tracks[
                ((tracks.season >= SH_seasons[0]) & (tracks.season <= SH_seasons[1]))
                | (tracks.hemisphere == "N")
            ]
    else:
        tracks["season"] = np.nan

    ## Intensity attributes
    if slp_col != None:
        if tracks[slp_col].mean() > 10000:
            tracks[slp_col] /= 100
        tracks["sshs"] = sshs_from_pres(tracks.slp.values)
    else:
        tracks["sshs"] = np.nan
    return tracks


_HRMIP_TRACK_data_vars = [
    "vor_tracked",
    "lon2",
    "lat2",
    "vor850",
    "lon3",
    "lat3",
    "vor700",
    "lon4",
    "lat4",
    "vor600",
    "lon5",
    "lat5",
    "vor500",
    "lon6",
    "lat6",
    "vor250",
    "lon7",
    "lat7",
    "wind10",
]
_TRACK_data_vars = [
    "vor_tracked",
    "lon1",
    "lat1",
    "vor850",
    "lon2",
    "lat2",
    "vor700",
    "lon3",
    "lat3",
    "vor600",
    "lon4",
    "lat4",
    "vor500",
    "lon5",
    "lat5",
    "vor400",
    "lon6",
    "lat6",
    "vor300",
    "lon7",
    "lat7",
    "vor200",
    "lon8",
    "lat8",
    "slp",
    "lon9",
    "lat9",
    "wind925",
    "lon10",
    "lat10",
    "wind10",
]


def read_TRACKfiles(
    file="tests/TRACK/19501951.dat",
    origin="HRMIP",
    season="19501951",
):
    """
    Parameters
    ----------
    file (str): Path to the TRACK output file
    origin (str): 'ERA5' or 'HRMIP'
    season (str): If None, is read from the data

    Returns
    -------
    pd.DataFrame
        Columns as described in the module header
    """
    if origin == "ERA5":
        data_vars = _TRACK_data_vars
        time_format = "calendar"
    else:
        data_vars = _HRMIP_TRACK_data_vars
        time_format = "time_step"
    f = open(file)
    tracks = pd.DataFrame()
    line0 = f.readline()
    line1 = f.readline()
    line2 = f.readline()
    nb_tracks = int(line2.split()[1])
    c = 0
    track_id = 0
    time_step = []
    lon = []
    lat = []
    data = [[]]
    for line in f:
        if line.startswith("TRACK_ID"):
            data = pd.DataFrame(
                np.array(data), columns=data_vars[: np.shape(np.array(data))[1]]
            )
            tracks = tracks.append(
                pd.DataFrame(
                    {
                        "track_id": [track_id] * len(time_step),
                        "time_step": time_step,
                        "lon": lon,
                        "lat": lat,
                    }
                ).join(data)
            )
            c += 1
            if season == None:
                season = line.split()[-1][:-6]
            track_id = str(season) + "-" + str(c)
            time_step = []
            lon = []
            lat = []
            data = []

        elif line.startswith("POINT_NUM"):
            pass
        else:
            time_step.append(line.split()[0])
            lon.append(float(line.split()[1]))
            lat.append(float(line.split()[2]))
            rest = line.split()[3:]
            mask = np.array(rest) == "&"
            data.append(np.array(rest)[~mask])
    f.close()

    SH = tracks.lat.mean() < 0
    if SH:
        tracks["track_id"] = "S" + tracks.track_id
        start = start = np.datetime64(str(int(season) - 1) + "-07-01 00:00:00")
    else:
        tracks["track_id"] = "N" + tracks.track_id
        start = np.datetime64(season + "-01-01 00:00:00")
    if time_format == "calendar":
        tracks["year"] = season
        tracks["month"] = tracks.time_step.str[-6:-4]
        tracks["day"] = tracks.time_step.str[-4:-2]
        tracks["hour"] = tracks.time_step.str[-2:]
        tracks["time"] = get_time(tracks.year, tracks.month, tracks.day, tracks.hour)
        tracks["delta"] = tracks["time"] - np.datetime64(season[-4:] + "-01-01 00")
        tracks["time"] = tracks["delta"] + start
    elif time_format == "time_step":
        tracks["time"] = [
            start + np.timedelta64(ts * 6, "h") for ts in tracks.time_step.astype(int)
        ]
    else:
        print("Please enter a valid time_format")
    time = pd.DatetimeIndex(tracks.time)
    tracks["year"] = time.year
    tracks["month"] = time.month
    tracks["day"] = time.day
    tracks["hour"] = time.hour
    tracks["hemisphere"] = "S" if SH else "N"
    tracks = add_season(tracks)
    tracks["basin"] = get_basin(tracks.lon, tracks.lat)
    if "vor850" in tracks.columns:
        tracks["vor850"] = tracks.vor850.astype(float)
    if "vor_tracked" in tracks.columns:
        tracks["vor_tracked"] = tracks.vor_tracked.astype(float)
    if "slp" not in tracks.columns:
        tracks["slp"] = np.nan
        tracks["sshs"] = np.nan
    else:
        tracks["slp"] = tracks.slp.astype(float)
        tracks["sshs"] = sshs_from_pres(tracks.slp)
    if "wind10" not in tracks.columns:
        tracks["wind10"] = np.nan
    else:
        tracks["wind10"] = tracks.wind10.astype(float)
    tracks["ACE"] = tracks.wind10 ** 2 * 1e-4
    if "wind925" not in tracks.columns:
        tracks["wind925"] = np.nan
    else:
        tracks["wind925"] = tracks.wind925.astype(float)
    return tracks[
        [
            "track_id",
            "time",
            "lon",
            "lat",
            "hemisphere",
            "basin",
            "season",
            "sshs",
            "slp",
            "wind10",
            "ACE",
            "vor_tracked",
            "vor850",
            "wind925",
            "year",
            "month",
            "day",
            "hour",
        ]
    ]


def open_TRACKpkl(
    path="",
    NH_seasons=[1980, 2019],
    SH_seasons=[1981, 2019],
):
    """
    Function to open TRACK files saved as pkl after read_TRACKfiles

    Parameters
    ----------
    path (str): Path to the pkl file
    NH_season (list of 2 ints): first and last season in the northern hemisphere
    SH_season (list of 2 ints): first and last season in the southern hemisphere

    Returns
    -------
    pd.DataFrame
        Columns as described in the module header
    """
    with open(path, "rb") as handle:
        tracks = pkl.load(handle)
    tracks = tracks[
        ((tracks.season >= NH_seasons[0]) & (tracks.season <= NH_seasons[1]))
        | (tracks.hemisphere == "S")
    ]
    tracks = tracks[
        ((tracks.season >= SH_seasons[0]) & (tracks.season <= SH_seasons[1]))
        | (tracks.hemisphere == "N")
    ]
    if "ET" not in tracks.columns:
        tracks["ET"] = np.nan
    return tracks


def load_CNRMtracks(
    file="tests/tracks_CNRM.csv",
    NH_seasons=[1980, 2019],
    SH_seasons=[1981, 2019],
):
    """

    Parameters
    ----------
    file (str): Path to the CNRM tracks file
    NH_season (list of 2 ints): first and last season in the northern hemisphere
    SH_season (list of 2 ints): first and last season in the southern hemisphere

    Returns
    -------
    pd.DataFrame
        Columns as described in the module header
    """
    tracks = pd.read_csv(file)
    tracks = tracks.rename(
        columns={
            "ID": "track_id",
            "Date": "time",
            "Longitude": "lon",
            "Latitude": "lat",
            "Pressure": "slp",
            "Wind": "wind10",
        }
    )
    tracks["hemisphere"] = np.where(tracks.lat > 0, "N", "S")
    tracks["lon"] = np.where(tracks.lon < 0, tracks.lon + 360, tracks.lon)
    tracks["basin"] = get_basin(tracks.lon.values, tracks.lat.values)
    tracks["time"] = tracks.time.astype(np.datetime64)
    tracks["year"] = tracks.time.dt.year
    tracks["month"] = tracks.time.dt.month
    tracks["day"] = tracks.time.dt.day
    tracks["hour"] = tracks.time.dt.hour
    tracks = add_season(tracks)
    tracks = tracks[
        ((tracks.season >= NH_seasons[0]) & (tracks.season <= NH_seasons[1]))
        | (tracks.hemisphere == "S")
    ]
    tracks = tracks[
        ((tracks.season >= SH_seasons[0]) & (tracks.season <= SH_seasons[1]))
        | (tracks.hemisphere == "N")
    ]
    tracks["sshs"] = sshs_from_pres(tracks.slp)
    if "ET" not in tracks.columns:
        tracks["ET"] = np.nan
    return tracks[
        [
            "track_id",
            "time",
            "lon",
            "lat",
            "hemisphere",
            "basin",
            "season",
            "sshs",
            "slp",
            "wind10",
            "year",
            "month",
            "day",
            "hour",
            "ET",
        ]
    ]


def apply_lsm_filter(tracks, lsm_file="tests/lsm1979.nc", lsm_threshold=0.5, min_ts=4):
    """
    Apply lsm filter to a tracks dataset : Any track that has less than min_ts time steps with an lsm value above lsm_threshold will be discarded.

    Parameters
    ----------
    tracks: tracks dataset as loaded by load_TEtracks
    lsm_file: path to the file with the reference land_sea_mask
    lsm_threshold: threshold to define the binary limit between ocean and land in the land-sea mask
    min_ts: minimum number of timesteps that a tracks must have over ocean

    Returns
    -------

    """
    # Take example on identify_ET to improve, with target_lon et sel

    lsm = xr.open_dataset(lsm_file).squeeze().lsm
    lat_idx = (360 - 4 * tracks.lat).astype(int)
    lon_idx = (tracks.lon * 4).astype(int)
    L = np.take(lsm.values, np.ravel_multi_index((lat_idx, lon_idx), lsm.values.shape))
    tracks["lsm"] = L
    t_ocean = tracks[tracks.lsm < 0.65]
    t_filt = tracks[
        tracks.track_id.isin(
            t_ocean.groupby("track_id")
            .time.count()[t_ocean.groupby("track_id").time.count() > 4]
            .index
        )
    ]
    t_out = tracks[
        ~tracks.track_id.isin(
            t_ocean.groupby("track_id")
            .time.count()[t_ocean.groupby("track_id").time.count() > 4]
            .index
        )
    ]
    return t_filt, t_out
