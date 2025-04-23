import numpy as np
import xarray as xr


def identify_ET(tracks, NH_lim, SH_lim, lon_name="longitude", minus_3h=True, fill=True):
    """
    Parameters
    ----------
    tracks (pd.Dataframe) : The dataframe containing the tracks data
    NH_lim (xr.DataArray) : The xarray DataArray with the latitude of the NH STJ as a function of lon and time
    SH_lim (xr.DataArray) : The xarray DataArray with the latitude of the SH STJ as a function of lon and time
    lon_name (str) : The name of the longitude coordinate in the NH_lim and SH_lim objects
    minus_3h (bool) : Set to True if you need to offset the NH_lim and SH_lim objects by 3 hours
    fill (bool) : When set to True, all point in a track after the detection of one ET point are considered ET.

    -------
    pd.Dataframe
        tracks with the ET column
    """

    # Pre-treat latitude limits
    ## Fill NAs in the latitude limits with linear interpolation
    NH_lim = NH_lim.interpolate_na(dim=lon_name).interpolate_na(dim="time")
    SH_lim = SH_lim.interpolate_na(dim=lon_name).interpolate_na(dim="time")
    ## Change time to -3h to fit with tracks
    if minus_3h:
        NH_lim["time"] = NH_lim.time - np.timedelta64(3, "h")
        SH_lim["time"] = SH_lim.time - np.timedelta64(3, "h")
    NH_lim = NH_lim.rename({lon_name: "longitude"})
    SH_lim = SH_lim.rename({lon_name: "longitude"})

    # Pre-treat tracks
    tracks = tracks.reset_index(drop=True)
    # tracks["lon"] = np.floor(tracks.lon * 4)/4
    tracks["lon"] = np.where(tracks.lon > 180, tracks.lon - 360, tracks.lon)

    # Detect ET points
    target_lon = xr.DataArray(tracks.lon, dims="points")
    target_time = xr.DataArray(tracks.time, dims="points")
    tracks["lat_STJ_NH"] = NH_lim.sel(
        time=target_time, longitude=target_lon, method="nearest"
    )
    tracks["lat_STJ_SH"] = SH_lim.sel(
        time=target_time, longitude=target_lon, method="nearest"
    )
    tracks["ET"] = (tracks.lat > tracks.lat_STJ_NH) | (tracks.lat < tracks.lat_STJ_SH)

    if fill:
        # Fill trajectories once one point is ET
        first_times = (
            tracks[tracks.ET == True]
            .sort_values("time")
            .groupby("track_id")
            .time.first()
        )
        t_ET = tracks[
            tracks.track_id.isin(first_times.index)
        ]  # Track that are labelled ET at one point
        t_noET = tracks[
            ~tracks.track_id.isin(first_times.index)
        ]  # Track that are not labelled ET at one point
        t_ET = t_ET.merge(first_times, on="track_id")
        t_ET["ET"] = t_ET.time_x >= t_ET.time_y
        t_ET = t_ET.rename(columns={"time_x": "time"}).drop(columns="time_y")

    return tracks

def remove_ET(tracks, trop_pts=1):
    """
    Parameters
    ----------
    tracks (pd.Dataframe) : The dataframe containing the tracks data
    trop_pts (int): Maximum number of tropical points in an ET track

    Returns
    -------
    tracks_trop (pd.DataFrame) : The tropical trajectories
    tracks_ET (pd.DataFrame) : The ET trajectories
    """
    tracks["trop"] = 1 - tracks.ET
    ET_track_ids = (
        tracks.groupby("track_id")["trop"]
        .sum()
        .index[tracks.groupby("track_id")["trop"].sum() < trop_pts + 1]
    )
    tracks_trop = tracks[~tracks.track_id.isin(ET_track_ids)]
    tracks_ET = tracks[tracks.track_id.isin(ET_track_ids)]
    return tracks_trop, tracks_ET
