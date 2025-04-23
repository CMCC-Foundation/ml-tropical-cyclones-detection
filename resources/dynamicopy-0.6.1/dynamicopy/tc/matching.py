import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit


def match_tracks(tracks1, tracks2, name1="algo", name2="ib", max_dist = 300, min_overlap=0, ref = True):
    """

    Parameters
    ----------
    tracks1 (pd.DataFrame): the first track dataframe to match
    tracks2 (pd.DataFrame): the second track dataframe to match
    name1 (str): Suffix for the first dataframe
    name2 (str): Suffix for the second dataframe
    max_dist (float) : Threshold for maximum distance between two tracks
    min_overlap (int) : Minimum number of overlapping time steps for matching
    ref (bool) : If True, tracks2 is considered a reference.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the matching tracks with
            the id from both datasets
            the number of matching time steps
            the distance between two tracks
    """
    tracks1, tracks2 = (
        tracks1[["track_id", "lon", "lat", "time"]],
        tracks2[["track_id", "lon", "lat", "time"]],
    )
    # Find corresponding points (same time step, less than 300km)
    merged = pd.merge(tracks1, tracks2, on="time")
    X = np.concatenate([[merged.lat_x], [merged.lon_x]]).T
    Y = np.concatenate([[merged.lat_y], [merged.lon_y]]).T
    merged["dist"] = haversine_vector(X, Y, unit=Unit.KILOMETERS)
    merged = merged[merged.dist <= max_dist]
    # Compute temporal overlap
    temp = (
        merged.groupby(["track_id_x", "track_id_y"])[["dist"]]
        .count()
        .rename(columns={"dist": "temp"})
    )
    # Build a table of all pairs of tracks sharing at least one point
    matches = (
        merged[["track_id_x", "track_id_y"]]
        .drop_duplicates()
        .join(temp, on=["track_id_x", "track_id_y"])
    )
    matches = matches[matches.temp >= min_overlap]
    if ref :
        # For each track of the first set, only keep the track of the second set with the longest overlap
        maxs = matches.groupby("track_id_x")[["temp"]].max().reset_index()
        matches = maxs.merge(matches)[["track_id_x", "track_id_y", "temp"]]
        # In case there remains duplicates:
        # For one track of the first set two tracks of the second correspond with the same overlap
        # We keep the closest one
        dist = merged.groupby(["track_id_x", "track_id_y"])[["dist"]].mean()
        matches = matches.join(dist, on=["track_id_x", "track_id_y"])
        mins = matches.groupby("track_id_x")[["dist"]].min().reset_index()
        matches = mins.merge(matches)[["track_id_x", "track_id_y", "temp", "dist"]]
    else :
        dist = merged.groupby(["track_id_x", "track_id_y"])[["dist"]].mean()
        matches = matches.merge(dist, on = ['track_id_x', 'track_id_y'])
    # Rename columns before output
    matches = matches.rename(
        columns={"track_id_x": "id_" + name1, "track_id_y": "id_" + name2}
    )
    return matches


def merge_duplicates(tracks1, tracks2, matches=None):
    """
    Function to manage cases where tracks from tracks2 match with several tracks in tracks1,
    by merging the duplicates into one track.

    Parameters
    ----------
    tracks1 (pd.Dataframe)
    tracks2 (pd.Dataframe)
    matches (pd.Dataframe): The output from match_tracks on tracks1 and tracks2.
        If None, match_tracks is run on tracks1 and tracks2.

    Returns
    -------
    pd.Dataframe
        The new match dataset
    + tracks1 modified in place
    """
    if type(matches) == type(None):
        matches = match_tracks(tracks1, tracks2)
    c1, c2 = matches.columns[:2]
    count = matches.groupby(c2)[[c1]].nunique()
    duplicates = count[(count > 1).values].index
    print("Handled " + str(len(duplicates)) + " duplicates")
    for i in duplicates:
        merge = matches[matches[c2] == i][c1].values
        new_id = "+".join(merge.astype(str))
        tracks1["track_id"] = tracks1.track_id.replace(
            {old_id: new_id for old_id in merge}
        )
    matches = match_tracks(tracks1, tracks2, c1[3:], c2[3:])
    return matches


def overlap(tracks1, tracks2, matches=None):
    """
    Function computing the overlap between matched tracks.

    Parameters
    ----------
    tracks1 (pd.Dataframe)
    tracks2 (pd.Dataframe)
    matches (pd.Dataframe): The output from match_tracks on tracks1 and tracks2.
        If None, match_tracks is run on tracks1 and tracks2.

    Returns
    -------
    pd.Dataframe
        Match dataset with added deltas
    """
    if type(matches) == type(None):
        matches = match_tracks(tracks1, tracks2)
    c1, c2 = matches.columns[:2].str.slice(3)
    matches = (
        matches.join(
            tracks1.groupby("track_id")[["time"]]
            .min()
            .rename(columns={"time": "tmin_" + c1}),
            on="id_" + c1,
        )
        .join(
            tracks1.groupby("track_id")[["time"]]
            .max()
            .rename(columns={"time": "tmax_" + c1}),
            on="id_" + c1,
        )
        .join(
            tracks2.groupby("track_id")[["time"]]
            .min()
            .rename(columns={"time": "tmin_" + c2}),
            on="id_" + c2,
        )
        .join(
            tracks2.groupby("track_id")[["time"]]
            .max()
            .rename(columns={"time": "tmax_" + c2}),
            on="id_" + c2,
        )
    )

    matches["delta_start"] = matches["tmin_" + c2] - matches["tmin_" + c1]
    matches["delta_end"] = matches["tmax_" + c2] - matches["tmax_" + c1]
    matches["delta_end"] = (
        matches.delta_end.dt.days + matches.delta_end.dt.seconds / 86400
    )
    matches["delta_start"] = (
        matches.delta_start.dt.days + matches.delta_start.dt.seconds / 86400
    )

    return matches[["id_" + c1, "id_" + c2, "temp", "dist", "delta_start", "delta_end"]]
