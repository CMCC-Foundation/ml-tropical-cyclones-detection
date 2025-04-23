import numpy as np
import pandas as pd


def tc_count(tracks):
    """
    Counts the number of tracks per basin and per SSHS category

    Parameters
    ----------
    tracks (pd.DataFrame): A track dataframe

    Returns
    -------
    pd.DataFrame
        A two-entry table with the basins as rows and the categories as columns, values are the counts. Total added.
    """
    tracks = tracks[~tracks.ET].copy()
    storms = tracks.groupby("track_id")[["hemisphere", "basin"]].agg(
        lambda x: x.value_counts().index[0]
    )
    storms["sshs"] = tracks.groupby("track_id")["sshs"].max()
    B = (
        storms.groupby(["sshs", "hemisphere"])[["basin"]]
        .count()
        .reset_index()
        .pivot_table(
            index="hemisphere",
            columns="sshs",
            fill_value=0.0,
            margins=True,
            aggfunc=np.sum,
        )
        .rename(columns={"basin": "count"})
    )
    C = (
        storms.groupby(["sshs", "basin"])[["hemisphere"]]
        .count()
        .reset_index()
        .pivot_table(
            index="basin", columns="sshs", fill_value=0.0, margins=True, aggfunc=np.sum
        )
        .rename(columns={"hemisphere": "count"})
    )
    return (
        B.append(C)
        .drop_duplicates()
        .rename(index={"All": "global"})
        .reindex(["global", "N", "WNP", "ENP", "NI", "NATL", "S", "SP", "SI", "SA"])
    )


def get_freq(tracks):  # TODO : A optimiser
    tracks = tracks[~tracks.ET].copy()
    storms = tracks.groupby("track_id")[["season", "hemisphere", "basin"]].agg(
        lambda x: x.value_counts().index[0]
    )
    basins = (
        storms.reset_index()
        .groupby(["season", "basin"])["track_id"]
        .count()
        .reset_index()
        .pivot_table(index=["basin"], columns="season", fill_value=0)
        .mean(axis=1)
    )
    hemi = (
        storms.reset_index()
        .groupby(["season", "hemisphere"])["track_id"]
        .count()
        .reset_index()
        .pivot_table(index=["hemisphere"], columns="season", fill_value=0)
        .mean(axis=1)
    )
    freq = hemi.append(basins)
    freq.loc["global"] = 0
    if "S" in freq.index:
        freq.loc["global"] += freq.loc["S"]
    if "N" in freq.index:
        freq.loc["global"] += freq.loc["N"]
    return freq.reindex(
        ["global", "N", "WNP", "ENP", "NI", "NATL", "S", "SP", "SI", "SA"]
    )


""" # Version qui avait des erreurs dans les sommes, à revoir si besoin de la distinction en SSHS
def get_freq(tracks):
    tracks = tracks[~tracks.ET].copy()
    storms = tracks.groupby("track_id")[["season", "hemisphere", "basin"]].agg(
        lambda x: x.value_counts().index[0]
    )
    storms["sshs"] = tracks.groupby("track_id")["sshs"].max()

    SH = (
        storms[storms.hemisphere == "S"]
        .groupby(["season", "sshs"])[["basin"]]
        .count()
        .reset_index()
        .pivot_table(index=["sshs"], columns="season", fill_value=0)
        .melt(ignore_index=False)
        .iloc[:, 2:]
        .groupby("sshs")
        .mean()
        .assign(basin="S")
        .reset_index()
    )

    NH = (
        storms[storms.hemisphere == "N"]
        .groupby(["season", "sshs"])[["basin"]]
        .count()
        .reset_index()
        .pivot_table(index=["sshs"], columns="season", fill_value=0)
        .melt(ignore_index=False)
        .iloc[:, 2:]
        .groupby("sshs")
        .mean()
        .assign(basin="N")
        .reset_index()
    )

    basins = (
        storms.groupby(["season", "sshs", "basin"])
        .count()
        .reset_index()
        .pivot_table(index=["basin", "sshs"], columns="season", fill_value=0)
        .melt(ignore_index=False)
        .iloc[:, 2:]
        .groupby(["sshs", "basin"])
        .mean()
        .reset_index()
    )

    freq = SH.append(NH).append(basins).pivot_table(index="basin", columns="sshs", fill_value=0.0, margins=True, aggfunc=np.sum).drop("All")

    freq.loc["global"] = freq.loc["S"] + freq.loc["N"]
    freq.columns = freq.columns.get_level_values(1)
    return freq.reindex(
        ["global", "N", "WNP", "ENP", "NI", "NATL", "S", "SP", "SI", "SA"]
    )
"""


def prop_intense(freq, sshs_min=4):
    """
    Retrieve the proportion of intense tc among all

    Parameters
    ----------
    freq (pd.Dataframe): output from the get_freq function

    Returns
    -------
    pd.Dataframe
        total freq, intense freq, intense prop
    """
    cat_45_cols = list(freq.columns[:-1] >= sshs_min) + [False]
    freq_45 = freq.loc[:, cat_45_cols].sum(axis=1)
    prop_45 = freq_45 / freq.loc[:, "All"]
    return freq[["All"]].assign(intense=freq_45).assign(prop=prop_45)


def storm_stats(tracks):
    """
    Statistics about each track

    Parameters
    ----------
    tracks (pd.Dataframe): The track dataframe

    Returns
    -------
    pd.Dataframe
        Grouped dataframe of the initial one
    """
    tracks = tracks.copy()
    tracks["wind10"] = tracks.wind10.round(2)
    tracks.loc[tracks.ET.isna(), "ET"] = False
    storms = (
        tracks.groupby(["track_id"])[["hemisphere", "basin", "season", "month"]].agg(
            lambda x: x.value_counts().index[0]
        )
        # TODO : line responsible for the slow behavior / remplacer par pd.Series.mode : https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
        # Teste, améliore de 34s -> 21s, mais pb dans le cas limite ou même nb de points dans 2 bassins, a gerer
        .reset_index()
    )
    storms = storms.merge(
        (tracks.groupby(["track_id"])[["time"]].count() / 4).reset_index()
    )
    # Intensity stats : Only on tropical part
    tracks = tracks[(1 - tracks.ET) == 1]
    storms = storms.merge(
        tracks.groupby(["track_id"])[["sshs", "wind10"]].max().reset_index()
    )
    storms = storms.merge(tracks.groupby(["track_id"])[["slp"]].min().reset_index())

    storms = storms.merge(  # Wind lat
        storms[["track_id", "wind10"]]
        .merge(tracks[["track_id", "wind10", "lat", "time"]])
        .groupby("track_id")
        .agg(lambda t: t.mean())
        .reset_index()
        .rename(columns={"lat": "lat_wind", "time": "time_wind"})
        .round(2),
        how="outer",
    )
    storms = storms.merge(  # slp lat
        storms[["track_id", "slp"]]
        .merge(tracks[["track_id", "slp", "lat", "time"]])
        .groupby("track_id")
        .agg(lambda t: t.mean())
        .reset_index()
        .rename(columns={"lat": "lat_slp", "time": "time_slp"}),
        how="outer",
    )
    return storms


def propagation_speeds(tracks): # TODO : Probleme quand un point traverse le méridien de greenwhich
    """
    Return the propagation speed of each track

    Parameters
    ----------
    tracks (pd.Dataframe): The track dataframe

    Returns
    -------
    dict
        keys are the track ids, values the arrays corresponding to propagation speed along the trajectory.
    """

    speeds = {}
    for t in tracks.track_id.unique():
        T = tracks[tracks.track_id == t]
        dlon = T[:-1].lon.values - T[1:].lon.values
        dlat = T[:-1].lat.values - T[1:].lat.values
        dist = np.sqrt(dlon ** 2 + dlat ** 2)  # Vérifier les aspects de GCD etc.
        speeds[t] = dist * 100 / 6
    return speeds  # Vitesses en centième de degré par heure


def genesis_points(tracks):
    """
    Return the first point of each track

    Parameters
    ----------
    tracks (pd.Dataframe)

    Returns
    -------
    pd.Dataframe
    """
    return (
        tracks.sort_values("time")
        .groupby("track_id")[
            ["hemisphere", "basin", "season", "month", "time", "lon", "lat"]
        ]
        .first()
    )
