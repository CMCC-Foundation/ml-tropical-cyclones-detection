# -*- coding: utf-8 -*-

# Utilitary functions to use inside the module

import numpy as np
import xarray as xr


def nbp2km(nbp):
    return np.sqrt(510e6 / (nbp * 1e3))


### ======================================================= ###
###          Utilitary _data manipulation functions          ###
### ======================================================= ###


def idx_closest(A, x):
    """Returns the index of the closest value to x in array A

    Parameters
    ----------
    A : list or 1D np.ndarray
        The array is which to find the closest value to x
    x : float or int
        The value we want to find the closest entry to in A

    Returns
    -------
    int
        The coordinate of the closest value to x in A
    """
    return np.argmin(abs(np.array(A) - x))


def sign_change_detect(A):
    """Indicates the index of the sign change in A

    Parameters
    ----------
    A : list or 1D array
        The _data in which we want to detect the sign change

    Returns
    -------
    int
        The index of the first value whose sign is different from the sign of the first value.
    """
    sign = np.sign(A)
    change = sign != sign[0]
    return np.min(np.where(change == True))


def hist2d(bdd, weights=None, n=None, resolution=4):
    if n == None:
        n = (bdd.time.dt.year.max() - bdd.time.dt.year.min() + 1) * 12
    H, X, Y = np.histogram2d(
        bdd.lon,
        bdd.lat,
        weights=weights,
        bins=[int(360 / resolution), int(180 / resolution)],
        range=((0, 360), (-90, 90)),
    )
    return (X[1:] + X[:-1]) / 2, (Y[1:] + Y[:-1]) / 2, H / n


if __name__ == "__main__":
    pass
