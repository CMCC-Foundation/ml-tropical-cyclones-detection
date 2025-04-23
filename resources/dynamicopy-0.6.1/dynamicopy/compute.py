# -*- coding: utf-8 -*-

# Computation tools to manipulate atmospheric variables.

import numpy as np
import xarray as xr

# from .utils_geo import get_south, get_north, apply_mask_axis

### ==================================== ###
###         Derivated variables          ###
### ==================================== ###


def omega2w(omega, p, T):
    """Convert omega (?) to w (vertical velocity)

    Parameters
    ----------
    omega : np.ndarray
        The ? field
    p : np.ndarray
        Pressure coordinate
    T : np.ndarray
        Temperature field

    Returns
    -------
    np.ndarray
        The vertical velocity field
    """
    omega, p, T = np.array(omega), np.array(p), np.array(T)
    R_gas = 287.058
    g = 9.80665
    rho = p / (R_gas * T)
    w = -omega / (rho * g)
    return w


def get_dx_dy(lon, lat):
    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    R = 6371000  # Earth radius
    dy = R * dlat * np.pi / 180  # Vertical size of a cell
    # Horizontal size of a cell depending on latitude
    lat_rad = lat * np.pi / 180  # Latitudes in rad
    r = np.sin(np.pi / 2 - abs(lat_rad)) * R
    dx = r * dlon * np.pi / 180
    dx = np.transpose([(dx[1:] + dx[:-1]) / 2] * (len(lon) - 1))
    return dx, dy


def compute_OWZ_from_files(
    u_file,
    v_file,
    vo_file=None,
    owz_file=None,
    level=[850, 500],
    u_name="u",
    v_name="v",
    vo_name="vo",
    lon_name="longitude",
    lat_name="latitude",
    p_name="level",
):
    """

    Parameters
    ----------
    u_file, v_file : str
        Paths to the respective files containing zonal and meridional wind field
    vo_file : str
        Path to the file containing the vorticity field. To be implemented : If None, compute vorticity from u and v.
    OWZ_file : str
        Path to which the OWZ field will be written. If None, not saved.
    owz_name : str
        Name of the owz to be written i the file if applicable.
    u_name, v_name, vo_name, lon_name, lat_name : str
        Names of zonal wind, meridional wind, vorticity, longitude, latitude respectively in the files.

    Returns
    -------
    OWZ : xr.Dataset
        OWZ field
    """
    if level != None:
        u = (
            xr.open_dataset(u_file)[u_name].rename({p_name: "level"}).sel(level=level)
        )  # .squeeze()
        v = (
            xr.open_dataset(v_file)[v_name].rename({p_name: "level"}).sel(level=level)
        )  # .squeeze()
        if vo_file != None:
            vo = (
                xr.open_dataset(vo_file)[vo_name]
                .rename({p_name: "level"})
                .sel(level=level)
            )  # .squeeze()
    else:
        u = xr.open_dataset(u_file).rename({p_name: "level"})  # .squeeze()
        v = xr.open_dataset(v_file).rename({p_name: "level"})  # .squeeze()
        if vo_file != None:
            vo = xr.open_dataset(vo_file).rename({p_name: "level"})  # .squeeze()

    OWZ = []
    if "time" in u.dims:
        for i, t in enumerate(u.time):
            OWZ.append([])
            for p in level:
                lon, lat, E, F = compute_E_F(
                    u.sel(time=t).sel(level=p).values,
                    v.sel(time=t).sel(level=p).values,
                    u[lat_name].values,
                    u[lon_name].values,
                )
                E = xr.DataArray(E, coords=[lat, lon], dims=[lat_name, lon_name])
                E = E.interp_like(u, kwargs={"fill_value": "extrapolate"})
                F = xr.DataArray(F, coords=[lat, lon], dims=[lat_name, lon_name])
                F = F.interp_like(u, kwargs={"fill_value": "extrapolate"})

                if vo_file == None:
                    lon, lat, vo_t = compute_vort(
                        u.sel(time=t).sel(level=p).values,
                        v.sel(time=t).sel(level=p).values,
                        u[lat_name].values,
                        u[lon_name].values,
                    )
                    vo_t = xr.DataArray(
                        vo_t, coords=[lat, lon], dims=[lat_name, lon_name]
                    )
                    vo_t = vo_t.interp_like(u, kwargs={"fill_value": "extrapolate"})
                else:
                    vo_t = vo.isel(time=i).sel(level=p)

                OWZ[-1].append(
                    compute_OWZ(vo_t.values, E.values, F.values, F[lat_name].values)
                )
                del E, F, vo_t
    else:
        for p in level:
            lon, lat, E, F = compute_E_F(
                u.sel(level=p).values,
                v.sel(level=p).values,
                u[lat_name].values,
                u[lon_name].values,
            )
            E = xr.DataArray(E, coords=[lat, lon], dims=[lat_name, lon_name])
            E = E.interp_like(u, kwargs={"fill_value": "extrapolate"})
            F = xr.DataArray(F, coords=[lat, lon], dims=[lat_name, lon_name])
            F = F.interp_like(u, kwargs={"fill_value": "extrapolate"})

            if vo_file == None:
                lon, lat, vo_t = compute_vort(
                    u.sel(level=p).values,
                    v.sel(level=p).values,
                    u[lat_name].values,
                    u[lon_name].values,
                )
                vo_t = xr.DataArray(vo_t, coords=[lat, lon], dims=[lat_name, lon_name])
                vo_t = vo_t.interp_like(u, kwargs={"fill_value": "extrapolate"})
            else:
                vo_t = vo.sel(level=p)
            OWZ.append(compute_OWZ(vo_t.values, E.values, F.values, F[lat_name].values))

    OWZ = xr.DataArray(OWZ, coords=u.coords, dims=u.dims)
    OWZ = OWZ.interp_like(u, kwargs={"fill_value": "extrapolate"})

    if owz_file != None:
        # TODO : Gestion des unites
        OWZ.to_dataset(name="owz").to_netcdf(owz_file, format="NETCDF4_CLASSIC")
    return OWZ

def compute_vort(u,v,lat,lon):
    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlon = dlon * np.pi / 180.
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    dlat = dlat * np.pi / 180.
    R = 6371000  # Earth  in m
    lat_rad = lat * np.pi / 180  # Latitudes in rad

    # Compute lat and lon corresponding to the vort matrix
    lat_vort = lat[1:-1]
    lat_vort_rad = lat_vort * np.pi / 180.
    lon_vort = lon[1:-1]

    W = np.zeros([len(lat) - 2, len(lon) - 2])  # Initialization of the matrix
    ucosphi = np.transpose(np.cos(lat_rad) * np.transpose(u))
    A1 = (v[1:-1, 2:] - v[1:-1, :-2]) * 1 / (2*dlon)
    A2 = (ucosphi[2:, 1:-1] - ucosphi[:-2, 1:-1]) * 1 / (2*dlat)
    W =  A1 - A2
    D = R * np.cos(lat_vort_rad)  # Denominateur
    W = np.transpose(np.transpose(W) / D)  # Manip de multiplication matricielle
    return lon_vort, lat_vort, W

def compute_E_F(u, v, lat, lon):
    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlon = dlon * np.pi / 180.
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    dlat = dlat * np.pi / 180.
    R = 6371000  # Earth  in m
    lat_rad = lat * np.pi / 180  # Latitudes in rad

    # Compute lat and lon corresponding to the vort matrix
    lat_vort = lat[1:-1]
    lat_vort_rad = lat_vort * np.pi / 180.
    lon_vort = lon[1:-1]

    usinphi = np.transpose(np.sin(lat_rad) * np.transpose(u))
    vsinphi = np.transpose(np.sin(lat_rad) * np.transpose(v))
    ucosphi = np.transpose(np.cos(lat_rad) * np.transpose(u))
    vcosphi = np.transpose(np.cos(lat_rad) * np.transpose(v))

    E1 = (u[1:-1, 2:] - u[1:-1, :-2]) * 1 / (2 * dlon)
    E2 = (vcosphi[2:, 1:-1] - vcosphi[:-2, 1:-1]) * 1 / (2 * dlat)

    F1 = (v[1:-1, 2:] - v[1:-1, :-2]) * 1 / (2 * dlon)
    F2 = (ucosphi[2:, 1:-1] - ucosphi[:-2, 1:-1]) * 1 / (2 * dlat)

    N = E1 - E2 - vsinphi[1:-1, 1:-1]  # Numerateur
    D = R * np.cos(lat_vort_rad)  # Denominateur
    E = np.transpose(np.transpose(N) / D)  # Manip de multiplication matricielle

    N = F1 + F2 + usinphi[1:-1, 1:-1]
    D = R * np.cos(lat_vort_rad)  # Denominateur
    F = np.transpose(np.transpose(N) / D)  # Manip de multiplication matricielle

    return lon_vort, lat_vort, E, F

def compute_ObukoWeiss(vort, E, F):
    """Compute the Obuko-Weiss Parameter

    Parameters
    ----------
    vort : np.ndarray
        Vorticity field
    E : np.ndarray
        Stretching deformation field
    F : np.ndarray
        Shearing deformation field

    Returns
    -------
    np.ndarray
        The Obuko-Weiss (OW) parameter field
    """
    return vort ** 2 - (E ** 2 + F ** 2)


def compute_ObukoWeiss_norm(vort, E, F):
    """Compute the normalized Obuko-Weiss Parameter

    Parameters
    ----------
    vort : np.ndarray
        Vorticity field
    E : np.ndarray
        Stretching deformation field
    F : np.ndarray
        Shearing deformation field

    Returns
    -------
    np.ndarray
        The normalized Obuko-Weiss (OW) parameter field
    """
    OW = compute_ObukoWeiss(vort, E, F)
    return OW / vort ** 2


def compute_Coriolis_param(lat):
    """Compute the coriolis parameter for a given latitude (array)

    Parameters
    ----------
    lat : float
        latitude (can also be a np.ndarrays of latitudes)

    Returns
    -------
    float
        Coriolis parameter
    """
    W = 7.2921e-5  # Rotation rate of the Earth
    phi = lat * np.pi / 180
    return 2 * W * np.sin(phi)


def compute_OWZ(vort, E, F, lat):
    OW_n = compute_ObukoWeiss_norm(vort, E, F)
    f = compute_Coriolis_param(lat)
    return np.transpose(
        np.sign(f)
        * (np.transpose(vort) + f)
        * np.transpose(np.maximum(OW_n, np.zeros(np.shape(OW_n))))
    )

def compute_OWZ_ncl(vort, E, F):
    OW_n = compute_ObukoWeiss_norm(vort, E, F)
    OW_n_pos = OW_n.where(OW_n > 0).fillna(0)
    f = compute_Coriolis_param(vort.latitude.values)
    f = np.transpose([list(f)]*len(vort.longitude))
    vort_abs = vort + f
    A = vort_abs * OW_n_pos # Intermediary value
    return xr.where(A.latitude > 0, A, -A)

def compute_OWZ_from_files_ncl(vo_file, E_file, F_file, owz_file= None):
    vo = xr.open_dataset(vo_file).vo
    E = xr.open_dataset(E_file).E
    F = xr.open_dataset(F_file).F
    OWZ = compute_OWZ_ncl(vo, E, F).rename("owz")
    if owz_file != None:
        OWZ.to_dataset(name="owz").to_netcdf(owz_file, format="NETCDF4_CLASSIC")
    return OWZ

def compute_grad(T, lat, lon):
    """Compute the gradient of a 2D field.

    Parameters
    ----------
    T : 2D np.ndarray
        The field on which to compute the gradient
    lat : 1D np.ndarray
        latitude coordinate of the field
    lon : 1D np.ndarray
        longitude coordinate of the field

    Returns
    -------
    4 np.ndarrays
        longitude and latitude coordinates of the following x and y coordinates of the gradient.
    """
    dlon = lon[1] - lon[0]  # resolution in longitude in deg
    dlat = lat[1] - lat[0]  # resolution in latitude in deg
    R = 6371000  # Earth radius
    dy = R * dlat * np.pi / 180  #

    Gx = np.zeros([len(lat) - 1, len(lon) - 1])
    Gy = np.zeros([len(lat) - 1, len(lon) - 1])
    for i in range(len(lon) - 1):  # i index of longitude
        for j in range(len(lat) - 1):  # j index of latitude
            # Compute dx geometrically
            lat_rad = lat[j] * np.pi / 180  # Current latitude in rad
            # radius of the current longitude circle
            r = np.sin(np.pi / 2 - abs(lat_rad)) * R
            dx = r * dlon * np.pi / 180

            Gx[j, i] = (T[j + 1, i] - T[j, i]) / dx
            Gy[j, i] = (T[j, i + 1] - T[j, i]) / dy

    lat_G = np.array([(lat[j + 1] + lat[j]) / 2 for j in range(len(lat) - 1)])
    lon_G = np.array([(lon[i + 1] + lon[i]) / 2 for i in range(len(lon) - 1)])

    return lon_G, lat_G, np.array(Gx), np.array(Gy)


def compute_EKE(u, v):  # Probably deserves optimization if useful later.
    """Compute Eddy Kinetic Energy from u and v fields.

    Parameters
    ----------
    u : np.ndarray
        zonal wind field
    v : np.ndarray
        meridional wind field

    Returns
    -------
    np.ndarray
        EKE field
    """

    shape = np.shape(u)
    nlon = shape[-1]
    U_zonal = np.mean(u, -1)
    V_zonal = np.mean(v, -1)
    U_zonal_lon = np.transpose([np.transpose(U_zonal) for i in range(nlon)])
    V_zonal_lon = np.transpose([np.transpose(V_zonal) for i in range(nlon)])

    EKE = 0.5 * (
        np.multiply(u - U_zonal_lon, u - U_zonal_lon)
        + np.multiply(v - V_zonal_lon, v - V_zonal_lon)
    )

    return EKE


### ========================================== ###
###         Geographical computations          ###
### ========================================== ###


def hemispheric_mean(var, lat, axis=-1, neg=False):
    """Computes the mean over both hemispheres.

    If assymetric _data is given, the mean is computed only on the common latitudes.

    Parameters
    ----------
    var : np.ndarray
        The field that we want to average
    lat : np.ndarray
        The latitude coordinate array
    axis : int, optional
        latitude axis in var, by default -1
    neg : bool, optional
        if True, apply a minus sign to the southern hemisphere (e.g. for vorticity), by default False

    Returns
    -------
    Two np.ndarrays
        The latitude array and the hemispheric mean
    """
    lat_south, var_south = get_south(var, lat, axis, flip=True)
    lat_north, var_north = get_north(var, lat, axis)

    common_lat = [l for l in abs(lat_south) if l in lat_north]
    mask_south = [-l in common_lat for l in lat_south]
    mask_north = [l in common_lat for l in lat_north]

    var_south = apply_mask_axis(var_south, mask_south, axis)
    var_north = apply_mask_axis(var_north, mask_north, axis)

    if neg:
        sign = -1
    else:
        sign = +1

    hemispheric_mean = var_north + sign * var_south

    return common_lat, hemispheric_mean


# Aliases
hemisphericMean = hemispheric_mean

if __name__ == "__main__":
    u_file = "tests/u.nc"
    v_file = "tests/v.nc"
    vo_file = "tests/vo_ncl.nc"
    E_file = "tests/E_ncl.nc"
    F_file = "tests/F_ncl.nc"
    OWZ_before = compute_OWZ_from_files(
        u_file,
        v_file,
        level = [850,850],
    )
    print("before ok")
    OWZ_after = compute_OWZ_from_files_new(
        u_file,
        v_file,
        vo_file,
        E_file,
        F_file,
        level=[850]
    )