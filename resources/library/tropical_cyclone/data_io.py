from multiprocessing import Process, cpu_count
from typing import List, Any
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
import itertools
import logging
import shutil
import glob
import os

from tropical_cyclone.cyclone import get_tropical_cyclone_positions
from tropical_cyclone.patch_proc import (
    get_all_adjacent_patches,
    get_random_patches,
    get_nearest_adjacent_patches,
)


class InterTwinDatasetWriter:
    def __init__(
        self,
        src_dir: str,
        georef: pd.DataFrame,
        patch_vars: List[str],
        coo_vars: List[str],
        patch_size: int = 40,
        label_no_cyclone: float = -1.0,
        sigma: int = 10,
        grid_res: float = 0.25,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__()
        self.src_dir = src_dir
        self.georef = georef
        self.patch_size = patch_size
        self.label_no_cyclone = label_no_cyclone
        self.sigma = sigma
        self.grid_res = grid_res
        self.dtype = dtype
        self.patch_vars = patch_vars
        self.coo_vars = coo_vars

    def process_year(self, dst_dir, year, is_test: bool = False):
        logging.info(f"Processing data of year {year}")
        # define zarr files
        tc_dst = os.path.join(dst_dir, f"cyclone-{year}.zarr")
        no_tc_dst = os.path.join(dst_dir, f"no_cyclone-{year}.zarr")
        # aa_dst = os.path.join(dst_dir, f'all_adjacent-{year}.zarr')
        nr_dst = os.path.join(dst_dir, f"nearest-{year}.zarr")
        rn_dst = os.path.join(dst_dir, f"random-{year}.zarr")
        # if zarr files already exist, then skip
        if (
            os.path.exists(tc_dst)
            or os.path.exists(no_tc_dst)
            or os.path.exists(nr_dst)
            or os.path.exists(rn_dst)
        ):
            logging.info(f"   year already processed, skipping")
            return
        # define empty data
        tc_X = np.empty(
            shape=(0, len(self.patch_vars), self.patch_size, self.patch_size),
            dtype=self.dtype,
        )
        tc_Y = np.empty(shape=(0, len(self.coo_vars), 2), dtype=self.dtype)
        no_tc_X = np.empty(
            shape=(0, len(self.patch_vars), self.patch_size, self.patch_size),
            dtype=self.dtype,
        )
        no_tc_Y = np.empty(shape=(0, len(self.coo_vars), 2), dtype=self.dtype)
        # aa_X = np.empty(shape=(0, len(self.patch_vars), self.patch_size, self.patch_size), dtype=self.dtype)
        # aa_Y = np.empty(shape=(0, len(self.coo_vars), 2), dtype=self.dtype)
        nr_X = np.empty(
            shape=(0, len(self.patch_vars), self.patch_size, self.patch_size),
            dtype=self.dtype,
        )
        nr_Y = np.empty(shape=(0, len(self.coo_vars), 2), dtype=self.dtype)
        rn_X = np.empty(
            shape=(0, len(self.patch_vars), self.patch_size, self.patch_size),
            dtype=self.dtype,
        )
        rn_Y = np.empty(shape=(0, len(self.coo_vars), 2), dtype=self.dtype)
        # for each filename in src
        filepaths = sorted(glob.glob(os.path.join(self.src_dir, f"{year}*.nc")))
        for _, filepath in zip(tqdm(filepaths), filepaths):
            # open the dataset
            ds = xr.load_dataset(filepath)
            # add tropical cyclone mask
            ds = self.__create_tc_maps(ds=ds)
            # get dataset iso time
            iso_time = self.__get_iso_time(ds)
            # select iso time georef
            georef = self.georef[self.georef["ISO_TIME"] == iso_time]
            # get patched dataset
            patch_ds = ds.coarsen(
                {"lat": self.patch_size, "lon": self.patch_size}, boundary="trim"
            ).construct({"lon": ("cols", "lon_range"), "lat": ("rows", "lat_range")})
            # get dataset patches ids
            tc_ids, no_tc_ids, aa_ids, nr_ids, rn_ids = self.__get_patch_ids(ds, georef)
            # fill patch dataset with tc info
            patch_ds = self.__fill_patch_ds_with_tc_info(patch_ds, tc_ids, georef)
            # get numpy arrays of the patch dataset
            X_data = np.transpose(
                patch_ds.isel(time=0)[self.patch_vars].to_array().data,
                axes=(1, 3, 0, 2, 4),
            )  # rows x cols x 40 x 40 x vars
            Y_data = np.transpose(
                patch_ds.isel(time=0)[self.coo_vars].to_array().data, axes=(1, 2, 0, 3)
            )
            # update datasets
            tc_X, tc_Y = self.__update_dataset(tc_X, tc_Y, X_data, Y_data, tc_ids)
            if is_test == True:
                no_tc_X, no_tc_Y = self.__update_dataset(
                    no_tc_X, no_tc_Y, X_data, Y_data, no_tc_ids
                )
            # if is_test == False:
            #     aa_X, aa_Y = self.__update_dataset(aa_X, aa_Y, X_data, Y_data, aa_ids)
            if is_test == False:
                nr_X, nr_Y = self.__update_dataset(nr_X, nr_Y, X_data, Y_data, nr_ids)
            if is_test == False:
                rn_X, rn_Y = self.__update_dataset(rn_X, rn_Y, X_data, Y_data, rn_ids)
        # store datasets to disk
        self.__create_zarr(tc_X, tc_Y, tc_dst)
        self.__create_zarr(no_tc_X, no_tc_Y, no_tc_dst) if is_test == True else None
        # self.__create_zarr(aa_X, aa_Y, aa_dst) if is_test == False else None
        self.__create_zarr(nr_X, nr_Y, nr_dst) if is_test == False else None
        self.__create_zarr(rn_X, rn_Y, rn_dst) if is_test == False else None

    def __create_zarr(self, X, Y, dst):
        ds = xr.Dataset(data_vars={})
        for i, var in enumerate(self.patch_vars):
            ds[var] = (("pid", "lat_range", "lon_range"), X[:, i, :, :])
        for i, var in enumerate(self.coo_vars):
            ds[var] = (("pid", "coordinate"), Y[:, i, :])
        ds = ds.assign_coords(
            coords={
                "pid": range(0, X.shape[0]),
                "lat_range": range(0, X.shape[2]),
                "lon_range": range(0, X.shape[3]),
                "coordinate": ["y", "x"],
            }
        )
        ds.to_zarr(dst)

    def __update_dataset(
        self,
        X: np.array,
        Y: np.array,
        x_data: np.array,
        y_data: np.array,
        ids: np.array,
    ) -> [np.array, np.array]:
        # remove out of bound indices
        if ids.shape == (1, 0) or ids.shape == (0, 2) or len(ids.shape) == 1:
            return X, Y
        ids = ids[ids[:, 1] < x_data.shape[1]]
        if ids.shape == (1, 0) or ids.shape == (0, 2) or len(ids.shape) == 1:
            return X, Y
        ids = ids[ids[:, 0] < x_data.shape[0]]
        if ids.shape == (1, 0) or ids.shape == (0, 2) or len(ids.shape) == 1:
            return X, Y
        # get only patches in ids
        x = x_data[ids[:, 0], ids[:, 1]]
        y = y_data[ids[:, 0], ids[:, 1]]
        # concatenate the data to the main
        X = np.concatenate([X, x], axis=0)
        Y = np.concatenate([Y, y], axis=0)
        return X, Y

    def process_years_MPI(self, dst_dir, years, is_test: bool = False):
        for i, year in enumerate(years):
            if i % self.size != self.rank:
                continue
            self.process_year(dst_dir=dst_dir, year=year, is_test=is_test)
        return self

    def process_years_multiproc(
        self, dst_dir, years, rank, num_processes, is_test: bool = False
    ):
        for i in range(rank, len(years), num_processes):
            year = years[i]
            self.process_year(dst_dir=dst_dir, year=year, is_test=is_test)
        return self

    def merge_zarr_files(self, dst_dir: str, pathname: str):
        # create empty numpy datasets for X and Y
        X = np.empty(
            shape=(0, len(self.patch_vars), self.patch_size, self.patch_size),
            dtype=self.dtype,
        )
        Y = np.empty(shape=(0, len(self.coo_vars), 2), dtype=self.dtype)
        # get all the zarr files corresponding to the pathname
        zarr_files = sorted(glob.glob(os.path.join(dst_dir, pathname)))
        # read all the zarr files in one dataset
        for file in zarr_files:
            # open zarr file
            ds = xr.open_zarr(file)
            # get x and y data
            x = ds[self.patch_vars].to_array().load().data  # C x N x H x W
            y = ds[self.coo_vars].to_array().load().data  # C x N x 2
            # transpose the datasets to invert N and C axes
            x = np.transpose(x, axes=(1, 0, 2, 3))  # N x C x H x W
            y = np.transpose(y, axes=(1, 0, 2))  # N x C x 2
            # concatenate the datasets
            X = np.concatenate([X, x], axis=0)
            Y = np.concatenate([Y, y], axis=0)
        # define output filename
        zarr_dst = os.path.join(dst_dir, pathname.split("-")[0])
        # create zarr file
        self.__create_zarr(X, Y, zarr_dst)
        # remove redundant zarr files
        for file in zarr_files:
            shutil.rmtree(file)

    def __create_tc_maps(self, ds: xr.Dataset) -> xr.Dataset:
        ds = get_tropical_cyclone_positions(ds=ds, georef=self.georef, sigma=self.sigma)
        return ds

    def __get_iso_time(self, ds: xr.Dataset) -> str:
        timestamp = ds["time"].data[0]
        iso_time = str(pd.to_datetime(timestamp))
        return iso_time

    def __get_patch_ids(self, ds: xr.Dataset, georef: pd.DataFrame):
        # get lats and lons from dataset
        lats, lons = ds["lat"].data, ds["lon"].data
        i_bound = len(lats) // self.patch_size
        j_bound = len(lons) // self.patch_size
        # get all patch ids
        all_patch_ids = [
            coords
            for coords in list(
                itertools.product(
                    [i for i in range(i_bound)], [i for i in range(j_bound)]
                )
            )
        ]
        # get tropical cyclone coordinates as list of tuples
        tc_ids = (georef[["YLAT", "XLON"]] // self.patch_size).to_numpy()
        # get list of ids not accounting a tc
        no_tc_ids = sorted(
            list(set(list(all_patch_ids)).difference(set(list(map(tuple, tc_ids)))))
        )
        # get intra-patch cyclone positions
        patch_cyclone_positions = (
            georef[["YLAT", "XLON"]] % self.patch_size
        ).to_numpy()
        # get all other ids aa_ids, rn_ids, nr_ids
        aa_ids = get_all_adjacent_patches(dataset=ds, patch_cyclone_ids=tc_ids)
        rn_ids = get_random_patches(dataset=ds, patch_cyclone_ids=tc_ids)
        nr_ids = get_nearest_adjacent_patches(
            dataset=ds,
            patch_cyclone_ids=tc_ids,
            patch_cyclone_positions=patch_cyclone_positions,
        )
        return (
            np.array(tc_ids),
            np.asarray(no_tc_ids),
            np.array(list(aa_ids)),
            np.array(list(nr_ids)),
            np.array(list(rn_ids)),
        )

    def __fill_patch_ds_with_tc_info(
        self, patch_ds: xr.Dataset, tc_ids: List, georef: pd.DataFrame
    ):
        # get from georef dataframe
        pcy_rowcol = (georef[["YLAT", "XLON"]] % self.patch_size).to_numpy()
        rcy_coords = georef[["RLAT", "RLON"]].to_numpy()
        gcy_rowcol = georef[["YLAT", "XLON"]].to_numpy()
        cy_coords = georef[["LAT", "LON"]].to_numpy()
        # create variable to store cyclone coordinates
        real_cyclone_data = np.full(
            shape=(
                patch_ds.time.data.shape[0],
                patch_ds.rows.data.shape[0],
                patch_ds.cols.data.shape[0],
                2,
            ),
            fill_value=self.label_no_cyclone,
            dtype=self.dtype,
        )
        rounded_cyclone_data = np.full(
            shape=(
                patch_ds.time.data.shape[0],
                patch_ds.rows.data.shape[0],
                patch_ds.cols.data.shape[0],
                2,
            ),
            fill_value=self.label_no_cyclone,
            dtype=self.dtype,
        )
        global_cyclone_data = np.full(
            shape=(
                patch_ds.time.data.shape[0],
                patch_ds.rows.data.shape[0],
                patch_ds.cols.data.shape[0],
                2,
            ),
            fill_value=self.label_no_cyclone,
            dtype=self.dtype,
        )
        patch_cyclone_data = np.full(
            shape=(
                patch_ds.time.data.shape[0],
                patch_ds.rows.data.shape[0],
                patch_ds.cols.data.shape[0],
                2,
            ),
            fill_value=self.label_no_cyclone,
            dtype=self.dtype,
        )
        # for each rowcol coordinate
        for tc_id, pt_rowcol, rcy_coo, gl_rowcol, cy_coo in zip(
            tc_ids, pcy_rowcol, rcy_coords, gcy_rowcol, cy_coords
        ):
            i, j = tc_id
            # check if the id is out of bound
            if i == patch_ds.rows.data.shape[0] or j == patch_ds.cols.data.shape[0]:
                continue
            # assign the cyclone to the variable
            patch_cyclone_data[0, i, j, :] = pt_rowcol
            rounded_cyclone_data[0, i, j, :] = rcy_coo
            real_cyclone_data[0, i, j, :] = cy_coo
            global_cyclone_data[0, i, j, :] = gl_rowcol
        # create patch_cyclone variable into the patch_ds dataset
        patch_ds = patch_ds.assign(
            {
                "real_cyclone": (
                    ("time", "rows", "cols", "coordinate"),
                    real_cyclone_data,
                ),
                "rounded_cyclone": (
                    ("time", "rows", "cols", "coordinate"),
                    rounded_cyclone_data,
                ),
                "global_cyclone": (
                    ("time", "rows", "cols", "coordinate"),
                    global_cyclone_data,
                ),
                "patch_cyclone": (
                    ("time", "rows", "cols", "coordinate"),
                    patch_cyclone_data,
                ),
            }
        )
        return patch_ds

    # def process_year(self,
    #         dst_dir,
    #         year,
    #         is_test: bool=False
    #     ):
    #     logging.info(f'Processing data of year {year}')
    #     # define zarr files
    #     tc_dst = os.path.join(dst_dir, f'{year}-cyclone.zarr')
    #     no_tc_dst = os.path.join(dst_dir, f'{year}-no_cyclone.zarr')
    #     aa_dst = os.path.join(dst_dir, f'{year}-all_adjacent.zarr')
    #     nr_dst = os.path.join(dst_dir, f'{year}-nearest.zarr')
    #     rn_dst = os.path.join(dst_dir, f'{year}-random.zarr')
    #     # if zarr files already exist, then skip
    #     if os.path.exists(tc_dst) or os.path.exists(no_tc_dst) or os.path.exists(aa_dst) or os.path.exists(nr_dst) or os.path.exists(rn_dst):
    #         logging.info(f'   year already processed, skipping')
    #         return
    #     # for each filename in src
    #     filepaths = sorted(glob.glob(os.path.join(self.src_dir, f'{year}*.nc')))
    #     # for each file in the dataset
    #     for _, filepath in zip(tqdm(filepaths), filepaths):
    #         # open the dataset
    #         ds = xr.load_dataset(filepath)
    #         # add tropical cyclone mask
    #         ds = self.__create_tc_maps(ds=ds)
    #         # get dataset iso time
    #         iso_time = self.__get_iso_time(ds)
    #         # select iso time georef
    #         georef = self.georef[self.georef['ISO_TIME'] == iso_time]
    #         # get patched dataset
    #         patch_ds = ds.coarsen({'lat':self.patch_size, 'lon':self.patch_size}, boundary="trim").construct({'lon':("cols", "lon_range"), 'lat':("rows", "lat_range")})
    #         # get dataset patches ids
    #         tc_ids, no_tc_ids, aa_ids, nr_ids, rn_ids = self.__get_patch_ids(ds, georef)
    #         # fill patch dataset with tc info
    #         patch_ds = self.__fill_patch_ds_with_tc_info(patch_ds, tc_ids, georef)
    #         # store datasets to disk
    #         self.__store_netcdf(dst_dir, patch_ds, tc_ids, 'cyclone')
    #         self.__store_netcdf(dst_dir, patch_ds, no_tc_ids, 'no_cyclone') if is_test == True else None
    #         self.__store_netcdf(dst_dir, patch_ds, aa_ids, 'alladjacent') if is_test == False else None
    #         self.__store_netcdf(dst_dir, patch_ds, nr_ids, 'nearest') if is_test == False else None
    #         self.__store_netcdf(dst_dir, patch_ds, rn_ids, 'random') if is_test == False else None

    # def __store_netcdf(self,
    #         dst: str,
    #         patch_ds: xr.Dataset,
    #         ids: List,
    #         patch_type: str,
    #     ) -> int:
    #     # for each id
    #     for i,id in enumerate(ids):
    #         # check if the id is out of bound
    #         if id[0] == patch_ds.rows.data.shape[0] or id[1] == patch_ds.cols.data.shape[0]:
    #             continue
    #         t = pd.to_datetime(patch_ds.time.data[0])
    #         # define destination file
    #         dst_file = os.path.join(dst, f'{patch_type}_{t.year}{t.month:02d}{t.day:02d}{t.hour:02d}_{i:02d}.nc')
    #         # select the TC from the patch ds
    #         patch = patch_ds.isel(time=0, rows=id[0], cols=id[1])
    #         # drop the coordinates
    #         patch = patch.drop(('time', 'lat', 'lon'))
    #         # store patch netcdf to disk
    #         patch.to_netcdf(dst_file)
    #         # increase the counter


class InterTwinMPIDatasetWriter(InterTwinDatasetWriter):
    def __init__(
        self,
        src_dir: str,
        georef: pd.DataFrame,
        patch_vars: List[str],
        coo_vars: List[str],
        patch_size: int = 40,
        label_no_cyclone: float = -1,
        sigma: int = 10,
        grid_res: float = 0.25,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__(
            src_dir,
            georef,
            patch_vars,
            coo_vars,
            patch_size,
            label_no_cyclone,
            sigma,
            grid_res,
            dtype,
        )
        from mpi4py import MPI

        # initialize MPI
        self.comm = MPI.COMM_WORLD

        # get rank and world size of the process
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def process(self, dst_dir, years, is_test: bool = False):
        self.process_years_MPI(dst_dir=dst_dir, years=years, is_test=is_test)


class InterTwinMultiprocDatasetWriter(InterTwinDatasetWriter):
    def __init__(
        self,
        src_dir: str,
        georef: pd.DataFrame,
        patch_vars: List[str],
        coo_vars: List[str],
        patch_size: int = 40,
        label_no_cyclone: float = -1,
        sigma: int = 10,
        grid_res: float = 0.25,
        dtype: Any = np.float32,
    ) -> None:
        super().__init__(
            src_dir,
            georef,
            patch_vars,
            coo_vars,
            patch_size,
            label_no_cyclone,
            sigma,
            grid_res,
            dtype,
        )

    def process(self, dst_dir, years, is_test: bool = False):
        # Get as many processes as possible
        num_processes = cpu_count()
        processes = []

        # Allocate processes
        for rank in range(num_processes):
            process = Process(
                target=self.process_years_multiproc,
                args=(dst_dir, years, rank, num_processes, is_test),
            )
            processes.append(process)

        # Start all processes
        for process in processes:
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()

        return self
