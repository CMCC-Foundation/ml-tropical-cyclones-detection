import xarray as xr
import pandas as pd
import argparse
import cdsapi
import shutil
import munch
import toml
import os

import sys

sys.path.append("../resources/library")
import tropical_cyclone as tc

rank = 0

# define download variables
variables_single_level = [
    "instantaneous_10m_wind_gust",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
]
wind_post_processing = ["10m_wind_gust_since_previous_post_processing"]
vorticity_variable = ["vorticity"]
vorticity_pressure_level = ["850"]
temperature_variable = ["temperature"]
temperature_pressure_levels = ["300", "500"]

# domain extent
south = 0  # °N
north = 70  # °N
west = 100  # °E
east = -40  # °E

# main directories setup
dataset_dir = f"../data/dataset"
dataset_tmp_data_dir = f"../data/tmp_data_{rank}"
dataset_tmp_wind_gust_dir = f"../data/tmp_wind_gust_{rank}"

# select only these ibtracs columns
columns = [
    "SID",
    "SEASON",
    "NUMBER",
    "BASIN",
    "SUBBASIN",
    "NAME",
    "ISO_TIME",
    "NATURE",
    "LAT",
    "LON",
    "WMO_WIND",
    "WMO_PRES",
    "TRACK_TYPE",
    "DIST2LAND",
    "LANDFALL",
    "USA_WIND",
    "USA_PRES",
    "STORM_SPEED",
    "STORM_DIR",
]

# open ibtracs dataset
ibtracs_src = (
    "../data/ibtracs/filtered/ibtracs_main-tracks_6h_1980-2021_TS-NR-ET-MX-SS-DS.csv"
)
ibtracs = pd.read_csv(
    ibtracs_src, usecols=columns, header=0, index_col=0, keep_default_na=False
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Program Start
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# init CDS-API Client
client = cdsapi.Client()

# create folders if not exist
if rank == 0:
    os.makedirs(dataset_dir, exist_ok=True)

# remove temporary directory files
shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
shutil.rmtree(dataset_tmp_wind_gust_dir, ignore_errors=True)

# create temporary directory
if rank == 0:
    os.makedirs(dataset_tmp_data_dir, exist_ok=True)
    os.makedirs(dataset_tmp_wind_gust_dir, exist_ok=True)

# barrier

dates = pd.to_datetime(sorted(ibtracs["ISO_TIME"].unique()))

# iterate over each iso time
for date in dates:
    year = date.year
    month = date.month
    day = date.day
    hour = date.hour

    print(
        f"Downloading ERA5 variable maps for iso time {year}-{month:02d}-{day:02d} {hour:02d}:00"
    )

    # define output filename
    out_fname = os.path.join(dataset_dir, f"{year}_{month:02d}_{day:02d}_{hour:02d}.nc")

    # skip if output filename already exists
    if os.path.exists(out_fname):
        print(f"File {out_fname} already exists. Skipping")
        continue

    # download wind gust since previous post-processing
    for i in range(6):
        prev_date = date - pd.DateOffset(hours=i)
        tc.era5.retrieve_era5_single_levels(
            client=client,
            out_dir=dataset_tmp_wind_gust_dir,
            variables=wind_post_processing,
            south=south,
            north=north,
            west=west,
            east=east,
            year=year,
            month=month,
            day=day,
            hour=hour,
        )

    break
