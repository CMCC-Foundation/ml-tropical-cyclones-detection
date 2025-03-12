import logging
import os


def retrieve_era5_single_levels(
    client, out_dir, variables, south, north, west, east, year, month, day, hour
):
    logging.info(f"\nDownloading {variables} ...")
    # create directory if not exist
    os.makedirs(out_dir, exist_ok=True)
    # determine output file
    download_file = os.path.join(
        out_dir, "_".join(variables) + f"{year}_{month:02d}_{day:02d}_{hour:02d}.nc"
    )
    # check if file has already been downloaded
    if os.path.exists(download_file):
        logging.info(f"   file already downloaded")
        return
    # download
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year": year,
            "month": month,
            "day": day,
            "time": f"{hour:02d}:00",
            "area": [north, west, south, east],
            "format": "netcdf",
        },
        download_file,
    )


def retrieve_era5_pressure_levels(
    client,
    out_dir,
    variables,
    pressure_levels,
    south,
    north,
    west,
    east,
    year,
    month,
    day,
    hour,
):
    logging.info(f"\nDownloading ['{variables} {pressure_levels} mb'] ...")
    # create directory if not exist
    os.makedirs(out_dir, exist_ok=True)
    # determine output file
    download_file = os.path.join(
        out_dir,
        "-".join(variables)
        + "_"
        + "-".join(pressure_levels)
        + f"{year}_{month:02d}_{day:02d}_{hour:02d}.nc",
    )
    # check if file has already been downloaded
    if os.path.exists(download_file):
        logging.info(f"   file already downloaded")
        return
    # download
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": variables,
            "pressure_level": pressure_levels,
            "year": year,
            "month": month,
            "day": day,
            "time": f"{hour:02d}:00",
            "area": [north, west, south, east],
        },
        download_file,
    )
