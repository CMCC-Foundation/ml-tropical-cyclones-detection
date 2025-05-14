
import os
import tempfile
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
import toml
from cdo import Cdo

class CMIP6Preprocessor:
    def __init__(self, config_path):
        self.config = toml.load(config_path)
        self.cdo = Cdo()
        self.root_dir = self.config["dir"]["root"]
        self.ua_path = self.config["files"]["ua"]
        self.va_path = self.config["files"]["va"]
        self.psl_path = self.config["files"]["psl"]
        self.output_file = os.path.join(self.root_dir, self.config["files"]["out_fname"])

        self.lat_min = self.config["coords"]["lat_min"]
        self.lat_max = self.config["coords"]["lat_max"]
        self.lon_min = self.config["coords"]["lon_min"]
        self.lon_max = self.config["coords"]["lon_max"]

    def load_cmip6_datasets(self):
        """path CMIP6 ua, va, psl"""
        ua = xr.open_dataset(os.path.join(self.root_dir, self.ua_path))
        va = xr.open_dataset(os.path.join(self.root_dir, self.va_path))
        psl = xr.open_dataset(os.path.join(self.root_dir, self.psl_path))

        return ua, va, psl

    def compute_vorticity(self, ua_850, va_850):
        """Compute vorticity 850 hPa"""
        ua_q = ua_850.ua.metpy.quantify()
        va_q = va_850.va.metpy.quantify()
        vort_q = mpcalc.vorticity(ua_q, va_q)

        return vort_q

    def build_dataset(self, ua_850, va_850, vort, psl):
        """new dataset building with ua, va, vor, psl"""
            # Creazione di una copia per evitare side-effect sull'input
        ds = ua_850.copy()

        # Aggiunta variabili al dataset in modo sicuro e leggibile
        ds["va"] = va_850["va"]
        ds["vor"] = vort if isinstance(vort, xr.DataArray) else vort["vor"]
        ds["psl"] = psl["psl"]

        return ds

    def apply_spatial_mask(self, ds):
        """crop lat and lon"""
        return ds.sel(
            lat=slice(self.lat_min, self.lat_max),
            lon=slice(self.lon_min, self.lon_max)
        )

    def save_dataset(self, ds):
        """Save final dataset"""
        ds.to_netcdf(self.output_file)
        print(f"Salvato in: {self.output_file}")

    def generate_cdo_gridfile_from_dataset(self, ds_grid, output_txt_path):
        """
        Generate grid file for CDO (gridfile .txt) regular dataset xarray.

        Args:
            ds_grid (xarray.Dataset): Dataset ERA5.
            output_txt_path (str): Path for saving griglia .txt.
        """

        lats = ds_grid.lat.values
        lons = ds_grid.lon.values

        dlat = round(abs(lats[1] - lats[0]), 6)
        dlon = round(abs(lons[1] - lons[0]), 6)

        ysize = len(lats)
        xsize = len(lons)
        yfirst = lats[0]
        xfirst = lons[0]

        with open(output_txt_path, "w") as f:
            f.write("gridtype = lonlat\n")
            f.write(f"xsize = {xsize}\n")
            f.write(f"ysize = {ysize}\n")
            f.write(f"xfirst = {xfirst}\n")
            f.write(f"xinc = {dlon}\n")
            f.write(f"yfirst = {yfirst}\n")
            f.write(f"yinc = {dlat}\n")

        print(f"[Griglia CDO] grid file saved in: {output_txt_path}")
    
    def regrid_to_era5(self, ds, era5_grid_path, output_path, method='remapcon'):
        """
        regridding from xarray.Dataset to target grid through CDO.
        """
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_input = tmp.name

        try:
            # Rimuove dimensioni non compatibili con CDO (es. plev)
            ds_cdo_safe = ds
            if "plev" in ds.dims:
                print("[Regridding] removing vertical domension 'plev' for CDO compatibility")
                ds_cdo_safe = ds.isel(plev=0).drop_vars("plev", errors="ignore")

            ds_cdo_safe.to_netcdf(tmp_input)
            print(f"[Regridding] {tmp_input} → {output_path} through grid {era5_grid_path}")

            if era5_grid_path.endswith(".txt"):
                grid_spec = era5_grid_path
            elif era5_grid_path.endswith(".nc"):
                grid_spec = f"grid={era5_grid_path}"
            else:
                raise ValueError(f"grid file fromat not supported: {era5_grid_path}")

            getattr(self.cdo, method)(
                grid_spec,
                input=tmp_input,
                output=output_path
            )

            regridded_ds = xr.open_dataset(output_path)

        except Exception as e:
            print(f"regridding during error: {e}")
            raise

        finally:
            if os.path.exists(tmp_input):
                os.remove(tmp_input)

        return regridded_ds


    def run(self):
        """Run preprocessing workflow for CMIP6."""
        ua, va, psl = self.load_cmip6_datasets()
        ua_850 = ua.sel(plev=85000)
        va_850 = va.sel(plev=85000)
        vort = self.compute_vorticity(ua_850, va_850)
        ds = self.build_dataset(ua_850, va_850, vort, psl)
        ds_masked = self.apply_spatial_mask(ds)
        
        self.generate_cdo_gridfile_from_dataset(ds_masked, self.config["regrid"]["era5_grid"])
        regrid_conf = self.config.get("regrid", {})
        if regrid_conf.get("enabled", False):
            era5_grid_path = regrid_conf["era5_grid"]
            output_regrid_path = regrid_conf.get("output_path", "masked_regridded.nc")
            ds_masked = self.regrid_to_era5(ds_masked, era5_grid_path, output_regrid_path)        

        self.save_dataset(ds_masked)