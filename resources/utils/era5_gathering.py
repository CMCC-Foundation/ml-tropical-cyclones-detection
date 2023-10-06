from netCDF4 import Dataset, date2num
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import shutil
import cdsapi
from datetime import timedelta
import yaml


def retrieve_era5_single_levels(client, out_dir, variables, south, north, west, east, year, month, day, time):
    print('\nDownloading {} ...'.format(variables))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if (len(variables) > 1):
        download_file = out_dir + '/' + 'single_levels_vars.' + year + '_' + month + '_' + day + '_' + time + '.nc'
    else:
        download_file = out_dir + '/' + '10m_wind_gust_since_previous_post_processing.' + year + '_' + month + '_' + day + '_' + time + '.nc'
        
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': year,
            'month': month,
            'day': day,
            'time': time + ':00',
            'area': [
                north, west, south,
                east,
            ],
            'format': 'netcdf',
        },
        download_file)


def retrieve_era5_pressure_levels(client, out_dir, variable, pressure_level, south, north, west, east, year, month, day, time):
    print('\nDownloading [\'{} {} mb\'] ...'.format(variable, pressure_level))
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': variable,
            'pressure_level': pressure_level,
            'year': year,
            'month': month,
            'day': day,
            'time': time + ':00',
            'area': [
                north, west, south,
                east,
            ],
        },
        out_dir + '/' + variable + '_' + pressure_level + '.' + year + '_' + month + '_' + day + '_' + time + '.nc')


restart_idx = 0
end_idx = 100000

# define CDS api client
url = 'url'
key = 'key'
home_dir = os.path.expanduser("~")
path_to_cdsapirc_file = os.path.join(home_dir, '.cdsapirc')
with open(path_to_cdsapirc_file, 'r') as f:
		credentials = yaml.safe_load(f)
client = cdsapi.Client(url=credentials[url], key=credentials[key])

path_ibtracs_1980_2020 = 'ibtracs_ALL.1980-2020_date_fields.csv'
path_ERA5 = '../data/ERA5/'
path_maps = '../data/ERA5/Maps/'
path_tmp = '../data/ERA5/Maps/tmp/'

if not os.path.isdir(path_ERA5):
    os.makedirs(path_ERA5)

if not os.path.isdir(path_maps):
    os.makedirs(path_maps)

if not os.path.isdir(path_tmp):
    os.makedirs(path_tmp)

columns_date = ['SID','SEASON','NUMBER','BASIN',
           'SUBBASIN','NAME','ISO_TIME', 
           'Year', 'Month', 'Day', 'Time',
           'NATURE', 'LAT','LON','WMO_WIND','WMO_PRES',
           'TRACK_TYPE','DIST2LAND','LANDFALL',
           'USA_WIND', 'USA_PRES',
           'STORM_SPEED','STORM_DIR']


ibtracs_1980_2020 = pd.read_csv(path_ibtracs_1980_2020, dtype='str')
ibtracs_1980_2020_date_df = pd.DataFrame(data=ibtracs_1980_2020, columns=['ISO_TIME'])

# Delete duplicate values
ibtracs_1980_2020_date_unique = ibtracs_1980_2020_date_df["ISO_TIME"].unique()
ibtracs_1980_2020_date_unique_df = pd.DataFrame(data=ibtracs_1980_2020_date_unique, columns=['ISO_TIME'])

# Sort by date
ibtracs_1980_2020_date_unique_df["ISO_TIME"] = pd.to_datetime(ibtracs_1980_2020_date_unique_df["ISO_TIME"])
ibtracs_1980_2020_date_unique_df = ibtracs_1980_2020_date_unique_df.sort_values(by="ISO_TIME")

# Reset index
ibtracs_1980_2020_date_unique_df = ibtracs_1980_2020_date_unique_df.reset_index(inplace=False, drop=True)

############################################# WORKFLOW #############################################
variables_single_levels = ['instantaneous_10m_wind_gust',
                           'mean_sea_level_pressure',
                           'sea_surface_temperature']
wind_post_processing = ['10m_wind_gust_since_previous_post_processing']

south = 0 # °N
north = 70 # °N
west = 100 # °E
east = -40 # °E

iso_time_previous = ''

wind_post_processing_download_failed = False
record_download_failed_file = open("record_download_failed.txt", "w")
# Closes the file to save in memory the deletion just made of the old content.
# The file will only be reopened when a download fails.
# In the meantime, it would make no sense to leave the file open without using it for writes or reads.
record_download_failed_file.close() 


for idx, record in ibtracs_1980_2020_date_unique_df.iterrows():

    if (idx < restart_idx):
        continue
    
    if (idx > end_idx):
        break

    print('\n#####################################################################################')
    print('Processing idx: {}'.format(idx))
    
    iso_time_current = record['ISO_TIME']

    year_current = iso_time_current.strftime('%Y')
    month_current = iso_time_current.strftime('%m')
    day_current = iso_time_current.strftime('%d')
    time_current = iso_time_current.strftime('%H')
    
    print('Processing date: {}-{}-{} {}:00'.format(year_current, month_current, day_current, time_current))
    
    out_file = path_maps + year_current + '_' + month_current + '_' + day_current + '_' + time_current
    
    if ((iso_time_previous == '') or ((iso_time_current-iso_time_previous) > timedelta(hours=3))):
        shutil.rmtree(path_tmp, ignore_errors=True)
        print('FOLDER DELETED - Processing date: {}-{}-{} {}:00 - Processing idx: {}'.format(year_current, month_current, day_current, time_current, idx))
        end = 6
    else:
        end = 3
        
    for i in range(0,end):
        try:
            date_i_h = iso_time_current - timedelta(hours=i)
            retrieve_era5_single_levels(client, path_tmp, wind_post_processing, south, north, west, east, date_i_h.strftime('%Y'), date_i_h.strftime('%m'), date_i_h.strftime('%d'), date_i_h.strftime('%H'))
        except:
            print('FAILED DOWNLOAD — 10 m wind gust since previous post-processing at {}:00 — Index: {}, {}-{}-{} {}:00'.format(date_i_h.strftime('%H'), idx, date_i_h.strftime('%Y'), date_i_h.strftime('%m'), date_i_h.strftime('%d'), date_i_h.strftime('%H')))
            shutil.rmtree(path_tmp, ignore_errors=True)
            wind_post_processing_download_failed = True
            break
    
    if (wind_post_processing_download_failed):
        record_download_failed_file = open("record_download_failed.txt", "a")  
        record_download_failed_file.write('{}\n'.format(idx))
        record_download_failed_file.close()  
        wind_post_processing_download_failed = False
        continue
            

    try:
        retrieve_era5_single_levels(client, path_tmp, variables_single_levels, south, north, west, east, year_current, month_current, day_current, time_current)
    except:
        print('FAILED DOWNLOAD — Single levels variables — Index: {}, {}-{}-{} {}:00'.format(idx, year_current, month_current, day_current, time_current))
        shutil.rmtree(path_tmp, ignore_errors=True)
        record_download_failed_file = open("record_download_failed.txt", "a")  
        record_download_failed_file.write('{}\n'.format(idx))
        record_download_failed_file.close()  
        continue
    
    try:
        retrieve_era5_pressure_levels(client, path_tmp, 'vorticity', '850', south, north, west, east, year_current, month_current, day_current, time_current)
    except:
        print('FAILED DOWNLOAD — Vorticity at 850 mb — Index: {}, {}-{}-{} {}:00'.format(idx, year_current, month_current, day_current, time_current))
        shutil.rmtree(path_tmp, ignore_errors=True)
        record_download_failed_file = open("record_download_failed.txt", "a")   
        record_download_failed_file.write('{}\n'.format(idx))
        record_download_failed_file.close()  
        continue
    
    try:
        retrieve_era5_pressure_levels(client, path_tmp, 'temperature', '300', south, north, west, east, year_current, month_current, day_current, time_current)
    except:
        print('FAILED DOWNLOAD — Temperature at 300 mb — Index: {}, {}-{}-{} {}:00'.format(idx, year_current, month_current, day_current, time_current))
        shutil.rmtree(path_tmp, ignore_errors=True)
        record_download_failed_file = open("record_download_failed.txt", "a")  
        record_download_failed_file.write('{}\n'.format(idx))
        record_download_failed_file.close()  
        continue
    
    try:
        retrieve_era5_pressure_levels(client, path_tmp, 'temperature', '500', south, north, west, east, year_current, month_current, day_current, time_current)
    except:
        print('FAILED DOWNLOAD — Temperature at 500 mb — Index: {}, {}-{}-{} {}:00'.format(idx, year_current, month_current, day_current, time_current))
        shutil.rmtree(path_tmp, ignore_errors=True)
        record_download_failed_file = open("record_download_failed.txt", "a")  
        record_download_failed_file.write('{}\n'.format(idx))
        record_download_failed_file.close()  
        continue
    
    
    wind_post_processing_maps_list = []

    for i in range(0,end):
        date_tmi = iso_time_current - timedelta(hours=i)
        path_wind_post_processing_tmi = path_tmp + '10m_wind_gust_since_previous_post_processing.' + date_tmi.strftime('%Y') + '_' + date_tmi.strftime('%m') + '_' + date_tmi.strftime('%d') + '_' + date_tmi.strftime('%H') + '.nc'
        wind_post_processing_tmi_data = Dataset(path_wind_post_processing_tmi, mode='r+', format='NETCDF3')
        if (i == 0):
            wind_post_processing_6h_missing_value = wind_post_processing_tmi_data['fg10'].missing_value
            wind_post_processing_6h_scale_factor = wind_post_processing_tmi_data['fg10'].scale_factor
            wind_post_processing_6h_add_offset = wind_post_processing_tmi_data['fg10'].add_offset
        # wind_post_processing_tmi_data.set_auto_mask(True)
        wind_post_processing_maps_list.append(wind_post_processing_tmi_data['fg10'][0,:,:])
        wind_post_processing_tmi_data.close()

    wind_post_processing_6h = ma.max(wind_post_processing_maps_list, axis=0)


    path_single_levels_variables = path_tmp + 'single_levels_vars.' + year_current + '_' + month_current + '_' + day_current + '_' + time_current + '.nc'
    single_levels_variables_data = Dataset(path_single_levels_variables, mode='r+', format='NETCDF3')
    lats = np.array(single_levels_variables_data['latitude'][:])
    lons = np.array(single_levels_variables_data['longitude'][:])

    instantaneous_wind = single_levels_variables_data['i10fg'][0,:,:]
    instantaneous_wind_missing_value = single_levels_variables_data['i10fg'].missing_value
    instantaneous_wind_scale_factor = single_levels_variables_data['i10fg'].scale_factor
    instantaneous_wind_add_offset = single_levels_variables_data['i10fg'].add_offset

    mean_sea_level_pressure = single_levels_variables_data['msl'][0,:,:]
    mean_sea_level_pressure_missing_value = single_levels_variables_data['msl'].missing_value
    mean_sea_level_pressure_scale_factor = single_levels_variables_data['msl'].scale_factor
    mean_sea_level_pressure_add_offset = single_levels_variables_data['msl'].add_offset

    sea_surface_temperature = single_levels_variables_data['sst'][0,:,:]
    sea_surface_temperature_missing_value = single_levels_variables_data['sst'].missing_value
    sea_surface_temperature_scale_factor = single_levels_variables_data['sst'].scale_factor
    sea_surface_temperature_add_offset = single_levels_variables_data['sst'].add_offset

    single_levels_variables_data.close()


    path_temperature_500 = path_tmp + 'temperature_500.' + year_current + '_' + month_current + '_' + day_current + '_' + time_current + '.nc'
    temperature_500_data = Dataset(path_temperature_500, mode='r+', format='NETCDF3')
    temperature_500 = temperature_500_data['t'][0,:,:]
    temperature_500_missing_value = temperature_500_data['t'].missing_value
    temperature_500_add_offset = temperature_500_data['t'].add_offset
    temperature_500_scale_factor = temperature_500_data['t'].scale_factor
    temperature_500_data.close()

    path_temperature_300 = path_tmp + 'temperature_300.' + year_current + '_' + month_current + '_' + day_current + '_' + time_current + '.nc'
    temperature_300_data = Dataset(path_temperature_300, mode='r+', format='NETCDF3')
    temperature_300 = temperature_300_data['t'][0,:,:]
    temperature_300_missing_value = temperature_300_data['t'].missing_value 
    temperature_300_scale_factor = temperature_300_data['t'].scale_factor
    temperature_300_add_offset = temperature_300_data['t'].add_offset
    temperature_300_data.close()

    path_vorticity_850 = path_tmp + 'vorticity_850.' + year_current + '_' + month_current + '_' + day_current + '_' + time_current + '.nc'
    vorticity_850_data = Dataset(path_vorticity_850, mode='r+', format='NETCDF3')
    vorticity_850 = vorticity_850_data['vo'][0,:,:]
    vorticity_850_missing_value = vorticity_850_data['vo'].missing_value
    vorticity_850_scale_factor = vorticity_850_data['vo'].scale_factor
    vorticity_850_add_offset = vorticity_850_data['vo'].add_offset
    vorticity_850_data.close()


    out_netcdf = Dataset(out_file + '.nc', mode='w', format='NETCDF3_CLASSIC')

    lat_dim = out_netcdf.createDimension('lat', 281)     # latitude axis
    lon_dim = out_netcdf.createDimension('lon', 881)    # longitude axis
    time_dim = out_netcdf.createDimension('time', None)

    lat = out_netcdf.createVariable('lat', np.float64, ('lat',))
    lat.units = 'degrees_north'
    lat.long_name = 'latitude'
    lon = out_netcdf.createVariable('lon', np.float64, ('lon',))
    lon.units = 'degrees_east'
    lon.long_name = 'longitude'
    time = out_netcdf.createVariable('time', np.float64, ('time',))
    time.units = 'hours since 1900-01-01 00:00:00.0'
    time.long_name = 'time'

    wind_post_processing_6h_var = out_netcdf.createVariable('fg10', np.float32, ('time','lat','lon')) # note: unlimited dimension is leftmost
    wind_post_processing_6h_var.units = 'm s**-1'
    wind_post_processing_6h_var.missing_value = wind_post_processing_6h_missing_value
    # wind_post_processing_6h_var.scale_factor = wind_post_processing_6h_scale_factor
    # wind_post_processing_6h_var.add_offset = wind_post_processing_6h_add_offset
    wind_post_processing_6h_var.long_name = '10 metre wind gust since previous post-processing (6h) [m s**-1]'
    wind_post_processing_6h_var.standard_name = '10m_wind_gust_since_previous_post_processing'

    instantaneous_wind_var = out_netcdf.createVariable('i10fg', np.int16, ('time','lat','lon'), fill_value=instantaneous_wind_missing_value) # note: unlimited dimension is leftmost
    instantaneous_wind_var.missing_value = instantaneous_wind_missing_value
    instantaneous_wind_var.scale_factor = instantaneous_wind_scale_factor
    instantaneous_wind_var.add_offset = instantaneous_wind_add_offset
    instantaneous_wind_var.units = 'm s**-1'
    instantaneous_wind_var.long_name = 'Instantaneous 10 metre wind gust [m s**-1]'
    instantaneous_wind_var.standard_name = 'instantaneous_10m_wind_gust'

    mean_sea_level_pressure_var = out_netcdf.createVariable('msl', np.int16, ('time','lat','lon'), fill_value=mean_sea_level_pressure_missing_value) # note: unlimited dimension is leftmost
    mean_sea_level_pressure_var.missing_value = mean_sea_level_pressure_missing_value
    mean_sea_level_pressure_var.scale_factor = mean_sea_level_pressure_scale_factor
    mean_sea_level_pressure_var.add_offset = mean_sea_level_pressure_add_offset
    mean_sea_level_pressure_var.units = 'Pa'
    mean_sea_level_pressure_var.long_name = 'Mean sea level pressure [Pa]'
    mean_sea_level_pressure_var.standard_name = 'air_pressure_at_mean_sea_level'

    sea_surface_temperature_var = out_netcdf.createVariable('sst', np.int16, ('time','lat','lon'), fill_value=sea_surface_temperature_missing_value) # note: unlimited dimension is leftmost
    sea_surface_temperature_var.missing_value = sea_surface_temperature_missing_value
    sea_surface_temperature_var.scale_factor = sea_surface_temperature_scale_factor
    sea_surface_temperature_var.add_offset = sea_surface_temperature_add_offset
    sea_surface_temperature_var.units = 'K'
    sea_surface_temperature_var.long_name = 'Sea surface temperature [K]'
    sea_surface_temperature_var.standard_name = 'sea_surface_temperature'

    temperature_500_var = out_netcdf.createVariable('t_500', np.int16, ('time','lat','lon'), fill_value=temperature_500_missing_value) # note: unlimited dimension is leftmost
    temperature_500_var.missing_value = temperature_500_missing_value
    temperature_500_var.scale_factor = temperature_500_scale_factor
    temperature_500_var.add_offset = temperature_500_add_offset
    temperature_500_var.units = 'K'
    temperature_500_var.long_name = 'Temperature at 500 mb [K]'
    temperature_500_var.standard_name = 'temperature_500'

    temperature_300_var = out_netcdf.createVariable('t_300', np.int16, ('time','lat','lon'), fill_value=temperature_300_missing_value) # note: unlimited dimension is leftmost
    temperature_300_var.missing_value = temperature_300_missing_value
    temperature_300_var.scale_factor = temperature_300_scale_factor
    temperature_300_var.add_offset = temperature_300_add_offset
    temperature_300_var.units = 'K'
    temperature_300_var.long_name = 'Temperature at 300 mb [K]'
    temperature_300_var.standard_name = 'temperature_300'

    vorticity_850_var = out_netcdf.createVariable('vo_850', np.int16, ('time','lat','lon'), fill_value=vorticity_850_missing_value) # note: unlimited dimension is leftmost
    vorticity_850_var.missing_value = vorticity_850_missing_value
    vorticity_850_var.scale_factor = vorticity_850_scale_factor
    vorticity_850_var.add_offset = vorticity_850_add_offset
    vorticity_850_var.units = 's**-1'
    vorticity_850_var.long_name = 'Vorticity (relative) at 850 mb [s**-1]'
    vorticity_850_var.standard_name = 'atmosphere_relative_vorticity_850'

    out_netcdf.variables['fg10'][0,:,:] = wind_post_processing_6h[:,:]
    out_netcdf.variables['i10fg'][0,:,:] = instantaneous_wind[:,:]
    out_netcdf.variables['msl'][0,:,:] = mean_sea_level_pressure[:,:]
    out_netcdf.variables['sst'][0,:,:] = sea_surface_temperature[:,:]
    out_netcdf.variables['t_500'][0,:,:] = temperature_500[:,:]
    out_netcdf.variables['t_300'][0,:,:] = temperature_300[:,:]
    out_netcdf.variables['vo_850'][0,:,:] = vorticity_850[:,:]
    out_netcdf.variables['lat'][:] = lats[:]
    out_netcdf.variables['lon'][:] = lons[:]
    out_netcdf.variables['time'][:] = date2num(iso_time_current, time.units)

    out_netcdf.close()
    
    ## REMAPCON
	# cdo.remapcon('./CMCC-CM3_grid', input=out_file + '.nc', output=out_file + '_remapcon.nc')
    
    
    ## REMOVE OLD FILES
    remove_file = (path_tmp + 'vorticity_850.' + year_current + '_' + month_current + 
                   '_' + day_current + '_' + time_current + '.nc')
    if os.path.exists(remove_file):
        os.remove(remove_file)
    
    remove_file = (path_tmp + 'temperature_300.' + year_current + '_' + month_current + 
                   '_' + day_current + '_' + time_current + '.nc')
    if os.path.exists(remove_file):
        os.remove(remove_file)
    
    remove_file = (path_tmp + 'temperature_500.' + year_current + '_' + month_current + 
                   '_' + day_current + '_' + time_current + '.nc')
    if os.path.exists(remove_file):
        os.remove(remove_file)
        
    remove_file = (path_tmp + 'single_levels_vars.' + year_current + '_' + month_current + 
                   '_' + day_current + '_' + time_current + '.nc')
    if os.path.exists(remove_file):
        os.remove(remove_file)
        
    for i in range(3,6):
        date_tmi = iso_time_current - timedelta(hours=i)
        remove_file = (path_tmp + '10m_wind_gust_since_previous_post_processing.' + 
                       date_tmi.strftime('%Y') + '_' + date_tmi.strftime('%m') + 
                       '_' + date_tmi.strftime('%d') + '_' + date_tmi.strftime('%H') + '.nc')
        if os.path.exists(remove_file):
            os.remove(remove_file)
    
    print('\OLD FILES DELETED')
    
    iso_time_previous = iso_time_current

shutil.rmtree(path_tmp, ignore_errors=True)