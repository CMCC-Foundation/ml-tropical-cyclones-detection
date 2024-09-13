# import warnings
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)



def plot_detections(detections, observations, lat_range=(0,70), lon_range=(100,320), outfile=None):
	# set map extent
	central_longitude = (lon_range[1] - lon_range[0])

	_ = plt.figure(figsize=(25,10))
	proj = ccrs.PlateCarree(central_longitude=central_longitude)
	ax = plt.axes(projection=proj)

	image_extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
	ax.set_extent(image_extent, crs=ccrs.PlateCarree())
	ax.coastlines(resolution='50m', lw=0.2)
	ax.add_feature(cf.LAND, facecolor='lightgrey', alpha=0.3)

	fontdict = {'weight':'bold', 'size':14}
	# title_fontdict = {'size':18}
	ticksize = 12

	# plot tracks in each basin
	marker_size = 10.0

	ax.scatter(detections['LON'], detections['LAT'], s=marker_size, marker='o', alpha=1.0, transform=ccrs.Geodetic(), label=f'TC Detections (#{len(detections)})')
	ax.scatter(observations['LON'], observations['LAT'], s=marker_size, marker='o', alpha=1.0, transform=ccrs.Geodetic(), label=f'TC Observations (#{len(observations)})')

	# x-axis
	longitudes = np.arange(lon_range[0], lon_range[1]+1, 10)
	lon_formatter = LongitudeFormatter(zero_direction_label=False)
	ax.xaxis.set_major_formatter(lon_formatter)
	ax.xaxis.set_major_locator(mticker.FixedLocator(longitudes-central_longitude))
	ax.set_xticklabels(longitudes, size=ticksize)
	ax.set_xticks(longitudes-central_longitude)
	ax.set_xlabel('Longitude [deg]', fontdict=fontdict)

	# y-axis
	latitudes = np.arange(lat_range[0], lat_range[1]+1, 10)
	lat_formatter = LatitudeFormatter()
	ax.yaxis.set_major_formatter(lat_formatter)
	ax.yaxis.set_major_locator(mticker.FixedLocator(latitudes))
	ax.set_yticklabels(latitudes, size=ticksize)
	ax.set_yticks(latitudes)
	ax.set_ylabel('Latitude [deg]', fontdict=fontdict)

	# ax.set_title(f"TSTORMS and ML inference on CMCC-CM3 (#{len(df)} matches)", fontdict=title_fontdict)

	# gridlines
	gl = ax.gridlines(crs=proj, draw_labels=False, dms=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
	gl.xlocator = mticker.FixedLocator(longitudes-central_longitude)
	gl.ylocator = mticker.FixedLocator(latitudes)
	gl.xlines = True
	gl.ylines = True

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='upper left', markerscale=2., edgecolor='black', framealpha=1, ncol=4, fontsize=14, bbox_to_anchor=(0.29,-0.11))

	if outfile:
		plt.savefig(f'{outfile}', dpi=300, bbox_inches='tight')
	plt.show()


def plot_tracks(det_tracks, obs_tracks, lat_range=(0,70), lon_range=(100,320), outfile=None):
	# set map extent
	central_longitude = (lon_range[1] - lon_range[0])

	fig = plt.figure(figsize=(25,10))
	proj = ccrs.PlateCarree(central_longitude=central_longitude)
	ax = plt.axes(projection=proj)

	image_extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
	ax.set_extent(image_extent, crs=ccrs.PlateCarree())
	ax.coastlines(resolution='50m', lw=0.2)
	ax.add_feature(cf.LAND, facecolor='lightgrey', alpha=0.3)

	fontdict = {'weight':'bold', 'size':14}
	title_fontdict = {'size':18}
	ticksize = 12

	# plot tracks in each basin
	alpha = 0.5
	marker_size = 10.0
	transform = ccrs.PlateCarree()

	for i,id in enumerate(obs_tracks['TRACK_ID'].unique()):
		ax.plot(obs_tracks[obs_tracks['TRACK_ID']==id]['LON'], obs_tracks[obs_tracks['TRACK_ID']==id]['LAT'], alpha=0.2, transform=transform, color='blue')
		ax.scatter(obs_tracks[obs_tracks['TRACK_ID']==id]['LON'], obs_tracks[obs_tracks['TRACK_ID']==id]['LAT'], s=marker_size, marker='o', alpha=0.9, transform=transform, color='blue', label=f'Observed Tracks (#{len(obs_tracks["TRACK_ID"].unique())})' if i==0 else None)
	for i,id in enumerate(det_tracks['TRACK_ID'].unique()):
		ax.plot(det_tracks[det_tracks['TRACK_ID']==id]['LON'], det_tracks[det_tracks['TRACK_ID']==id]['LAT'], alpha=0.2, transform=transform, color='red')
		ax.scatter(det_tracks[det_tracks['TRACK_ID']==id]['LON'], det_tracks[det_tracks['TRACK_ID']==id]['LAT'], s=marker_size, marker='o', alpha=0.9, transform=transform, color='red', label=f'Detected Tracks (#{len(det_tracks["TRACK_ID"].unique())})' if i==0 else None)

	# x-axis
	longitudes = np.arange(lon_range[0], lon_range[1]+1, 10)
	lon_formatter = LongitudeFormatter(zero_direction_label=False)
	ax.xaxis.set_major_formatter(lon_formatter)
	ax.xaxis.set_major_locator(mticker.FixedLocator(longitudes-central_longitude))
	ax.set_xticklabels(longitudes, size=ticksize)
	ax.set_xticks(longitudes-central_longitude)
	ax.set_xlabel('Longitude [deg]', fontdict=fontdict)

	# y-axis
	latitudes = np.arange(lat_range[0], lat_range[1]+1, 10)
	lat_formatter = LatitudeFormatter()
	ax.yaxis.set_major_formatter(lat_formatter)
	ax.yaxis.set_major_locator(mticker.FixedLocator(latitudes))
	ax.set_yticklabels(latitudes, size=ticksize)
	ax.set_yticks(latitudes)
	ax.set_ylabel('Latitude [deg]', fontdict=fontdict)

	# ax.set_title(f"TSTORMS and ML inference on CMCC-CM3 (#{len(df)} matches)", fontdict=title_fontdict)

	# gridlines
	gl = ax.gridlines(crs=proj, draw_labels=False, dms=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
	gl.xlocator = mticker.FixedLocator(longitudes-central_longitude)
	gl.ylocator = mticker.FixedLocator(latitudes)
	gl.xlines = True
	gl.ylines = True

	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
	ax.legend(loc='upper left', markerscale=2., edgecolor='black', framealpha=1, ncol=4, fontsize=14, bbox_to_anchor=(0.29,-0.11))

	if outfile: 
		plt.savefig(f'{outfile}', dpi=300, bbox_inches='tight')
	plt.show()