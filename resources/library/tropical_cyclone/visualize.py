# import warnings
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


def plot_detections(
    detections, observations, lat_range=(0, 70), lon_range=(100, 320), outfile=None
):
    # set map extent
    central_longitude = lon_range[1] - lon_range[0]

    _ = plt.figure(figsize=(25, 10))
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax = plt.axes(projection=proj)

    image_extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    ax.set_extent(image_extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", lw=0.2)
    ax.add_feature(cf.LAND, facecolor="lightgrey", alpha=0.3)

    fontdict = {"weight": "bold", "size": 14}
    # title_fontdict = {'size':18}
    ticksize = 12

    # plot tracks in each basin
    marker_size = 10.0

    ax.scatter(
        detections["LON"],
        detections["LAT"],
        s=marker_size,
        marker="o",
        alpha=1.0,
        transform=ccrs.Geodetic(),
        label=f"TC Detections (#{len(detections)})",
    )
    ax.scatter(
        observations["LON"],
        observations["LAT"],
        s=marker_size,
        marker="o",
        alpha=1.0,
        transform=ccrs.Geodetic(),
        label=f"TC Observations (#{len(observations)})",
    )

    # x-axis
    longitudes = np.arange(lon_range[0], lon_range[1] + 1, 10)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.xaxis.set_major_locator(mticker.FixedLocator(longitudes - central_longitude))
    ax.set_xticklabels(longitudes, size=ticksize)
    ax.set_xticks(longitudes - central_longitude)
    ax.set_xlabel("Longitude [deg]", fontdict=fontdict)

    # y-axis
    latitudes = np.arange(lat_range[0], lat_range[1] + 1, 10)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.set_major_locator(mticker.FixedLocator(latitudes))
    ax.set_yticklabels(latitudes, size=ticksize)
    ax.set_yticks(latitudes)
    ax.set_ylabel("Latitude [deg]", fontdict=fontdict)

    # ax.set_title(f"TSTORMS and ML inference on CMCC-CM3 (#{len(df)} matches)", fontdict=title_fontdict)

    # gridlines
    gl = ax.gridlines(
        crs=proj,
        draw_labels=False,
        dms=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.xlocator = mticker.FixedLocator(longitudes - central_longitude)
    gl.ylocator = mticker.FixedLocator(latitudes)
    gl.xlines = True
    gl.ylines = True

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        loc="upper left",
        markerscale=2.0,
        edgecolor="black",
        framealpha=1,
        ncol=4,
        fontsize=14,
        bbox_to_anchor=(0.29, -0.11),
    )

    if outfile:
        plt.savefig(f"{outfile}", dpi=300, bbox_inches="tight")
    plt.show()


def plot_tracks(det_tracks, obs_tracks, lat_range, lon_range, title, outfile=None):
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
    track_id_col = 'TRACK_ID' if 'TRACK_ID' in det_tracks.columns else 'track_id'
    lon_col = 'LON' if 'LON' in det_tracks.columns else 'lon'
    lat_col = 'LAT' if 'LAT' in det_tracks.columns else 'lat'

    # plot tracks in each basin
    alpha = 0.5
    marker_size = 10.0
    transform = ccrs.PlateCarree()
    plt.title(title, fontsize = 18)
    if obs_tracks is not None:
        for i,id in enumerate(obs_tracks[track_id_col].unique()):
            ax.plot(
                obs_tracks[obs_tracks[track_id_col]==id][lon_col], 
                obs_tracks[obs_tracks[track_id_col]==id][lat_col], 
                alpha=0.2, 
                transform=transform, 
                color='blue'
                )
            ax.scatter(
                obs_tracks[obs_tracks[track_id_col]==id][lon_col], 
                obs_tracks[obs_tracks[track_id_col]==id][lat_col], 
                s=marker_size, 
                marker='o', 
                alpha=0.9, 
                transform=transform, 
                color='blue', 
                label=f'Observed Tracks (#{len(obs_tracks[track_id_col].unique())})' if i==0 else None
            )
    for i,id in enumerate(det_tracks[track_id_col].unique()):
        ax.plot(
            det_tracks[det_tracks[track_id_col]==id][lon_col], 
            det_tracks[det_tracks[track_id_col]==id][lat_col], 
            alpha=0.2, 
            transform=transform, 
            color='red'
        )
        ax.scatter(
            det_tracks[det_tracks[track_id_col]==id][lon_col], 
            det_tracks[det_tracks[track_id_col]==id][lat_col], 
            s=marker_size, 
            marker='o', 
            alpha=0.9, 
            transform=transform, 
            color='red', 
            label=f'Detected Tracks (#{len(det_tracks[track_id_col].unique())})' if i==0 else None
        )
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
    gl = ax.gridlines(
        crs=proj, 
        draw_labels=False, 
        dms=True, 
        linewidth=0.5, 
        color='gray', 
        alpha=0.5, 
        linestyle='--'
    )
    gl.xlocator = mticker.FixedLocator(longitudes-central_longitude)
    gl.ylocator = mticker.FixedLocator(latitudes)
    gl.xlines = True
    gl.ylines = True

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        loc='upper left', 
        markerscale=2., 
        edgecolor='black', 
        framealpha=1, 
        ncol=4, 
        fontsize=14, 
        bbox_to_anchor=(0.29,-0.11)
    )

    if outfile: 
        plt.savefig(f'{outfile}', dpi=300, bbox_inches='tight')
    else:
        plt.show()



def plot_pod_and_far(
    algo_results: pd.DataFrame, label, outfile = None
):
    x = np.arange(len(algo_results))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained', figsize=(12,6))
    
    # add horizontal mean
    ax.hlines(y=algo_results['pod'].mean(), xmin=-0.3, xmax=len(algo_results)-1.2, zorder=0, color='tab:blue', label='Average POD')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, algo_results['pod'].mean(), "{:.0f} %".format(algo_results['pod'].mean()), color="tab:blue", transform=trans, ha="right", va="center")
    
    ax.hlines(y=algo_results['far'].mean(), xmin=-0.3, xmax=len(algo_results)-1.2, zorder=999, color='tab:red', label='Average FAR')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, algo_results['far'].mean(), "{:.0f} %".format(algo_results['far'].mean()), color="tab:red", transform=trans, ha="right", va="center")
    
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(algo_results['pod'],2), width, label=f'POD' + label, color='tab:blue')
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(algo_results['far'],2), width, label=f'FAR' + label, color='tab:red')
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score', size = 14)
    ax.set_xlabel('TC trackers', size = 14)
    ax.set_xticks(x + width/2, algo_results['algo'], fontsize=12)
    ax.set_ylim(0, 100)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper left', markerscale=10, edgecolor='gray', framealpha=1, ncol=4, bbox_to_anchor=(0.2 , 0.98))
    
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()


def plot_track_durations(
    model_name, det_tracks, obs_tracks, outfile=None
):
    plt.figure(figsize=(10,6))

    track_durations = obs_tracks.track_id.value_counts().to_numpy()//4
    bins = (track_durations).max()
    freq_x, bin_edges_x = np.histogram(track_durations, bins=bins, range=(3,bins+3))
    plt.plot(bin_edges_x[:-3], freq_x[:-2], label='IBTrACS', drawstyle='steps', linewidth=2.0, color='tab:blue')

    track_durations = det_tracks.track_id.value_counts().to_numpy()//4
    bins = (track_durations).max()
    freq_x, bin_edges_x = np.histogram(track_durations, bins=bins, range=(3,bins+3))
    plt.plot(bin_edges_x[:-3], freq_x[:-2], label=model_name.upper(), drawstyle='steps', linewidth=2.0, color='tab:red')

    plt.xlabel('Track duration (days)', fontdict={'weight':'bold'})
    plt.title(f'Track Duration')

    plt.legend()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()