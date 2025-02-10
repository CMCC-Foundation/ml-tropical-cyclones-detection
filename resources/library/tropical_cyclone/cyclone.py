from tropical_cyclone.macros import DENSITY_MAP_TC, SQUARE_MAP_TC, LABEL_MAP_TC
from tropical_cyclone.georeferencing import from_local_to_global

from math import sin, cos, sqrt, atan2, radians
from scipy.ndimage import gaussian_filter
from haversine import haversine_vector
import pandas as pd
import xarray as xr
import numpy as np
import os



def haversine_distance(lat1, lat2, lon1, lon2):
    # Approximate radius of earth in km
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance



def get_tc_map_labels(georef:pd.DataFrame, max_ts_tcs:int=20) -> pd.DataFrame:
    """
    Create a label pd.DataFrame in which, for each unique `ISO_TIME` in `georef`,
    it is associated the same label to each TC `SID` contained in georef.

    """
    # create a label pd.DataFrame
    label = pd.DataFrame(data={'ISO_TIME':georef['ISO_TIME'].unique()})
    for l in range(max_ts_tcs):
        # create a label to the label dataframe
        label[f'{l+1}'] = [np.nan] * len(label)
    # for each of the TCs in the dataset
    for sid in sorted(georef['SID'].unique()):
        # get the iso times in which it has developed
        iso_times = georef[georef['SID']==sid]['ISO_TIME'].sort_values().to_numpy()
        # get the first iso time in the label pd.DataFrame
        labels = label[label['ISO_TIME']==iso_times[0]].iloc[0,1:]
        # get the first nan index within the labels
        first_nan_idx = pd.isna(labels).argmax(axis=0)
        # add the TC SID to the corresponding label `first_nan_idx`
        label.loc[label['ISO_TIME'].isin(iso_times), f'{first_nan_idx+1}'] = sid
    return label



def get_tropical_cyclone_positions(ds:xr.Dataset, georef:pd.DataFrame, sigma:int=10) -> xr.Dataset:
    """
    Add Tropical Cyclone maps into the passed xr.Dataset. The maps built from `georef` and `label`
    pd.DataFrames are 3:
    1. `dm_tc` : map that, for every TC position has a gaussian filter of dimension 
        `sigma` pixels around its center.
    2. `sq_tc` - map that, for every TC position has a square of `radius` pixels around
        its center.
    3. `lb_tc` - map that, for every TC position has a square of `radius` pixels around
        its center, whose value is the same across each cyclone SID. This means that the 
        label correspondence between each TC occurrence is preserved among different timesteps.

    Parameters
    ----------
    ds : xr.Dataset
        Xarray dataset in which will be added the maps
    georef : pd.DataFrame
        Georeferencing dataset that contains the positions of each TC

    Returns
    -------
    ds : xr.Dataset
        The updated dataset

    Examples
    --------
    >>> ds = xr.open_dataset('path/to/netcdf/dataset')
    >>> georef = pd.DataFrame(data={'ISO_TIME':[], 'SID':[], 'LAT':[], 'LON':[], 'RLAT':[], 'RLON':[], 'YLAT':[], 'XLON':[]})
    >>> label = tc.cyclone.get_tc_map_labels(georef)
    >>> ds = tc.cyclone.get_tropical_cyclone_positions(ds, georef, label)

    """
    # make a copy of georeferencing pd.DataFrame
    gr = georef.copy()
    # convert gr column into datetime
    gr['ISO_TIME'] = pd.to_datetime(gr['ISO_TIME'])
    # get the shape of the 2D cyclone maps to be added to the xr.Dataset
    shape = (ds['time'].shape[0], ds['lat'].shape[0], ds['lon'].shape[0])
    # create a gaussian map that will contain the density map of each TC
    gaussian_map = np.zeros(shape=shape)
    # select only iso times that are compatible with our dataset
    gr = gr[gr['ISO_TIME'].isin(pd.to_datetime(ds['time']))]
    # loop over each time-step
    for t,iso_time in enumerate(gr['ISO_TIME'].unique()):
        rows = gr[gr['ISO_TIME']==iso_time]
        for _,row in rows.iterrows():
            j,i = row['XLON'], row['YLAT']
            # place 1 on the TC center
            gaussian_map[t,i,j] = 1
        # apply gaussian filtering over timestep t
        gaussian_map[t,:,:] = gaussian_filter(gaussian_map[t,:,:], sigma=sigma)
        # rescale the density map in [0,1]
        gaussian_map[t,:,:] = (gaussian_map[t,:,:] - gaussian_map[t,:,:].min()) / (gaussian_map[t,:,:].max() - gaussian_map[t,:,:].min())
    # create a density map in the xr.Dataset
    ds[DENSITY_MAP_TC] = (('time','lat','lon'), gaussian_map)
    # return the updated dataset
    return ds



def retrieve_predicted_tc(y_pred, ds, patch_ds, patch_size, eps: float = 0.1):
    """
    Retrieves the latitude-longitude coordinates from the passed predicted TCs

    eps: float = 0.1 - small value > 0

    """
    # create a latlons matrix with the same shape of y_pred_reshaped filled with nan
    cyclone_latlon_coords = np.full_like(y_pred, fill_value=np.nan)
    cyclone_rowcol_coords = np.full_like(y_pred, fill_value=np.nan)
    # for each timestep
    for t in range(y_pred.shape[0]):
        # for each row
        for i in range(y_pred.shape[1]):
            # for each column
            for j in range(y_pred.shape[2]):
                # correct the prediction if `eps` <= x <= 0.0 (slightly negative, could be an oscillation)
                if y_pred[t,i,j,0] < 0.0:
                    if y_pred[t,i,j,0] >= -eps:
                        y_pred[t,i,j,0] = 0.0
                if y_pred[t,i,j,1] < 0.0:
                    if y_pred[t,i,j,1] >= -eps:
                        y_pred[t,i,j,1] = 0.0
                # correct the prediction if x > 39.0 (too high, could be an oscillation)
                if y_pred[t,i,j,0] >= patch_size - 1:
                    y_pred[t,i,j,0] = patch_size - 1
                if y_pred[t,i,j,1] >= patch_size - 1:
                    y_pred[t,i,j,1] = patch_size - 1
                # if the model prediction is valid
                if y_pred[t,i,j,0] >= 0.0 and y_pred[t,i,j,1] >= 0.0:
                    try:
                        # retrieve global row-col coordinates of the TC
                        global_rowcol = from_local_to_global((i,j), y_pred[t,i,j,:], patch_size)
                        # retrieve global lat-lon coordinates of the TC
                        cyclone_latlon_coords[t,i,j,:] = (ds['lat'].data[global_rowcol[0]], ds['lon'].data[global_rowcol[1]])
                        cyclone_rowcol_coords[t,i,j,:] = (global_rowcol[0], global_rowcol[1])
                    except:
                        continue
    # update patch_ds cyclone_information
    patch_ds['patch_cyclone_pred'] = (('time','rows','cols','coordinate'), cyclone_latlon_coords)
    patch_ds['patch_rowcol_pred'] = (('time','rows','cols','rowcol'), cyclone_rowcol_coords)
    return patch_ds



def init_track_dataframe(detections: pd.DataFrame):
    # copy the dataframe
    tracks = detections.copy()
    # select only predicted cyclones
    tracks = tracks[(~np.isnan(tracks['LAT'])) & (~np.isnan(tracks['LON']))][['ISO_TIME', 'LAT', 'LON', 'WS']].reset_index(drop=True)
    # add empty track id data
    tracks['TRACK_ID'] = ['' for _ in range(len(tracks))]
    # add empty distance with previous TC
    tracks['HAVERSINE'] = [np.inf for _ in range(len(tracks))]
    return tracks



def tracking_algorithm(detected_tracks, max_distance=400.0, min_track_count=12):
    # get all the iso times within the detected tracks
    iso_times = sorted(detected_tracks['ISO_TIME'].unique())
    for t in range(len(iso_times)-1):
        # init track progressive
        k = 0
        # get current detections
        cur_detections = detected_tracks[detected_tracks['ISO_TIME']==iso_times[t]]
        # get next detections
        next_detections = detected_tracks[detected_tracks['ISO_TIME']==iso_times[t+1]]

        # check of 6-hour timelag
        time_delta_days = (pd.to_datetime(iso_times[t+1]) - pd.to_datetime(iso_times[t])).days
        time_delta_hr = ((pd.to_datetime(iso_times[t+1]) - pd.to_datetime(iso_times[t])).seconds / 3600)

        # prepare track ids
        for i,detection in cur_detections.iterrows():
            # prepare a track id
            new_track_id = str(iso_times[t]).replace(' ', '-').replace('-','').split(':')[0]+f'_{k}'
            k += 1
            # if the detection is new, it will have a new track id
            if detection['TRACK_ID'] == '':
                # assign a new track id to this track
                detected_tracks.iloc[i, detected_tracks.columns.get_loc('TRACK_ID')] = new_track_id
                cur_detections.loc[i, 'TRACK_ID'] = new_track_id

        # check if the timelag is 6 hours
        if not time_delta_hr == 6.0 or time_delta_days != 0:
            continue

        # merge current and next detections 
        merge = pd.merge(cur_detections, next_detections, how='cross')
        # compute haversine distance
        merge['HAVERSINE'] = haversine_vector(array1=merge[['LAT_x','LON_x']].to_numpy().astype(np.float32), array2=merge[['LAT_y','LON_y']].to_numpy(), normalize=True)
        # sort the values by haversine distance
        merge = merge.sort_values(by='HAVERSINE').reset_index(drop=True)
        # remove too distant matches
        merge = merge[merge['HAVERSINE'] <= max_distance]

        while True:
            # skip if there are no correspondences left
            if len(merge) == 0: break
            for i,row in merge.iterrows():
                # assign the track id into the original dataframe
                if row['TRACK_ID_y'] == '':
                    detected_tracks.iloc[detected_tracks[(detected_tracks['LAT']==row['LAT_y']) & (detected_tracks['LON']==row['LON_y']) & (detected_tracks['ISO_TIME']==row['ISO_TIME_y'])].index, detected_tracks.columns.get_loc('TRACK_ID')] = row['TRACK_ID_x']
                    detected_tracks.iloc[detected_tracks[(detected_tracks['LAT']==row['LAT_x']) & (detected_tracks['LON']==row['LON_x']) & (detected_tracks['ISO_TIME']==row['ISO_TIME_x'])].index, detected_tracks.columns.get_loc('HAVERSINE')] = row['HAVERSINE']
                else:
                    continue
                # remove from merge the other y rows
                merge = merge[~((merge['ISO_TIME_x']==row['ISO_TIME_x']) & (merge['LAT_x']==row['LAT_x']) & (merge['LON_x']==row['LON_x']))]
                merge = merge[~((merge['ISO_TIME_y']==row['ISO_TIME_y']) & (merge['LAT_y']==row['LAT_y']) & (merge['LON_y']==row['LON_y']))]
                break

    # prepare track ids for last day
    cur_detections = detected_tracks[detected_tracks['ISO_TIME']==iso_times[-1]]
    k = 0
    for i,detection in cur_detections.iterrows():
        # prepare a track id
        new_track_id = str(iso_times[-1]).replace(' ', '-').replace('-','').split(':')[0]+f'_{k}'
        k += 1
        # if the detection is new, it will have a new track id
        if detection['TRACK_ID'] == '':
            # assign a new track id to this track
            detected_tracks.iloc[i, detected_tracks.columns.get_loc('TRACK_ID')] = new_track_id
            cur_detections.loc[i, 'TRACK_ID'] = new_track_id

    # remove too short tracks
    detected_tracks = detected_tracks.groupby('TRACK_ID').filter(lambda x: len(x) >= min_track_count).reset_index(drop=True)
    return detected_tracks



def paper_tracking_algorithm(detected_tracks, max_distance=400.0, min_track_count=12, min_wind_speed=17.0):

    iso_times = sorted(detected_tracks['ISO_TIME'].unique())

    for t,iso_time in enumerate(iso_times):
        # progressivo del track id
        k=0

        # loop over each detection in that iso time 
        for i,detection in detected_tracks[detected_tracks['ISO_TIME']==iso_time].iterrows():

            # prepare a track id
            new_track_id = str(iso_time).replace(' ', '-').replace('-','').split(':')[0]+f'_{k}'
            k+=1

            # if the detection is new, it will have a new track id
            if detection['TRACK_ID'] == '':
                # assign a new track id to this track
                detected_tracks.iloc[i, detected_tracks.columns.get_loc('TRACK_ID')] = new_track_id
                track_id = new_track_id
            else:
                # get the current track id
                track_id = detection['TRACK_ID']

            try:
                # check of 6-hour timelag
                time_delta_days = (pd.to_datetime(iso_times[t+1]) - pd.to_datetime(iso_time)).days
                time_delta_hr = ((pd.to_datetime(iso_times[t+1]) - pd.to_datetime(iso_time)).seconds / 3600)
            except:
                continue

            # check if the timelag is 6 hours
            if not time_delta_hr == 6.0 or time_delta_days != 0:
                continue

            # verify distance
            next_iso_time = iso_times[t+1]

            # get next detections dataframe
            next_detections = detected_tracks[detected_tracks['ISO_TIME']==next_iso_time]

            # get current lat lon TC position
            cur_lat, cur_lon = detection['LAT'], detection['LON']

            # get next lats and lons from dataframe
            next_lats = next_detections['LAT'].to_numpy()
            next_lons = next_detections['LON'].to_numpy()

            dists = []
            # compute haversine distance for each lat lon couple
            for next_lat,next_lon in zip(next_lats,next_lons):
                dists.append(haversine_distance(cur_lat, next_lat, cur_lon, next_lon))

            # get min and argmin of the distances
            min_dist, min_arg = np.min(dists), np.argmin(dists)

            # get min arg index
            min_arg_idx = next_detections.index[min_arg]

            # if distance threshold is verified
            if min_dist <= max_distance:
                # update min distance of this record
                detected_tracks.iloc[i, detected_tracks.columns.get_loc('HAVERSINE')] = min_dist
                
                # define min track id for the update
                min_dist_track_id = next_detections.iloc[min_arg]['TRACK_ID']
                
                if min_dist_track_id == '':
                    #next_detections.iloc[min_arg, next_detections.columns.get_loc('TRACK_ID')] = track_id
                    detected_tracks.iloc[min_arg_idx, detected_tracks.columns.get_loc('TRACK_ID')] = track_id
                else:
                    new_detection = {
                        'ISO_TIME': [next_iso_time], 
                        'LAT': [next_detections.iloc[min_arg]['LAT']], 
                        'LON': [next_detections.iloc[min_arg]['LON']], 
                        'WS': [next_detections.iloc[min_arg]['WS']], 
                        'TRACK_ID': [track_id], 
                        'HAVERSINE': [next_detections.iloc[min_arg]['HAVERSINE']]
                        }
                    #pred_track_df = pred_track_df.append(new_detection, ignore_index=True)
                    detected_tracks = pd.concat([detected_tracks, pd.DataFrame(new_detection)])

    clean_pred_track_with_wind_df = pd.DataFrame({'ISO_TIME':[],'LAT':[],'LON':[],'WS':[],'TRACK_ID':[],'HAVERSINE':[]})
    clean_pred_track_without_wind_df = pd.DataFrame({'ISO_TIME':[],'LAT':[],'LON':[],'WS':[],'TRACK_ID':[],'HAVERSINE':[]})

    # remove tracks that are not sufficiently long
    for track_id in detected_tracks['TRACK_ID'].unique():
        track_id_df = detected_tracks[detected_tracks['TRACK_ID']==track_id]
        wind_track_id_df = track_id_df[track_id_df['WS'] > min_wind_speed]
        if len(wind_track_id_df) >= min_track_count:
            clean_pred_track_with_wind_df = pd.concat([clean_pred_track_with_wind_df, track_id_df])

    # remove tracks that are not sufficiently long
    for track_id in detected_tracks['TRACK_ID'].unique():
        track_id_df = detected_tracks[detected_tracks['TRACK_ID']==track_id]
        if len(track_id_df) >= min_track_count:
            clean_pred_track_without_wind_df = pd.concat([clean_pred_track_without_wind_df, track_id_df])
    
    return clean_pred_track_without_wind_df



def track_matching(
        detected_tracks: pd.DataFrame, 
        observed_tracks: pd.DataFrame, 
        max_track_distance: float=300.0
        ):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #           PHASE 0 : ASSOCIATE EACH PREDICTED TC DETECTION WITH TRUE TCs (1:M)                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    complete_match_df = pd.DataFrame(data={'ISO_TIME':[],'OBS_TRACK_ID':[], 'DET_TRACK_ID':[]})
    # iterate over all predicted track ids:
    for detected_track_id in detected_tracks['TRACK_ID'].unique():

        # get the entire predicted track
        for i,detected_location in detected_tracks[detected_tracks['TRACK_ID']==detected_track_id].iterrows():

            # get all the observed tracks in that iso time
            selected_observed_points = observed_tracks[observed_tracks['ISO_TIME']==detected_location['ISO_TIME']].copy()

            # no matches are found
            if not len(selected_observed_points):
                complete_match_df = pd.concat([complete_match_df, pd.DataFrame(data={'ISO_TIME':[detected_location['ISO_TIME']], 'OBS_TRACK_ID':[[]], 'DET_TRACK_ID':[detected_track_id]})])
                continue

            # get current lat lon TC position
            det_lat, det_lon = detected_location['LAT'], detected_location['LON']

            # get next lats and lons from dataframe
            obs_lats = selected_observed_points['LAT'].to_numpy()
            obs_lons = selected_observed_points['LON'].to_numpy()

            distances = []
            # compute haversine distance for each lat lon couple
            for obs_lat,obs_lon in zip(obs_lats,obs_lons):
                distances.append(haversine_distance(det_lat, obs_lat, det_lon, obs_lon))

            selected_observed_points['OD_DIST'] = distances

            # get only observed points that have a distance less than 300.0 km with respect to current detection
            OD_paired = selected_observed_points[selected_observed_points['OD_DIST'] <= max_track_distance]

            # update match dataframe with new track ids
            complete_match_df = pd.concat([complete_match_df, pd.DataFrame(data={'ISO_TIME':[detected_location['ISO_TIME']], 'OBS_TRACK_ID':[list(OD_paired['TRACK_ID'].unique())], 'DET_TRACK_ID':[detected_track_id]})])

    # reset the index
    complete_match_df = complete_match_df.reset_index(drop=True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #           PHASE 1 : GET THE LONGEST TRUE SEQUENCE AMONG THE TRUE 1:M TRACKS                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # create the track match dataframe
    match_df = pd.DataFrame(data={'OBS_TRACK_ID':[], 'DET_TRACK_ID':[]})

    for detected_track_id in detected_tracks['TRACK_ID'].unique():
        # get all unique true track ids
        obs_track_ids_unique = list(set([tti for ttis in complete_match_df[complete_match_df['DET_TRACK_ID']==detected_track_id]['OBS_TRACK_ID'].to_numpy() for tti in ttis]))

        # initialize true track count
        obs_track_id_count = {}
        for id in obs_track_ids_unique:
            obs_track_id_count.update({id:0})

        # update the track count with the occurrences that we find
        for obs_track_ids in complete_match_df[complete_match_df['DET_TRACK_ID']==detected_track_id]['OBS_TRACK_ID'].to_numpy():
            for obs_track_id in obs_track_ids:
                obs_track_id_count[obs_track_id]+=1

        # we haven't found true tracks corresponding to this predicted track
        if not len(obs_track_id_count.keys()):
            match_df = pd.concat([match_df, pd.DataFrame(data={'OBS_TRACK_ID':[''], 'DET_TRACK_ID':[detected_track_id]})])
            continue

        # update the match_df with the track with maximum number of elements
        match_df = pd.concat([match_df, pd.DataFrame(data={'OBS_TRACK_ID':[max(obs_track_id_count, key=obs_track_id_count.get)], 'DET_TRACK_ID':[detected_track_id]})])

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #           PHASE 2 : MERGE DET TRACKS THAT ARE ASSOCIATED WITH THE SAME OBS TRACK              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    for obs_track_id in match_df[match_df['OBS_TRACK_ID']!='']['OBS_TRACK_ID'].unique():
        # get all the true tracks in that iso time
        select_det_track_ids = match_df[match_df['OBS_TRACK_ID']==obs_track_id]['DET_TRACK_ID'].to_numpy()
        # if the number of pred tracks is more than 1...
        if len(select_det_track_ids) > 1:
            # leaving the first pred track as main...
            for detected_track_id in select_det_track_ids[1:]:
                # ... merge the other tracks (i.e., remove from the match dataframe)
                match_df = match_df[~((match_df['OBS_TRACK_ID']==obs_track_id) & (match_df['DET_TRACK_ID']==detected_track_id))]

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #           PHASE 3 : ADD DETECTED TRACK IDS WITHOUT CORRESPONDING OBSERVED TRACK ID            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # add those observed track ids without corresponding detected track
    missed_track_ids = set(observed_tracks['TRACK_ID'].unique()).difference(set(match_df['OBS_TRACK_ID'].unique()))

    for track_id in missed_track_ids:
        # update match dataframe with new track ids
        match_df = pd.concat([match_df, pd.DataFrame(data={'OBS_TRACK_ID':[track_id], 'DET_TRACK_ID':['']})])

    # reset the index
    match_df = match_df.reset_index(drop=True)

    return match_df



# INCORRECT TRACKER
# def tracking_track_algorithm(tracks, max_distance=400.0, min_track_count=12):
#     # define columns and rename columns
#     columns = ['ISO_TIME_y', 'LAT_y', 'LON_y', 'TRACK_ID_x', 'HAVERSINE']
#     rename_cols = {'ISO_TIME_y':'ISO_TIME', 'LAT_y':'LAT', 'LON_y':'LON', 'TRACK_ID_x': 'TRACK_ID'}
#     # create an empty detected tracks `pd.DataFrame`
#     detected_tracks = pd.DataFrame(data={'ISO_TIME':[], 'LAT':[], 'LON':[], 'TRACK_ID':[], 'HAVERSINE':[]})
#     # get all iso times
#     iso_times = tracks['ISO_TIME'].unique()
#     # convert iso times to pandas datetime
#     iso_times = pd.to_datetime(iso_times)
#     # get first detections with first iso time
#     dets = tracks[tracks['ISO_TIME'] == iso_times[0]]
#     # add cosine as infinite
#     dets['HAVERSINE'] = np.inf
#     # assign track ids to the detections
#     dets['TRACK_ID'] = [i for i in range(len(dets))]
#     # add the detections to the tracks dataframe
#     detected_tracks = pd.concat([detected_tracks, dets])
#     # for each iso time (from the 2nd to the last)
#     for iso_time in iso_times[1:]:
#         # get last 6h detections from tracks
#         prev_dets = detected_tracks[detected_tracks['ISO_TIME'] == iso_time - pd.DateOffset(hours=6)]
#         # get current detections
#         cur_dets = tracks[tracks['ISO_TIME'] == iso_time]
#         # if no previous tracks are found
#         if len(prev_dets) == 0:
#             # add new track ids
#             cur_dets['TRACK_ID'] = [i+detected_tracks['TRACK_ID'].max()+1 for i in range(len(cur_dets))]
#             # add the detections to the detected_tracks
#             detected_tracks = pd.concat([detected_tracks, cur_dets])
#             continue
#         # set multiply between previous detections and current detections
#         merge_dets = pd.merge(left=prev_dets, right=cur_dets, how='cross')
#         # compute haversine distance
#         merge_dets['HAVERSINE'] = haversine_vector(array1=merge_dets[['LAT_x','LON_x']].to_numpy(), array2=merge_dets[['LAT_y','LON_y']].to_numpy(), normalize=True)
#         # remove high distance detections
#         merge_dets = merge_dets[merge_dets['HAVERSINE'] < max_distance]
#         # remove multiple correspondences on y (get min haversine)
#         merge_dets_tmp = merge_dets.copy()
#         for i,row in merge_dets[['LAT_y','LON_y']].drop_duplicates().iterrows():
#             md = merge_dets[(merge_dets['LAT_y']==row['LAT_y']) & (merge_dets['LON_y']==row['LON_y'])]
#             md_id = md[md['HAVERSINE'] != md.min()['HAVERSINE']].index
#             merge_dets_tmp = merge_dets_tmp.drop(index=md_id)
#         merge_dets = merge_dets_tmp
#         # remove multiple correspondences on x (get min haversine)
#         merge_dets_tmp = merge_dets.copy()
#         for i,row in merge_dets[['LAT_x','LON_x']].drop_duplicates().iterrows():
#             md = merge_dets[(merge_dets['LAT_x']==row['LAT_x']) & (merge_dets['LON_x']==row['LON_x'])]
#             md_id = md[md['HAVERSINE'] != md.min()['HAVERSINE']].index
#             merge_dets_tmp = merge_dets_tmp.drop(index=md_id)
#         merge_dets = merge_dets_tmp
#         # get only some columns
#         merge_dets = merge_dets[columns]
#         # rename columns
#         merge_dets = merge_dets.rename(columns=rename_cols)
#         # add merge detections to detected tracks
#         detected_tracks = pd.concat([detected_tracks, merge_dets])
#     # remove too short tracks
#     detected_tracks = detected_tracks.groupby('TRACK_ID').filter(lambda x: len(x) >= min_track_count).reset_index(drop=True)
#     # reset index
#     detected_tracks = detected_tracks.reset_index(drop=True)
#     return detected_tracks
