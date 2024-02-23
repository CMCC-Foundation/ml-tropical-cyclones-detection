import numpy as np
import torch
import os



def seed_everything(seed: int):
    """
    Seed RNG states of the execution environment.

    """
    # python seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # numpy seed
    np.random.seed(seed)
    # pytorch seed
    torch.manual_seed(seed)
    print(f'Execution seeded with seed {seed}')


def get_ibtracs_by_basins(ibtracs_df, basins=['WP', 'EP', 'NA']):
    """
    Given the list of basins, return the IBTrACS DataFrame

    Parameters
    ----------

    ibtracs_df : pd.DataFrame
        ibtracs DataFrame containing all basins information

    basins : list(str)
        List of basins from which we want to extract basin information.
    
    """
    # get SIDs list for each basin
    SIDs_list = [ibtracs_df[ibtracs_df['BASIN']==basin]['SID'].unique().tolist() for basin in basins]
    # convert SIDs list into sets
    SIDs_sets = [set(SIDs) for SIDs in SIDs_list]
    
    # compute difference between basins to find out basin-only TCs
    SIDs_basin_only = {}
    for sids, basin in zip(SIDs_sets, basins):
        # get list of SIDs present only in that basin
        for s,b in zip(SIDs_sets,basins):
            # skip if b is equal to basin
            if basin == b:
                continue
            # remove from sids the other basins
            sids = sids.difference(s)
        SIDs_basin_only.update({basin:sids})

    # compute intersection between basins to find out cross-basin TCs
    SIDs_basin_intersection = {}
    if len(basins) > 1:
        for sids, basin in zip(SIDs_sets, basins):
            # get list of SIDs present only in that basin
            for s,b in zip(SIDs_sets,basins):
                # skip if b is equal to basin
                if basin == b:
                    continue
                if not basin+'_and_'+b in list(SIDs_basin_intersection.keys()) and not b+'_and_'+basin in list(SIDs_basin_intersection.keys()):
                    # intersection between sids and the other's basins sids
                    SIDs_basin_intersection.update({basin+'_and_'+b:sids.intersection(s)})
    
    # get only records containing the already extracted SIDs
    ibtracs_only_dfs = {}
    for basin,sids in SIDs_basin_only.items():
        ibtracs_only_dfs.update({basin:ibtracs_df[ibtracs_df['SID'].isin(sids)]})
    ibtracs_intersection_dfs = {}
    for basin,sids in SIDs_basin_intersection.items():
        ibtracs_intersection_dfs.update({basin:ibtracs_df[ibtracs_df['SID'].isin(sids)]})
    
    return ibtracs_only_dfs, ibtracs_intersection_dfs


def pixel_to_km(arr, grid_resolution=0.25, km_to_deg=110.474):
    return arr * grid_resolution * km_to_deg


def pixel_to_deg(arr, grid_resolution=0.25):
    return arr * grid_resolution


def coo_rot180(data):
    X, y = data
    y = y[0]
    patch_size = X.shape[1]
    X = torch.permute(torch.rot90(torch.permute(X, dims=(1,2,0)), k=2, dims=(0,1)), dims=(2,0,1))
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [-y[0] + patch_size -1, -y[1] + patch_size -1]
    return (X, torch.as_tensor(y1).unsqueeze(0))


def coo_left_right(data):
    X,y = data
    y = y[0]
    patch_size = X.shape[1]
    X = torch.permute(torch.fliplr(torch.permute(X, dims=(1,2,0))), dims=(2,0,1))
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [y[0], - y[1] + patch_size -1]
    return (X, torch.as_tensor(y1).unsqueeze(0))


def coo_up_down(data):
    X,y = data
    y = y[0]
    patch_size = X.shape[1]
    X = torch.permute(torch.flipud(torch.permute(X, dims=(1,2,0))), dims=(2,0,1))
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [- y[0] + patch_size -1, y[1]]
    return (X, torch.as_tensor(y1).unsqueeze(0))
