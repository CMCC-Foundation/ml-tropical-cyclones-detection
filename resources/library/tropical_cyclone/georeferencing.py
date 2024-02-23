import numpy as np
import math




def patch_to_rowcol_range(point, mag=None, patch_size=40, rowcol_size=1):
    """
    Converts the pixel coordinates (i,j) from the coarser resolution to the fine coordinates window.
    
    Parameters
    ----------
    point : (int,int)
        (row,col) coordinates of the pixel.
    ratio : int | None
        floor( coarse / fine ) value. Default to None.
    low_res : int 
        coarser resolution in km. Default to 36.
    high_res : int
        finer resolution in km. Default to 9.
    """
    if not mag:
        mag = math.floor( patch_size / rowcol_size ) # 4 
    return (mag*point[0], mag*point[0]+mag), (mag*point[1], mag*point[1]+mag)



def rowcol_range_to_latlon_range(ds, row_range, col_range):
    """
    Converts row-column range of values into latitude-longitude range of values from the provided dataset.

    """
    lats = ds.isel(lat=slice(*row_range), lon=slice(*col_range))['lat'].data
    lons = ds.isel(lat=slice(*row_range), lon=slice(*col_range))['lon'].data
    return (lats[0],lats[-1]), (lons[0],lons[-1])



def rowcol_to_patch(point, mag=None, patch_size=40, rowcol_size=1):
    """
    Converts the pixel coordinates (i,j) from the finer resolution to the coarser one.
    Parameters
    ----------
    point : (int,int)
        (row,col) coordinates of the pixel.
    mag : int | None
        floor( coarse / fine ) value. Default to None.
    low_res : int 
        coarser resolution in km. Default to 36.
    high_res : int
        finer resolution in km. Default to 9.
    """
    if not mag:
        mag = math.floor( patch_size / rowcol_size ) # 4 
    return math.floor(point[0]/mag), math.floor(point[1]/mag)



def get_global_rowcol_coords(lats, lons, latlon):
    """
    Get the row-col coordinates (considering the map as a matrix) corresponding to the 
    passed latlon geographical coordinates.

    """
    return np.array([[np.argwhere(lats==l)[0][0] for l in latlon[:,0]],[np.argwhere(lons==l)[0][0] for l in latlon[:,1]]]).transpose()



def from_local_to_global(patch_id, patch_row_col, patch_size=40):
    """
    Returns the global row-col coordinates corresponding to the local patch row-col coordinates.

    Parameters
    ----------
    patch_id : tuple(int, int)
        Row-column coordinates of the position of the patch
    patch_row_col : tuple(int, int)
        Row-column coordinates of the TC inside the patch
    patch_size : int
        Size of the patch
    
    Returns
    -------
    global_row_col: tuple(int, int)
        Row-column coordinates of the TC inside the entire domain

    """
    global_row = patch_size * patch_id[0] + patch_row_col[0]
    global_col = patch_size * patch_id[1] + patch_row_col[1]
    return (int(global_row), int(global_col))



def round_to_grid(x, grid_res=0.25):
    return grid_res * np.round(x/grid_res)

