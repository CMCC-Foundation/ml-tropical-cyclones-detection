from math import floor
import itertools
import random


def get_nocyclones_patches(dataset, cyclone_patch_ids, patch_size=40):
    """
    Dato un dataset ed una lista di coordinate delle patch contenenti cicloni, otteniamo una lista 
    di patch che invece non contengono un ciclone.
    
    Parameters
    ----------
    dataset:
        E' il xr.Dataset da un timestep contentente la mappa da suddividere in patch.
    patch_cyclone_ids:
        E' una lista di coordinate riga-colonna delle patch della mappa.

    """
    row_blocks = floor(len(dataset.lat.data) / patch_size)
    col_blocks = floor(len(dataset.lon.data) / patch_size)

    patch_row_idx = [i for i in range(row_blocks)]
    patch_col_idx = [i for i in range(col_blocks)]

    # get all coordinates
    all_coords = [coords for coords in list(itertools.product(patch_row_idx, patch_col_idx))]

    # remove from all coordinates the TC coords
    no_tc_coords = list(set(all_coords).difference(set(cyclone_patch_ids)))

    nocyclone_patch_ids = set()
    for coo in no_tc_coords:
        nocyclone_patch_ids.add(coo)

    return nocyclone_patch_ids



def get_all_adjacent_patches(dataset, patch_cyclone_ids, patch_size=40):
    """
    Data una lista di patch id contenti cicloni, si ottengono tutte le patch nell'intorno.

    Parameters
    ----------
    dataset:
        E' il xr.Dataset da un timestep contentente la mappa da suddividere in patch.
    patch_cyclone_ids:
        E' una lista di coordinate riga-colonna delle patch della mappa.

    """
    row_blocks = floor(len(dataset.lat.data) / patch_size)
    col_blocks = floor(len(dataset.lon.data) / patch_size)
    tmp_all_adjacent_ids = []
    for patch_id in patch_cyclone_ids:
        i, j = patch_id
        tmp_all_adjacent_ids += [a for a in itertools.product([i-1, i, i+1], [j-1, j, j+1])]
    all_adjacent_ids = set()
    for aa_id in tmp_all_adjacent_ids:
        i, j = aa_id
        if 0 <= i < row_blocks and 0 <= j < col_blocks:
            all_adjacent_ids.add(aa_id)
    all_adjacent_ids = all_adjacent_ids.difference(set(list(map(tuple, patch_cyclone_ids))))
    return all_adjacent_ids



def get_nearest_adjacent_patches(dataset, patch_cyclone_ids, patch_cyclone_positions, patch_size=40):
    """
    Data una lista di patch id contenti cicloni e delle relative posizioni locali (nella patch), si ottengono le tre patch più vicine al ciclone.

    Parameters
    ----------
    dataset:
        E' il xr.Dataset da un timestep contentente la mappa da suddividere in patch.
    patch_cyclone_ids:
        E' una lista di coordinate riga-colonna delle patch della mappa.
    patch_cyclone_positions:
        E' una lista di coordinate riga-colonna che identificano la posizione del ciclone all'interno della relativa patch.

    """
    def is_first_quadrant(y, x, half_patch_size):
        return y < half_patch_size and x >= half_patch_size
    def is_second_quadrant(y, x, half_patch_size):
        return y < half_patch_size and x < half_patch_size
    def is_third_quadrant(y, x, half_patch_size):
        return y >= half_patch_size and x < half_patch_size
    def is_fourth_quadrant(y, x, half_patch_size):
        return y >= half_patch_size and x >= half_patch_size

    row_blocks = len(dataset.lat.data) // patch_size
    col_blocks = len(dataset.lon.data) // patch_size
    nearest_patch_ids = set()
    half_patch_size = patch_size / 2
    for patch_id, cyclone_position in zip(patch_cyclone_ids, patch_cyclone_positions):
        i,j = patch_id
        y,x = cyclone_position
        if is_first_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i-1, j   )) if i-1 >= 0 else None
            nearest_patch_ids.add(( i-1, j+1 )) if i-1 >= 0 and j+1 < col_blocks else None
            nearest_patch_ids.add(( i  , j+1 )) if j+1 < col_blocks else None
        elif is_second_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i-1, j-1 )) if i-1 >= 0 and j-1 >= 0 else None
            nearest_patch_ids.add(( i  , j-1 )) if j-1 >= 0 else None
            nearest_patch_ids.add(( i-1, j   )) if i-1 >= 0 else None
        elif is_third_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i  , j-1 )) if j-1 >= 0 else None
            nearest_patch_ids.add(( i+1, j-1 )) if i+1 < row_blocks and j-1 >= 0 else None
            nearest_patch_ids.add(( i+1, j   )) if i+1 < row_blocks else None
        elif is_fourth_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i  , j+1 )) if j+1 < col_blocks else None
            nearest_patch_ids.add(( i+1, j+1 )) if i+1 < row_blocks and j+1 < col_blocks else None
            nearest_patch_ids.add(( i+1, j   )) if i+1 < row_blocks else None
    # remove patch cyclone ids from the nearest ones
    nearest_patch_ids = nearest_patch_ids.difference(set(list(map(tuple, patch_cyclone_ids))))
    return nearest_patch_ids



def get_random_patches(dataset, patch_cyclone_ids, patch_size=40):
    """
    Dato un dataset ed una lista di coordinate delle patch contenenti cicloni, otteniamo una lista (di lunghezza uguale 
    al numero di cicloni presente nel campione) di patch random che non si trovano nell'intorno della patch considerata.
    
    Parameters
    ----------
    dataset:
        E' il xr.Dataset da un timestep contentente la mappa da suddividere in patch.
    patch_cyclone_ids:
        E' una lista di coordinate riga-colonna delle patch della mappa.

    """
    row_blocks = floor(len(dataset.lat.data) / patch_size)
    col_blocks = floor(len(dataset.lon.data) / patch_size)

    patch_row_idx = [i for i in range(row_blocks)]
    patch_col_idx = [i for i in range(col_blocks)]

    inv_coords = []
    for patch_id in patch_cyclone_ids:
        i, j = patch_id

        #(5,5)
        #(4,5),(5,4),(4,4),(6,6),(6,5),(5,6),(6,4),(4,6)
        inv_i = [i-1, i, i+1]
        inv_j = [j-1, j, j+1]
        inv_coords.extend(list(itertools.product(inv_i, inv_j)))

    no_tc_coords_candidates = [coords for coords in list(itertools.product(patch_row_idx, patch_col_idx)) if coords not in inv_coords]
    no_tc_coords = random.sample(no_tc_coords_candidates, len(patch_cyclone_ids))

    random_patch_ids = set()
    for coo in no_tc_coords:
        random_patch_ids.add(coo)

    return random_patch_ids
