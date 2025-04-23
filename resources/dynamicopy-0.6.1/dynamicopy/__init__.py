# If any private function or specific import is defined in a module, remove '*' and list the functions.
from .utils import idx_closest, sign_change_detect, hist2d
from .utils_geo import *
from .ncload import *
from .compute import *
from .plot import lon_lat_plot, zonal_plot
from dynamicopy.tc._basins import NH, SH, basins
from .tc_metrics import *
from .tc.matching import *
from .tc.ibtracs import *
from .tc.ET import *
from .tc.load_tracks import *

try:
    import cartopy.crs as ccrs
    from .cartoplot import *
except ImportError:
    print(
        "Failure in importing the cartopy library, the dynamicopy.cartoplot will not be loaded. \
    Please install cartopy if you wish to use it."
    )
