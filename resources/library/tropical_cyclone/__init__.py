# ==============================================================================
#
# Author : Davide Donno
#
# ==============================================================================
"""
Top-level module of Tropical Cyclone Detection library. By convention, we refer to this module as
`tc` instead of `tropical_cyclone`, following the common practice of importing
TropicalCyclone via the command `import tropical_cyclone as tc`.

The primary function of this module is to import all of the public TropicalCyclone
interfaces into a single place. The interfaces themselves are located in
sub-modules, as described below.

"""
from . import callbacks
from . import cyclone
from . import data_io as io
from . import dataset
from . import era5
from . import georeferencing as georef
from . import inference
from . import macros
from . import models
from . import patch_proc as patchproc
from . import sampler
from . import scaling
from . import tester
from . import trainer
from . import utils
from . import visualize
