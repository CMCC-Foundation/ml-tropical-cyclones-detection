# TUTTE LE VARIABILI DISPONIBILI
ALL_DRIVER_VARS = ['fg10', 'i10fg', 'msl', 'sst', 't_500', 't_300', 'vo_850']
ALL_COORDINATE_VARS = ['real_cyclone', 'rounded_cyclone', 'global_cyclone', 'patch_cyclone']
CYCLONE_VARS = ['patch_cyclone']
MASK_VARS = ['cyclone_mask']

DENSITY_MAP_TC = 'density_map_tc'
SQUARE_MAP_TC = 'square_map_tc'
LABEL_MAP_TC = 'label_map_tc'
TARGET_VARS = [
    DENSITY_MAP_TC, 
    SQUARE_MAP_TC, 
    LABEL_MAP_TC, 
]

# dataset parameters
PATCH_SIZE = 40
SHAPE = (PATCH_SIZE, PATCH_SIZE)

# 30 years training-set
TRAIN_YEARS = [1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
# 4 years validation-set
VALID_YEARS = [2010, 2011, 2012, 2013]
# 8 years test-set
TEST_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
