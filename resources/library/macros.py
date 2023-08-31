# TUTTE LE VARIABILI DISPONIBILI
ALL_DRIVER_VARS = ['fg10', 'i10fg', 'msl', 'sst', 't_500', 't_300', 'vo_850']
ALL_COORDINATE_VARS = ['real_cyclone', 'rounded_cyclone', 'global_cyclone', 'patch_cyclone']
CYCLONE_VAR = 'patch_cyclone'
MASK_VAR = 'cyclone_mask'

# ESPERIMENTI CON 4 VARIABILI
DRV_VARS_4 = ['fg10', 'msl', 't_500', 't_300']
DRV_VARS_6 = ['fg10', 'i10fg', 'msl', 't_500', 't_300', 'vo_850']

# VARIABILI DI CMCC-CM3
# CMCC_CM3_VARS = ['WSPDSRFMX','PSL','T500','T300']

# dataset parameters
PATCH_SIZE = 40
SHAPE = (PATCH_SIZE, PATCH_SIZE)

# YEARS SPLIT
TRAIN_YEARS = [y for y in range(1980, 2001)]
VALID_YEARS = [y for y in range(2001, 2011)]
TEST_YEARS = [y for y in range(2011, 2021)]