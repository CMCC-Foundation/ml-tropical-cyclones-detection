import tensorflow as tf
from enum import Enum

# default loss configuration reduction
REDUCTION = tf.keras.losses.Reduction.NONE

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                         #
#                                           NetCDF Variables                                              #
#                                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# TUTTE LE VARIABILI DISPONIBILI
ALL_DRIVER_VARS = ['fg10', 'i10fg', 'msl', 'sst', 't_500', 't_300', 'vo_850']
ALL_COORDINATE_VARS = ['real_cyclone', 'rounded_cyclone', 'global_cyclone', 'patch_cyclone']
CYCLONE_VAR = 'patch_cyclone'
MASK_VAR = 'cyclone_mask'

# ESPERIMENTI TIPO 1
# variabili per la prima parte degli esperimenti (regressione per trovare coordinate row-col intra-patch)
DRV_VARS_1 = ['fg10', 'msl', 't_500', 't_300']
COO_VARS_1 = ['patch_cyclone']
MSK_VAR_1 = None

# ESPERIMENTI TIPO 2
# variabili per la seconda parte degli esperimenti (segmentazione ma con le stesse variabili dell'esperimento 1)
DRV_VARS_2 = ['fg10', 'msl', 't_500', 't_300']
COO_VARS_2 = None
MSK_VAR_2 = 'cyclone_mask'

# ESPERIMENTI TIPO 3
# tutti i driver (tranne sst) per il task di regressione
DRV_VARS_3 = ['fg10', 'i10fg', 'msl', 't_500', 't_300', 'vo_850']
COO_VARS_3 = ['patch_cyclone']
MSK_VAR_3 = None

# ESPERIMENTI TIPO 4
# tutti i driver dell'esperimento 3, ma applicati al task di segmentazione
DRV_VARS_4 = ['fg10', 'i10fg', 'msl', 't_500', 't_300', 'vo_850']
COO_VARS_4 = None
MSK_VAR_4 = 'cyclone_mask'


# dataset parameters
PATCH_SIZE = 40
SHAPE = (PATCH_SIZE, PATCH_SIZE)

TEST_YEARS = [] # for test purposes
TRAINVAL_YEARS = [2000, 2001, 2002] # for test purposes



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                                         #
#                                               Enumerations                                              #
#                                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# descrive il tipo di patch che bisogna prendere 
class PatchType(Enum):
    ALLADJACENT = 'alladjacent'
    CYCLONE = 'cyclone'
    NEAREST = 'nearest'
    RANDOM = 'random'
    NOCYCLONE = 'nocyclone'

# descrive il tipo di augmentation che deve essere effettuata
class AugmentationType(Enum):
    ALL_PATCHES = 'all_patches'
    ONLY_TCS = 'only_tcs'

# descrive il nome del modello di rete neurale da utilizzare 
class Network(Enum):
    VGG_V1 = 'vgg_v1'       # map-to-coord
    VGG_V2 = 'vgg_v2'       # map-to-coord
    VGG_V3 = 'vgg_v3'       # map-to-coord
    VGG_V4 = 'vgg_v4'       # map-to-coord
    MODEL_V5 = 'model_v5'   # map-to-coord
    UNET = 'unet'           # map-to-map
    UNETPP = 'unetpp'       # map-to-map
    PIX2PIX = 'pix2pix'     # map-to-map

# ritorna nome della loss utilizzata in fase di training
class Losses(Enum):
    # Mean Absolute Error
    MAE = ('mae', 'mae') 
    # Mean Squared Error
    MSE = ('mse', 'mse') 
    # Cyclone Classification Localization
    #CCL = ('ccl', CycloneClassificationLocalizationLoss(reduction=REDUCTION, name='ccl')) 
    # Dice Loss
    #DL = ('dice', DiceLoss(reduction=REDUCTION, name='dice'))
    # Binary CrossEntropy + Dice Coefficient
    #BCEDL = ('bce_dice', BCEDiceLoss(reduction=REDUCTION, name='bce_dice')) 
    # Binary CrossEntropy
    BCE = ('bce', tf.keras.losses.BinaryCrossentropy(reduction=REDUCTION, name='bce')) 
    # Sparse Categorical CrossEntropy
    SCCE = ('scce', tf.keras.losses.SparseCategoricalCrossentropy(reduction=REDUCTION, name='scce')) 
    # No specified loss
    NONE = ('none', None) 

# descrive il tipo di esperimento che si tenta di eseguire
class Experiment(Enum):
    EXP_1 = ('exp_1', (DRV_VARS_1, COO_VARS_1, MSK_VAR_1, [len(DRV_VARS_1), 2]) )
    EXP_2 = ('exp_2', (DRV_VARS_2, COO_VARS_2, MSK_VAR_2, [len(DRV_VARS_2), 1]) )
    EXP_3 = ('exp_3', (DRV_VARS_3, COO_VARS_3, MSK_VAR_3, [len(DRV_VARS_3), 2]) )
    EXP_4 = ('exp_4', (DRV_VARS_4, COO_VARS_4, MSK_VAR_4, [len(DRV_VARS_4), 1]) )

# descrive la forza della regolarizzazione
class RegularizationStrength(Enum):
    WEAK = ('weak', tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0001)) # l1=0 - l2=0.0001
    MEDIUM = ('medium', tf.keras.regularizers.l1_l2(l1=0.0001, l2=0.0001)) # l1=0.0001 - l2=0.0001
    STRONG = ('strong', tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)) # l1=0.001 - l2=0.001
    VERY_STRONG = ('very_strong', tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)) # l1=0.01 - l2=0.01
    NONE = ('none', None)   # no regularization

# descrive l'attivazione dell'ultimo layer del modello
class Activation(Enum):
    RELU = 'relu'
    LINEAR = 'linear'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'

# label assegnata ad un ciclone assente
class LabelNoCyclone(Enum):
    ZERO_3 = -0.3
    ONE = -1.0
    NONE = None
