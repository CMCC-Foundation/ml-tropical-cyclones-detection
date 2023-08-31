from enum import Enum

# descrive il tipo di patch che bisogna prendere 
class PatchType(Enum):
    ALLADJACENT = 'alladjacent'
    CYCLONE = 'cyclone'
    NEAREST = 'nearest'
    RANDOM = 'random'
    NOCYCLONE = 'nocyclone'
