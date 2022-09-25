"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""
from enum import Enum

FONT_SIZE = "12"
ENGLISH_MAX_LEN = 55718
GERMAN_MAX_LEN = 65872 # Longest audio signal useful for zero padding
# ENGLISH_TRAIN_NUM_SAMPLES = 4011
# ENGLISH_TEST_NUM_SAMPLES = 1079
# GERMAN_TRAIN_NUM_SAMPLES = 4145
# GERMAN_TEST_NUM_SAMPLES = 1185

class FilterBank(Enum):
    aGC = 0
    cGC = 1
    GT = 2

class Example(Enum):
    SIG_ID = 2
    BATCH_ID = 61
