"""
Created on 30.09.2020
@author: Soufiyan Bahadi
@director: Jean Rouat
@co-director: Eric Plourde
"""

FONT_SIZE = "12"

class FilterBank(Enum):
    aGC = 0
    cGC = 1
    GT = 2

class Example(Enum):
    SIG_ID = 2
    BATCH_ID = 61