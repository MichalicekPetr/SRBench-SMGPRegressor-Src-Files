# Utility helper functions used by genetic operators.
# Includes probabilistic and random selection support methods.

from __future__ import annotations

import random
import math

def get_declining_random_fast(n: int, rate: float = 1.0, rng: random.Random = None) -> int:
    rng = rng if rng is not None else random
    u = rng.random()  # Náhodné číslo od 0.0 do 1.0
    
    # Inverzní transformace pro exponenciální rozdělení
    # (převádí rovnoměrné rozdělení na exponenciální klesající)
    val = -math.log(1 - u * (1 - math.exp(-rate))) / rate
    
    # Přemapujeme hodnotu z rozsahu (0..1) na celé číslo od 1 do n
    result = int(val * n) + 1
    
    # Pojistka pro extrémní případ, kdy random() vrátí přesně 1.0
    return min(result, n)