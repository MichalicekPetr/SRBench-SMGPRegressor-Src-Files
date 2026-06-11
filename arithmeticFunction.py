# Arithmetic function primitives used by the GP framework.
# This module defines simple function wrappers for addition, subtraction, and multiplication.

from __future__ import annotations

import numpy as np
from typing import Callable

class Function:
    def __init__(self, formula: Callable, name: str):
        self.formula = formula
        self.name = name

    def __call__(self, x, y):
        return self.formula(x, y)

    # --- BINÁRNÍ FUNKCE ---
    @staticmethod
    def addFunction(): return Function(lambda x, y: x + y, "+")
    
    @staticmethod   
    def subFunction(): return Function(lambda x, y: x - y, "-")    
    
    @staticmethod
    def multFunction(): return Function(lambda x, y: x * y, "*")
    
    @staticmethod
    def divFunction(): 
        # Bezpečné dělení: pokud je jmenovatel blízko nuly, vrátíme 1.0, jinak bezpečně vydělíme
        return Function(lambda x, y: np.where(np.abs(y) < 1e-9, 1.0, x / np.where(np.abs(y) < 1e-9, 1.0, y)), "/")
    
    @staticmethod
    def powFunction(): 
        # Bezpečné umocňování: Základ (x) vynutíme, aby byl nezáporný pomocí np.maximum(x, 0),
        # tím pádem NumPy nikdy nevygeneruje komplexní číslo (imaginární 'j')
        return Function(lambda x, y: np.clip(np.maximum(x, 0.0) ** y, -1e10, 1e10), "^")

    @staticmethod
    def logFunction(): 
        # Bezpečný logaritmus: Vynutíme, aby uvnitř logu byla minimálně hodnota 1e-9,
        # tím pádem logaritmus nikdy neuteče do -nekonečna
        return Function(lambda x, y: np.log(np.maximum(np.abs(x), 1e-9)), "log")

    # --- UNÁRNÍ FUNKCE ---
    @staticmethod
    def sinFunction(): return Function(lambda x, y: np.sin(x), "sin")
    
    @staticmethod
    def cosFunction(): return Function(lambda x, y: np.cos(x), "cos")
    
    @staticmethod
    def tanFunction(): return Function(lambda x, y: np.tan(x), "tan")
    
    @staticmethod
    def expFunction(): 
        return Function(lambda x, y: np.exp(np.clip(x, -100, 100)), "exp")
    
    @staticmethod
    def sqrtFunction(): return Function(lambda x, y: np.sqrt(np.abs(x)), "sqrt")
    
    @staticmethod
    def absFunction(): return Function(lambda x, y: np.abs(x), "abs")
    
    @staticmethod
    def asinFunction(): 
        return Function(lambda x, y: np.arcsin(np.clip(x, -1, 1)), "asin")
    
    @staticmethod
    def acosFunction(): 
        return Function(lambda x, y: np.arccos(np.clip(x, -1, 1)), "acos")
    
    @staticmethod
    def atanFunction(): return Function(lambda x, y: np.arctan(x), "atan")
    
    @staticmethod
    def sinhFunction(): 
        return Function(lambda x, y: np.sinh(np.clip(x, -50, 50)), "sinh")
    
    @staticmethod
    def coshFunction(): 
        return Function(lambda x, y: np.cosh(np.clip(x, -50, 50)), "cosh")
    
    @staticmethod
    def tanhFunction(): return Function(lambda x, y: np.tanh(x), "tanh")

    # --- STATICKÁ MAPA (Inicializuje se pouze jednou při importu třídy) ---
    _registry = {}

    @classmethod
    def decoder(cls, name: str) -> "Function":
        """
        Maximálně efektivní vyhledávání v předpřipravené mapě bez alokace nové paměti.
        """
        clean_name = name.strip().lower()
        if clean_name in cls._registry:
            return cls._registry[clean_name]
        raise ValueError(f"Funkce s názvem '{name}' není v setu SRBench podporována.")

# Naplnění registru jednorázově hned pod třídou
Function._registry = {
    # Binární
    "+": Function.addFunction(), "add": Function.addFunction(),
    "-": Function.subFunction(), "sub": Function.subFunction(), "subb": Function.subFunction(),
    "*": Function.multFunction(), "mul": Function.multFunction(), "mult": Function.multFunction(),
    "/": Function.divFunction(), "div": Function.divFunction(),
    "^": Function.powFunction(), "**": Function.powFunction(), "pow": Function.powFunction(),
    
    # Unární
    "sin": Function.sinFunction(),
    "cos": Function.cosFunction(),
    "tan": Function.tanFunction(),
    "exp": Function.expFunction(),
    "log": Function.logFunction(), "ln": Function.logFunction(),
    "sqrt": Function.sqrtFunction(),
    "abs": Function.absFunction(),
    "asin": Function.asinFunction(), "arcsin": Function.asinFunction(),
    "acos": Function.acosFunction(), "arccos": Function.acosFunction(),
    "atan": Function.atanFunction(), "arctan": Function.atanFunction(),
    "sinh": Function.sinhFunction(),
    "cosh": Function.coshFunction(),
    "tanh": Function.tanhFunction(),
}
