# Smooth multifunction evaluation and Taylor interpolation support.
# This module maps continuous f-values to buffered arithmetic operations.
from __future__ import annotations

from typing import Callable, List
import arithmeticFunction
from taylorFunc import TaylorFunc

class SmoothMultifunctionSet:
    """Container for smooth multifunction evaluation and interpolation.

    Stores a mapping of discrete arithmetic functions and optionally applies
    a triangle-wave transformation to the function value before interpolation.
    """

    def __init__(self, name: str = "", minVal: int = 0, maxVal: int = 3, taylorSumElements: int = 100, useTriangleFval: bool = True):
        """Initialize a smooth multifunction set.

        Args:
            name: Human-readable name for the function set.
            minVal: Minimum function index.
            maxVal: Maximum function index.
            taylorSumElements: Number of interpolation samples for triangle mapping.
            useTriangleFval: If True, use triangular f-value mapping instead of raw fval.
        """
        self.name = name
        self.minVal = minVal
        self.maxVal = maxVal
        self.taylorSumElements = taylorSumElements
        self.useTriangleFval = useTriangleFval
        self.functionMap = dict()

    def addFunction(self, func: Callable, val: int) -> None:
        """Add a function object for a specific discrete index value."""
        self.functionMap[val] = func

    def calculateResult(self, x: float, y: float, fval: float, N: int | None = None) -> float:
        """Calculate a smooth result from two operands and an f-value.

        The triangular mapping is applied when useTriangleFval is enabled.
        The final result is a weighted blend of the two nearest functions.
        """
        if N is None:
            N = self.taylorSumElements

        if(fval > self.maxVal):
            print("Val is greater than maxVal")
            raise ValueError()
        
        if(fval < self.minVal):
            print("Val is lower than minVal")
            raise ValueError()
        
        if self.useTriangleFval:
            val = TaylorFunc.x_triangle(fval, N)
        else:
            val = fval

        f1 = self.functionMap.get(int(val))
        # Use modulo maxVal to safely wrap around the circular buffer
        f2 = self.functionMap.get((int(val) + 1) % self.maxVal)

        if f1 is None:
            raise ValueError(f"Function for index {int(val)} not found in functionMap")
        if f2 is None:
            raise ValueError(f"Function for index {(int(val) + 1) % self.maxVal} not found in functionMap")

        fval2 = val - int(val)
        fval1 = 1 - fval2
        
        result1 = fval1 * f1.formula(x, y)
        result2 = fval2 * f2.formula(x, y)

        return result1 + result2
    
    @staticmethod
    def createClassicMultifunctionSet(taylorSumElements: int = 100, useTriangleFval: bool = True) -> SmoothMultifunctionSet:
        """Create a default multifunction set with add, subtract, and multiply."""
        fset = SmoothMultifunctionSet(name = "+,-,*", minVal=0, maxVal=3, taylorSumElements=taylorSumElements, useTriangleFval=useTriangleFval)
        fset.addFunction(Function.addFunction(), 0)
        fset.addFunction(Function.subFunction(), 1)
        fset.addFunction(Function.multFunction(), 2)
        return fset

    @staticmethod
    def createMultifunctionSetByNames(function_names: List[str], taylorSumElements: int = 100, useTriangleFval: bool = True) -> SmoothMultifunctionSet:
        """Dynamically create a SmoothMultifunctionSet based on a list of function names from SRBench."""
        if not function_names:
            raise ValueError("The function list for MultifunctionSet cannot be empty.")

        # Lokální import pro bezpečné rozbití cyklických importů
        from arithmeticFunction import Function

        set_name = ",".join(function_names)
        num_functions = len(function_names)

        fset = SmoothMultifunctionSet(
            name=set_name, 
            minVal=0, 
            maxVal=num_functions, 
            taylorSumElements=taylorSumElements, 
            useTriangleFval=useTriangleFval
        )

        for index, name in enumerate(function_names):
            try:
                # Voláme tvůj originální, skvělý dekodér!
                decoded_func = Function.decoder(name)
                fset.addFunction(decoded_func, index)
            except ValueError as e:
                print(f"Initialization warning: {e}")
                raise e

        return fset

    @staticmethod
    def createBasicFunctionSet(taylorSumElements: int = 100, useTriangleFval: bool = True) -> SmoothMultifunctionSet:
        """Create a default basic multifunction set for black-box regression tasks."""
        # Lokální import pro jistotu i zde
        from arithmeticFunction import Function

        blackbox_functions = ["add", "sub", "mul", "div", "pow", "sin", "log"]
        num_functions = len(blackbox_functions)
        
        symbols = ["+", "-", "*", "/", "^", "sin", "log"]
        set_name = ",".join(symbols)

        fset = SmoothMultifunctionSet(
            name=set_name, 
            minVal=0, 
            maxVal=num_functions, 
            taylorSumElements=taylorSumElements, 
            useTriangleFval=useTriangleFval
        )

        for index, name in enumerate(blackbox_functions):
            decoded_func = Function.decoder(name)
            fset.addFunction(decoded_func, index)

        return fset