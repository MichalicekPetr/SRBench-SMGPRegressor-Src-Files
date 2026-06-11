# Variable container used during expression evaluation.
# Represents a named feature value in the GP individual evaluation.

from __future__ import annotations

class Variable:
    def __init__(self, name: str):
        self.name = name
        self.initialized = False
        self.value = None

    def setValue(self, value: float):
        self.value = value
        self.initialized = True

    def __str__(self) -> str:
        if not self.initialized:
            return self.name
        else:
            return f"{self.name}({self.value:.2f})"
        
    @staticmethod
    def setVariableValues(variableList: list, values: list) -> None:
        if len(variableList) != len(values):
            raise ValueError("Length of variable list and values list must be the same")

        for i in range(len(variableList)):
            variableList[i].setValue(values[i])