# GP tree node model used for human-readable tree output.
# Supports function nodes, terminal constants, and variable references.

from __future__ import annotations

from nodeType import NodeType

import random

from variable import Variable

class Node:
    def __init__(self, type: NodeType, value: float = None, variable: Variable = None, funcName: str = ""):
        self.type = type
        self.value = value
        self.variable = variable
        self.funcName = funcName

    def isFunctionNode(self):
        return self.type == NodeType.FUNCTIONNODE
    
    def isTerminalNode(self):
        return self.type == NodeType.TERMINALNODE

    def __str__(self) -> str:
        if self.variable is not None:
            return f" {self.variable} "
        if self.value is None:
            return f"{self.type.name}:None"

        if self.isFunctionNode():
            formatted = f"{self.value:.2f}"
            return f"{self.funcName}({formatted})"

        formatted = f"{self.value:.2f}"
        return formatted

    @staticmethod
    def createRandomFunctionNode(minVal: float = 0, maxVal: float = 2, rng: random.Random = None)  -> Node:
        rng = rng if rng is not None else random
        value = rng.uniform(minVal, maxVal)
        return Node(NodeType.FUNCTIONNODE, value)
    
    @staticmethod
    def createRandomTerminalNode(minVal: float = 0, maxVal: float = 2, variableList: list = None, variableProb: float = 0.5, rng: random.Random = None) -> Node:
        rng = rng if rng is not None else random
        value = None
        variable = None
        if variableList is not None and rng.random() < variableProb:
            variable = rng.choice(variableList)
        else:
            value = rng.uniform(minVal, maxVal)
        return Node(NodeType.TERMINALNODE, value, variable)