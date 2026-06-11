# Population container for GP individuals.
# Manages initialization and storage of the population.

from __future__ import annotations

import random
from individual import Individual
from smoothMultifunctionSet import SmoothMultifunctionSet

class Population:
    def __init__(self, smoothMultifunctionSet: SmoothMultifunctionSet = None):
        self.individualList = []
        self.smoothMultifunctionSet = smoothMultifunctionSet

    def initializePopulationFullRandomMethod(self, populationSize: int = 0, depth: int = 0, funcList: list = [],
                                             variableList: list = None, variableProbability: float = 0.5,
                                             minTerminalNodeVal = 0, maxTerminalNodeVal = 10, rng: random.Random = None) -> None:
        for _ in range(populationSize):
            newRandomIndividual = Individual.createRandomIndividual(depth = depth, variableList= variableList, variableProbability=variableProbability,
                                                                    minTerminalNodeVal=minTerminalNodeVal, maxTerminalNodeVal=maxTerminalNodeVal, functionList=funcList, rng=rng)
            self.individualList.append(newRandomIndividual)

