# Mutation operators for modifying individuals in the GP population.
# Implements node-level and vector-level mutation strategies.

from __future__ import annotations

from individual import Individual, PrintableIndividual
import random

class Mutation():
    """Mutation operators for GP individuals."""

    @staticmethod
    def vectorMutation(mutationRate: float, individual: Individual, fList: list, varlist: list, minTerminalVal: float, maxTerminalVal: float, variableProbability: float, rng: random.Random = None) -> None:
        """Mutate a flat vector individual with given mutation rate and terminals."""
        rng = rng if rng is not None else random
        nodeCnt = individual.vector.size

        for i in range(individual.vector.size):
            seed = rng.random()
            if seed < mutationRate:
                originVal = individual.vector[i]
                if i < ((nodeCnt / 2) - 1): # func node
                    if (i % 2) == 0: # Value
                        f = fList[int(individual.vector[i+1])]
                        newVal = rng.uniform(f.minVal, f.maxVal)
                        individual.vector[i] = newVal
                    else: # Type
                        newType = rng.randint(0, len(fList) - 1)
                        f = fList[newType]
                        newVal = rng.uniform(f.minVal, f.maxVal)
                        individual.vector[i] = newType
                        individual.vector[i - 1] = newVal

                else: # terminal node
                    if (i % 2) == 0:
                        type = int(individual.vector[i + 1])
                        if type == 0: #constant
                            newVal = rng.uniform(minTerminalVal, maxTerminalVal)
                            individual.vector[i] = newVal
                        else: # variable
                            newVal = rng.randint(0, len(varlist) - 1)
                            individual.vector[i] = newVal
                    else:
                        seed = rng.random()
                        if seed >= variableProbability:
                            newVal = rng.uniform(minTerminalVal, maxTerminalVal)
                            individual.vector[i - 1] = newVal
                            individual.vector[i] = 0
                        else:
                            newVal = rng.randint(0, len(varlist) - 1)
                            individual.vector[i - 1] = newVal
                            individual.vector[i] = 1


    @staticmethod
    def nodeMutation(individual: PrintableIndividual, mutationRate: float = 0.1,
               minFunctionNodeVal: float = 0, maxFunctionNodeVal: float = 2,
               minTerminalNodeVal: float = 0, maxTerminalNodeVal: float = 10, 
               variableList: list = None, variableProbability: float = 0.3, rng: random.Random = None) -> None:
        """Mutate a tree-based individual by changing nodes or terminals."""
        rng = rng if rng is not None else random
        originalIndividual = individual.getTreeString()
        for i in range(len(individual.nodes)):
            if rng.random() < mutationRate:
                node = individual.nodes[i]
                if node is not None:
                    if node.isFunctionNode():
                        new_value = rng.uniform(minFunctionNodeVal, maxFunctionNodeVal)
                        individual.nodes[i] = type(node)(node.type, new_value)
                    else:
                        if len(variableList) > 0 and rng.random() < variableProbability:
                            new_variable = rng.choice(variableList)
                            individual.nodes[i] = type(node)(node.type, None, new_variable)
                        else:
                            new_value = rng.uniform(minTerminalNodeVal, maxTerminalNodeVal)
                            individual.nodes[i] = type(node)(node.type, new_value, None)

    @staticmethod
    def vectorOnePointMutation(vectorIndividual: list, mutationRate: float = 0.1,
               minTerminalValue: float = 0, maxTerminalValue: float = 10, rng: random.Random = None) -> None:
        """Apply one-point mutation to a flat vector individual."""
        rng = rng if rng is not None else random
        for i in range(len(vectorIndividual)):
            if rng.random() < mutationRate:
                vectorIndividual[i] = rng.uniform(minTerminalValue, maxTerminalValue)