# Core evolution engine for the GP implementation.
# Contains the main VectorEvolutionAlgorithm and EvolutionAlgorithm classes.

from __future__ import annotations

from data import DataSource
from population import Population
from smoothMultifunctionSet import SmoothMultifunctionSet
from typing import Any, Callable
from individual import Individual
from evolutionStrategy import EvolutionStrategy

import math
import random
import time
from copy import deepcopy

class VectorEvolutionAlgorithm:
    """A vector-based genetic programming evolution engine.

    Uses flat vector individuals, crossover, and mutation operators to evolve
    a population of candidate solutions.
    """

    def __init__(self, fList: list, dataSource: DataSource = None, fitnessFunction = None, 
                 dataIndexes: list = None, mutationFunc: Callable = None, crossoverFunc: Callable = None,
                 rng: random.Random = None, taylorSumElements: int = 100, useTriangleFval: bool = True):
        """Initialize vector evolution settings and propagate function set parameters."""
        self.population = Population()
        self.fList = fList
        self.fitnessValues = []
        self.dataSource = dataSource
        self.variableList = None
        self.fitnessFunction = fitnessFunction
        self.dataIndexes = dataIndexes
        self.mutationFunc = mutationFunc
        self.crossoverFunc = crossoverFunc
        self.rng = rng if rng is not None else random.Random()
        self.taylorSumElements = taylorSumElements
        self.useTriangleFval = useTriangleFval
        for func in self.fList:
            if hasattr(func, "taylorSumElements"):
                func.taylorSumElements = taylorSumElements
            if hasattr(func, "useTriangleFval"):
                func.useTriangleFval = useTriangleFval

        for func in self.fList:
            if hasattr(func, "useTriangleFval") and func.useTriangleFval != useTriangleFval:
                raise ValueError(
                    f"Unable to configure useTriangleFval={useTriangleFval} on function set {func}"
                )

    def initFitnessValues(self, populationSize: int = 0):
        """Initialize the fitness value list for the population."""
        self.fitnessValues = [-1] * populationSize

    def runEvolution(self, maxGenerations: int = 100, populationSize: int = 100, depth: int = 4, mutationRate: float = 0.01,
                randomIndividualRate: float = 0.1, variableProbability: float = 0.5,
                minTerminalNodeVal: float = 0, maxTerminalNodeVal: float = 10, maxSeconds: float | None = None) -> Individual | None:
        """Run the vector evolution loop and return the best found individual."""
        self.variableList = self.dataSource.createVariableList()
        self.population.initializePopulationFullRandomMethod(populationSize = populationSize, depth = depth, funcList= self.fList,
                                                             variableList = self.variableList, variableProbability = variableProbability,
                                                             minTerminalNodeVal = minTerminalNodeVal, maxTerminalNodeVal = maxTerminalNodeVal,
                                                             rng=self.rng)
        self.initFitnessValues(populationSize = populationSize)

        best_fitness = float('-inf')
        best_individual = None
        best_generation = None
        start_time = time.monotonic() if maxSeconds is not None else None

        for generation in range(maxGenerations):
            #print("------------------------ Generation: " + str(generation) + " ------------------------")
            
            # Fitness evaluation
            for i in range(populationSize):
                self.fitnessValues[i] = self.fitnessFunction.evaluateFitness(individual = self.population.individualList[i], 
                        dataSource = self.dataSource, dataIndexes = self.dataIndexes, fList = self.fList, variableList = self.variableList)

            # Summary statistics for fitness
            if populationSize > 0:
                max_fitness = max(self.fitnessValues)
                min_fitness = min(self.fitnessValues)
                avg_fitness = sum(self.fitnessValues) / float(populationSize)
               # print(f"Fitness summary — max: {max_fitness:.4f}, min: {min_fitness:.4f}, avg: {avg_fitness:.4f}")

                # Update best individual seen so far
                try:
                    best_idx = self.fitnessValues.index(max_fitness)
                    candidate = deepcopy(self.population.individualList[best_idx])
                    if max_fitness > best_fitness:
                        best_fitness = max_fitness
                        best_individual = candidate
                        best_generation = generation
                    if math.isinf(max_fitness):
                       # print(f"Perfect fit found in GP at generation {generation}. Stopping evolution.")
                        return candidate
                except Exception:
                    pass

            # Selection
            # === OPTIMIZED SELECTION ===
            newIndividualList = []
            newFitnessValues = [] # Auxiliary array to maintain the fitness of the new generation
            sortedIndexes = sorted(range(populationSize), key=lambda k: self.fitnessValues[k], reverse=True)

            for i in range(0, populationSize):
                idx1 = sortedIndexes[i] 
                idx2 = sortedIndexes[(i+1) % populationSize]  
                original = self.population.individualList[idx1]
                partner = self.population.individualList[idx2]
                new = self.crossoverFunc(ind1=original, ind2=partner, fset=self.fList, varlist=self.variableList, 
                                         minTerminalVal=minTerminalNodeVal, maxTerminalVal=maxTerminalNodeVal, rng=self.rng)
                
                scoreOriginal = self.fitnessValues[idx1]
                scoreNew = self.fitnessFunction.evaluateFitness(individual = new, 
                        dataSource = self.dataSource, dataIndexes = self.dataIndexes, fList = self.fList, variableList = self.variableList)
                
                if scoreNew >= scoreOriginal:
                    newIndividualList.append(new)
                    newFitnessValues.append(scoreNew) # We know exactly what the fitness is
                else:
                    newIndividualList.append(original)
                    newFitnessValues.append(scoreOriginal) # We know exactly what the fitness is

            # Injection of random individuals (their fitness is not yet known, set to -1)
            randomIndividualCount = int(round(randomIndividualRate * populationSize))
            for idx in range(randomIndividualCount):
                pos = max(0, populationSize - 1 - idx)
                new = Individual.createRandomIndividual(
                    depth=depth,
                    variableList=self.variableList,
                    variableProbability=variableProbability,
                    functionList=self.fList,
                    minTerminalNodeVal=minTerminalNodeVal,
                    maxTerminalNodeVal=maxTerminalNodeVal,
                    rng=self.rng,
                )
                newIndividualList[pos] = new
                newFitnessValues[pos] = -1 # Flag indicating that fitness must be evaluated

            # === OPTIMIZED MUTATION ===
            for i in range(populationSize):
                mutated = deepcopy(newIndividualList[i])
                self.mutationFunc(individual = mutated, mutationRate = mutationRate, fList = self.fList,
                             minTerminalVal = minTerminalNodeVal, maxTerminalVal = maxTerminalNodeVal,
                             varlist = self.variableList, variableProbability = variableProbability, rng=self.rng)
                
                # If it is a random individual from the end of the array, we must first evaluate its base fitness
                if newFitnessValues[i] == -1:
                    originalFitness = self.fitnessFunction.evaluateFitness(individual = newIndividualList[i], 
                            dataSource = self.dataSource, dataIndexes = self.dataIndexes, fList = self.fList, variableList = self.variableList)
                    newFitnessValues[i] = originalFitness
                else:
                    originalFitness = newFitnessValues[i] # <--- Time saved here! No function call, retrieved from the array.
                
                mutatedFitness = self.fitnessFunction.evaluateFitness(individual = mutated, 
                        dataSource = self.dataSource, dataIndexes = self.dataIndexes, fList = self.fList, variableList = self.variableList)
                
                if mutatedFitness > originalFitness:
                    newIndividualList[i] = mutated
                    newFitnessValues[i] = mutatedFitness # Update the value for potential subsequent generations

            self.population.individualList = newIndividualList

            if maxSeconds is not None and start_time is not None and (time.monotonic() - start_time) >= maxSeconds:
                #print("Stopping evolution because maxSeconds was exceeded.")
                break

        # After completing all generations, return the best individual
        if best_individual is not None:
            #print("\n=== Best individual found ===")
            #print(f"Generation: {best_generation}, Fitness: {best_fitness:.6f}")
            try:
                best_individual.printVerticalTree()
            except Exception:
                print(str(best_individual))
        return best_individual


class EvolutionAlgorithm:
    """A tree-based genetic programming evolution engine.

    Uses function-tree individuals with selectable selection, mutation, and crossover.
    """

    def __init__(self, smoothMultifunctionSet: SmoothMultifunctionSet, dataSource: DataSource = None, fitnessFunction: Any = None, 
                 dataIndexes: list = None,  selectionMethod: Callable = None, mutationFunc: Callable = None, crossoverFunc: Callable = None,
                 rng: random.Random = None, taylorSumElements: int = 100, useTriangleFval: bool = True):
        """Initialize tree-based evolution settings and configure the function set."""
        self.population = Population(smoothMultifunctionSet)
        self.fitnessValues = []
        self.dataSource = dataSource
        self.variableList = None
        self.fitnessFunction = fitnessFunction
        self.dataIndexes = dataIndexes
        self.selectionMethod = selectionMethod
        self.mutationFunc = mutationFunc
        self.crossoverFunc = crossoverFunc
        self.rng = rng if rng is not None else random.Random()
        self.evolutionStrategy = None
        self.taylorSumElements = taylorSumElements
        self.useTriangleFval = useTriangleFval
        if smoothMultifunctionSet is not None:
            smoothMultifunctionSet.taylorSumElements = taylorSumElements
            smoothMultifunctionSet.useTriangleFval = useTriangleFval

        self.esParamsSet = False
        self.esmaxseconds = None
        self.esmaxgenerations = None
        self.escrossoverRate = None
        self.esmutationRate = None
        self.esrandomIndividualRate = None
        self.esoriginalIndividualRatio = None
        self.esPopulationSize = None

    def initEvolutionStrategy(self, evolutionStrategy: EvolutionStrategy):
        """Configure an existing evolution strategy instance."""
        self.evolutionStrategy = evolutionStrategy

    def initEvolutionStrategy(self, fitnessFunction: Any, selectionMethod: Callable, mutationMethod: Callable, crossoverMethod: Callable):
        """Create and configure a default EvolutionStrategy instance."""
        self.evolutionStrategy = EvolutionStrategy(fitnessFunction = fitnessFunction, selectionMethod = selectionMethod, 
                                                   mutationFunc = mutationMethod, crossoverFunc = crossoverMethod, 
                                                   dataSource = self.dataSource, dataIndexes = self.dataIndexes,
                                                   smoothMultifunctionSet = self.population.smoothMultifunctionSet, variableList = self.variableList,
                                                   rng=self.rng)
        
    def initFitnessValues(self, populationSize: int = 0):
        """Initialize the fitness value list for the population."""
        self.fitnessValues = [-1] * populationSize

    def setEvolutionStrategyParameters(self, crossoverRate: float = 0.8, mutationRate: float = 0.1,
                     randomIndividualRate: float = 0.1, maxSeconds: float = None, maxGenerations: int = None, 
                     originalIndividualRatio: float = 0.5, populationSize: int = 100) -> None:
        """Set parameters for the evolution strategy before running evolution."""
        self.esmaxseconds = maxSeconds
        self.esmaxgenerations = maxGenerations
        self.escrossoverRate = crossoverRate
        self.esmutationRate = mutationRate
        self.esrandomIndividualRate = randomIndividualRate
        self.esoriginalIndividualRatio = originalIndividualRatio
        self.esParamsSet = True
        self.esPopulationSize = populationSize

    def runEvolution(self, maxGenerations: int = 100, populationSize: int = 100, depth: int = 4, 
                     crossoverRate: float = 0.8, mutationRate: float = 0.1,
                     randomIndividualRate: float = 0.1, variableProbability: float = 0.5,
                     minTerminalNodeVal: float = 0, maxTerminalNodeVal: float = 10, tuneConstants: bool = False) -> Individual | None:
        """Run the evolution loop for tree-based individuals and return the best found solution."""
        self.variableList = self.dataSource.createVariableList()
        self.population.initializePopulationFullRandomMethod(populationSize = populationSize, depth = depth, 
                                                             variableList = self.variableList, variableProbability = variableProbability,
                                                             minTerminalNodeVal = minTerminalNodeVal, maxTerminalNodeVal = maxTerminalNodeVal,
                                                             rng=self.rng)
        self.initFitnessValues(populationSize = populationSize)

        best_fitness = float('-inf')
        best_individual = None
        best_generation = None

        for generation in range(maxGenerations):
           #print("------------------------ Generation: " + str(generation) + " ------------------------")
            
            # Fitness evaluation
            for i in range(populationSize):
                self.fitnessValues[i] = self.fitnessFunction.evaluateFitness(individual = self.population.individualList[i], 
                        dataSource = self.dataSource, dataIndexes = self.dataIndexes, 
                        smoothmultifunctionset = self.population.smoothMultifunctionSet, variableList = self.variableList)

            # Summary statistics for fitness
            if populationSize > 0:
                max_fitness = max(self.fitnessValues)
                min_fitness = min(self.fitnessValues)
                avg_fitness = sum(self.fitnessValues) / float(populationSize)
                print(f"Fitness summary — max: {max_fitness:.4f}, min: {min_fitness:.4f}, avg: {avg_fitness:.4f}")

                # Update best individual seen so far
                try:
                    best_idx = self.fitnessValues.index(max_fitness)
                    candidate = deepcopy(self.population.individualList[best_idx])
                    if max_fitness > best_fitness:
                        best_fitness = max_fitness
                        best_individual = candidate
                        best_generation = generation
                    if math.isinf(max_fitness):
                        print(f"Perfect fit found in GP at generation {generation}. Stopping evolution.")
                        return candidate
                except Exception:
                    pass

            # Selection
            newIndividualList = []
            totalFitness = sum(self.fitnessValues)

            while len(newIndividualList) < populationSize:
                if self.rng.random() < randomIndividualRate:
                    newIndividualList.append(Individual.createRandomIndividual(self.population.smoothMultifunctionSet, depth, 
                                                                         variableList = self.variableList, variableProbability = variableProbability, 
                                                                         minTerminalNodeVal = minTerminalNodeVal, maxTerminalNodeVal = maxTerminalNodeVal,
                                                                         rng=self.rng))
                else:
                    if self.rng.random() < crossoverRate:
                        selectedIndividual1 = self.selectionMethod(self.population, self.fitnessValues, totalFitness, rng=self.rng)
                        selectedIndividual2 = self.selectionMethod(self.population, self.fitnessValues, totalFitness, rng=self.rng)
                        newIndividual1, newIndividual2 = self.crossoverFunc(selectedIndividual1, selectedIndividual2, rng=self.rng)
                        newIndividualList.append(newIndividual1)
                        if len(newIndividualList) < populationSize:
                            newIndividualList.append(newIndividual2)
                    else:
                        selectedIndividual = self.selectionMethod(self.population, self.fitnessValues, totalFitness, rng=self.rng)
                        newIndividual = deepcopy(selectedIndividual)
                        newIndividualList.append(newIndividual)

            # Mutation
            for i in range(populationSize):
                self.mutationFunc(individual = newIndividualList[i], mutationRate = mutationRate, 
                             minFunctionNodeVal = self.population.smoothMultifunctionSet.minVal, 
                             maxFunctionNodeVal = self.population.smoothMultifunctionSet.maxVal,
                             minTerminalNodeVal = minTerminalNodeVal, maxTerminalNodeVal = maxTerminalNodeVal,
                             variableList = self.variableList, variableProbability = variableProbability, rng=self.rng)
            
            if tuneConstants:
                if self.esParamsSet:
                    result = self.evolutionStrategy.tuneConstants(individualList = newIndividualList, maxGenerations = self.esmaxgenerations, maxTimeSeconds = self.esmaxseconds, 
                                                                    minTerminalValue = minTerminalNodeVal, maxTerminalValue = maxTerminalNodeVal, 
                                                                    originalIndividualRatio = self.esoriginalIndividualRatio, randomIndividualRatio = self.esrandomIndividualRate, 
                                                                    crossoverRate = self.escrossoverRate, mutationRate = self.esmutationRate, populationSize = self.esPopulationSize)
                else:
                    result = self.evolutionStrategy.tuneConstants(individualList = newIndividualList, maxGenerations = 100, maxTimeSeconds = 2, 
                                                                    minTerminalValue = minTerminalNodeVal, maxTerminalValue = maxTerminalNodeVal, 
                                                                    originalIndividualRatio = 0.5, randomIndividualRatio = 0.5, crossoverRate = 0.5)
                if result is not None:
                    tuned_individual, tuned_fitness = result
                    if math.isinf(tuned_fitness):
                        print("Perfect fit found during constant tuning; stopping GP.")
                        return tuned_individual

            self.population.individualList = newIndividualList

        # After completing all generations, print the best individual
        if best_individual is not None:
            print("\n=== Best individual found ===")
            print(f"Generation: {best_generation}, Fitness: {best_fitness:.6f}")
            try:
                best_individual.printVerticalTree()
            except Exception:
                print(str(best_individual))
        return best_individual