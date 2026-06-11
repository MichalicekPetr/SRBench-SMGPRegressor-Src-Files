# Evolution strategy for constant tuning within the GP pipeline.
# This module performs local search on constant values inside individuals.

from __future__ import annotations

from copy import deepcopy
from math import floor, isinf
import random
from typing import Callable, Any
import time

from data import DataSource
from population import Population
from smoothMultifunctionSet import SmoothMultifunctionSet


class EvolutionStrategy:
    def __init__(self, smoothMultifunctionSet: SmoothMultifunctionSet, dataSource: DataSource = None, fitnessFunction: Any = None, 
                 selectionMethod: Callable = None, mutationFunc: Callable = None, crossoverFunc: Callable = None, 
                 variableList: list = None, dataIndexes: list = None, rng: random.Random = None):
        self.fitnessFunction = fitnessFunction
        self.selectionMethod = selectionMethod
        self.mutationFunc = mutationFunc
        self.crossoverFunc = crossoverFunc
        self.population = None
        self.smoothMultifunctionSet = smoothMultifunctionSet
        self.dataSource = dataSource
        self.variableList = variableList
        self.dataIndexes = dataIndexes
        self.rng = rng if rng is not None else random.Random()

    def tuneConstants(self, individualList: list, maxGenerations: int = None, maxTimeSeconds: float = None,
                      minTerminalValue: float = 0, maxTerminalValue: float = 10, originalIndividualRatio: float = 0.5,
                      randomIndividualRatio: float = 0.5, crossoverRate: float = 0.5, mutationRate: float = 0.1,
                      populationSize: int = 100) -> tuple | None:
        
        print("Starting constant tuning...")

        if maxGenerations is None and maxTimeSeconds is None:
            raise ValueError("At least one of maxGenerations or maxTimeSeconds must be specified.")
        
        if self.variableList is None:
            self.variableList = self.dataSource.createVariableList()

        fitness_improvements = []
        for individual in individualList:
            #print(f"Tuning individual:\n{individual.getVerticalTreeString()}")
            constantNodesVector = individual.getConstantNodesVector()
            original_fitness = self.fitnessFunction.evaluateFitness(individual = individual, dataSource = self.dataSource, dataIndexes = self.dataIndexes, smoothmultifunctionset = self.smoothMultifunctionSet, variableList = self.variableList)
            #print(f"Original constant values: {[node.value for node in constantNodesVector]}")

            if not constantNodesVector:
                #print("No constant nodes found for this individual; skipping constant tuning.")
                continue
            originalVector = [node.value for node in constantNodesVector]
            startingIndividualList = EvolutionStrategy.createRandomIndividualsForEvolutionStrategy(originalVector, populationSize = populationSize, 
                                                                         minVal = minTerminalValue, maxVal = maxTerminalValue , 
                                                                         originalIndividualRatio = originalIndividualRatio, rng=self.rng)
            self.population = Population(self.smoothMultifunctionSet)
            self.population.individualList = startingIndividualList

            generationNum = 0
            startTime = time.time()
            best_fitness = float('-inf')
            best_vector = None
            perfect_found = False
            while True:
                if maxGenerations is not None and generationNum >= maxGenerations:
                    #print(f"Reached maximum generations ({maxGenerations}). Stopping evolution.")
                    break
                if maxTimeSeconds is not None and generationNum > 0:
                    elapsedTime = time.time() - startTime
                    if elapsedTime >= maxTimeSeconds:
                        #print(f"Reached maximum time limit ({maxTimeSeconds} seconds). Stopping evolution.")
                       # print(f"Generation number: {generationNum}")
                        break
                generationNum += 1
                #print(f"--- Tuning constants, generation {generationNum} ---")
                fitnessValues = []

                for vectorIndividual in self.population.individualList:
                    for i, node in enumerate(constantNodesVector):
                        node.value = vectorIndividual[i]
                    fitness = self.fitnessFunction.evaluateFitness(individual = individual, 
                                                              dataSource = self.dataSource, dataIndexes = self.dataIndexes, 
                                                              smoothmultifunctionset = self.smoothMultifunctionSet, variableList = self.variableList)
                    fitnessValues.append(fitness)
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_vector = vectorIndividual.copy()
                    if isinf(fitness):
                        perfect_found = True
                        break

                # Summary statistics for fitness
                if fitnessValues:
                    max_fitness = max(fitnessValues)
                    min_fitness = min(fitnessValues)
                    avg_fitness = sum(fitnessValues) / float(len(fitnessValues))
                    #print(f"Fitness summary — max: {max_fitness:.4f}, min: {min_fitness:.4f}, avg: {avg_fitness:.4f}")
                    totalFitness = sum(fitnessValues)

                if perfect_found:
                    print(f"Perfect fit found in evolution strategy at generation {generationNum}.")
                    break

                # Selection and reproduction
                newPopulation = []
                while len(newPopulation) < len(self.population.individualList):
                    if self.rng.random() < randomIndividualRatio:
                        # Add a random individual
                        newIndividual = [self.rng.uniform(minTerminalValue, maxTerminalValue) for _ in range(len(originalVector))]
                        newPopulation.append(newIndividual)
                    else:
                        # Select two parents based on fitness
                        parent1 = self.selectionMethod(self.population, fitnessValues, totalFitness, rng=self.rng)
                        parent2 = self.selectionMethod(self.population, fitnessValues, totalFitness, rng=self.rng)

                        # Crossover
                        if self.rng.random() < crossoverRate:
                            child1, child2 = self.crossoverFunc(parent1, parent2, rng=self.rng)
                            newPopulation.extend([child1, child2])
                        else:
                            newPopulation.extend([parent1, parent2])

                
                # Mutation
                for i in range(len(newPopulation)): 
                    self.mutationFunc(vectorIndividual = newPopulation[i], mutationRate = mutationRate, 
                                      minTerminalValue = minTerminalValue, maxTerminalValue = maxTerminalValue, rng=self.rng)
                    
                self.population.individualList = newPopulation[:len(self.population.individualList)]  # Ensure population size is maintained

            if best_vector is not None:
                for i, node in enumerate(constantNodesVector):
                    node.value = best_vector[i]

            final_fitness = self.fitnessFunction.evaluateFitness(individual = individual, dataSource = self.dataSource, dataIndexes = self.dataIndexes, smoothmultifunctionset = self.smoothMultifunctionSet, variableList = self.variableList)
            improvement = ((final_fitness - original_fitness) / original_fitness) * 100 if original_fitness != 0 else 0.0
            fitness_improvements.append(improvement)

           # print(f"Finished tuning constants for individual:\n{individual.getVerticalTreeString()}")
           # print(f"Best constant values found: {[node.value for node in constantNodesVector]}")
           # print(f"Original fitness: {original_fitness:.4f}, Final fitness: {final_fitness:.4f}, Improvement: {improvement:.4f}")
            if perfect_found:
                average_improvement = sum(fitness_improvements) / len(fitness_improvements) if fitness_improvements else 0.0
                print(f"Average fitness improvement after constant tuning so far: {average_improvement:.4f}")
                return individual, best_fitness

        if fitness_improvements:
            average_improvement = sum(fitness_improvements) / len(fitness_improvements)
            print(f"Average fitness improvement after constant tuning: {average_improvement:.4f}%")
        else:
            print("No constant tuning improvements were calculated.")

        return None

    @staticmethod 
    def createRandomIndividualsForEvolutionStrategy(originalConstantNodesVector: list, populationSize: int = 100,
                                                   minVal: float = 0, maxVal: float = 10, 
                                                   originalIndividualRatio: float = 0.5, rng: random.Random = None) -> list:
        rng = rng if rng is not None else random
        population = []
        individualLength = len(originalConstantNodesVector)
        if len(originalConstantNodesVector) == 0:
            return population
        
        for _ in range(floor(populationSize * originalIndividualRatio)):
            newIndividual = deepcopy(originalConstantNodesVector)
            population.append(newIndividual)
        for _ in range(populationSize - floor(populationSize * originalIndividualRatio)):
            newIndividual = []
            for _ in range(individualLength):
                newIndividual.append(rng.uniform(minVal, maxVal))
            population.append(newIndividual)

        return population
        


    

 