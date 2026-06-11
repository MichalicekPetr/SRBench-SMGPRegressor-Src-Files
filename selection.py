# Selection methods for choosing individuals during evolution.
# Includes roulette wheel selection and other selection strategies.

from __future__ import annotations

from population import Population
import random

class Selection:
    @staticmethod
    def rouletteSelection(population: Population, fitnessValues: list, totalFitness: float = None, rng: random.Random = None):
        
        rng = rng if rng is not None else random
        totalFitness = sum(fitnessValues)
        if totalFitness == 0:
            return population.individualList[0]

        selectionProbabilities = [fitness / totalFitness for fitness in fitnessValues]
        selectedIndividual = rng.choices(population.individualList, weights=selectionProbabilities, k=1)[0]
        return selectedIndividual