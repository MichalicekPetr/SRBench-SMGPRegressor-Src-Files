# Fitness evaluation functions for the genetic programming algorithm.
# Provides MSE-based fitness scoring for both vector and tree individuals.

from __future__ import annotations

from individual import Individual
from data import DataSource
from smoothMultifunctionSet import SmoothMultifunctionSet
from variable import Variable


class MeanSquaredErrorFitnessFunctionVector:
    def evaluateFitness(self, individual: Individual, dataSource: DataSource, dataIndexes: list, 
                        fList: list, variableList: list) -> float:
        if dataSource is None:
            raise Exception("Data source is required for fitness evaluation")

        totalError = 0.0
        numRows = len(dataIndexes)

        for i in range(numRows):
            row = dataSource.getRow(dataIndexes[i])
            inputValues = row[1:-1]
            targetValue = row[-1]
            Variable.setVariableValues(variableList, inputValues)   
            predictedValue = individual.evaluate(fList=fList, varList=variableList)
            error = (predictedValue - targetValue) ** 2
            totalError += error

        meanSquaredError = totalError / numRows

        if meanSquaredError == 0:
            print("Perfect fit found!")
            fitnessValue = float('inf')  # Perfect fit
        else:
            fitnessValue = 1 / meanSquaredError  # Inverse of MSE as fitness

        return fitnessValue

class MeanSquaredErrorFitnessFunction:
    def evaluateFitness(self, individual: Individual, dataSource: DataSource, dataIndexes: list, 
                        smoothmultifunctionset: SmoothMultifunctionSet, variableList: list) -> float:
        if dataSource is None:
            raise Exception("Data source is required for fitness evaluation")

        totalError = 0.0
        numRows = len(dataIndexes)

        for i in range(numRows):
            row = dataSource.getRow(dataIndexes[i])
            inputValues = row[1:-1]
            targetValue = row[-1]
            Variable.setVariableValues(variableList, inputValues)   
            predictedValue = individual.evaluate(smoothmultifunctionset)
            error = (predictedValue - targetValue) ** 2
            totalError += error

        meanSquaredError = totalError / numRows

        if meanSquaredError == 0:
            print("Perfect fit found!")
            fitnessValue = float('inf')  # Perfect fit
        else:
            fitnessValue = 1 / meanSquaredError  # Inverse of MSE as fitness

        return fitnessValue