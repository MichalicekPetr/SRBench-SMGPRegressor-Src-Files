# Data source abstractions used for fitness evaluation.
# Supports in-memory array data and MySQL-backed datasets.

from __future__ import annotations

import numpy as np

from variable import Variable

class DataSource:
    def getRow(self, index: int) -> list:
        pass

    def createVariableList(self) -> list:
        pass

class ArrayDataSource(DataSource):
    def __init__(self, X, y, columnNames: list):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.columnNames = list(columnNames)
        self.localSaved = True

        if self.X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if self.y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

    def getRow(self, index: int) -> list:
        if index < 0 or index >= self.X.shape[0]:
            raise IndexError("Index out of range")
        return [index] + self.X[index].tolist() + [self.y[index]]

    def createVariableList(self) -> list:
        return [Variable(name) for name in self.columnNames]