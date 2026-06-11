# Data source abstractions used for fitness evaluation.
# Supports in-memory array data and MySQL-backed datasets.

from __future__ import annotations

import re
import numpy as np
import mysql.connector

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

class MysqlDataSource(DataSource):
    def __init__(self, host: str, user: str, password: str, database: str, table: str, columnNames: list, primaryKey: str, targetColumn: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.table = table
        self.columnNames = columnNames
        self.primaryKey = primaryKey
        self.targetColumn = targetColumn
        self.connection = None
        self.localSaved = False

        self.Rows = []

        self.connect()

    def close(self):
        if self.connection is not None and self.connection.is_connected():
            self.connection.close()

    def _validate_identifier(self, name: str) -> str:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError(f"Invalid identifier: {name}")
        return name

    def connect(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
    def isConnected(self) -> bool:
        return self.connection is not None and self.connection.is_connected()
    
    def saveRowsLocally(self) -> list:
        if not self.isConnected():
            raise Exception("Not connected to the database")

        cursor = self.connection.cursor()
        # Validate identifiers to avoid injection via table/column names
        self._validate_identifier(self.table)
        self._validate_identifier(self.primaryKey)
        self._validate_identifier(self.targetColumn)
        for col in self.columnNames:
            self._validate_identifier(col)

        cols = ", ".join(self.columnNames)
        query = f"SELECT {self.primaryKey}, {cols}, {self.targetColumn} FROM {self.table}"
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        self.Rows = rows
        self.localSaved = True
    
    def getRow(self, index: int) -> list:
        if not self.localSaved:
            cursor = self.connection.cursor()
            # Validate identifiers and parameterize values
            self._validate_identifier(self.table)
            self._validate_identifier(self.primaryKey)
            query = f"SELECT * FROM {self.table} WHERE {self.primaryKey} = %s"
            cursor.execute(query, (index,))
            row = cursor.fetchone()
            cursor.close()
            return row
        else:
            return self.Rows[index]
        
    def createVariableList(self) -> list:
        return [Variable(name) for name in self.columnNames]
