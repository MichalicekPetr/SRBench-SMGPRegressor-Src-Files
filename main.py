# Example entrypoint and script for running the genetic programming model.
# Demonstrates how to configure data sources, function sets, and the main evolution loop.

from __future__ import annotations

from crossover import Crossover
from evolution import EvolutionAlgorithm, VectorEvolutionAlgorithm
from fitness import MeanSquaredErrorFitnessFunction, MeanSquaredErrorFitnessFunctionVector
from individual import Individual
from mutation import Mutation
from smoothMultifunctionSet import SmoothMultifunctionSet
from data import MysqlDataSource
from variable import Variable
from selection import Selection
import random
import numpy as np
import pandas as pd
from regressor import SMGPRegressor, model

def run_srbench_integration_test():
    print("=" * 60)
    print("STARTING SRBENCH INTEGRATION TEST FOR SMGPREGRESSOR")
    print("=" * 60)

    # 1. GENEROVÁNÍ DAT (Přesně tak, jak je posílá SRBench - Pandas DataFrame)
    print("\n[1/5] Generating synthetic benchmark data...")
    np.random.seed(42)
    n_samples = 150
    
    # Vytvoříme 3 proměnné s reálnými názvy
    X_df = pd.DataFrame({
        "temperature": np.random.uniform(-2.0, 2.0, n_samples),
        "pressure": np.random.uniform(0.5, 5.0, n_samples),
        "humidity": np.random.uniform(10.0, 90.0, n_samples)
    })
    
    # Cílová funkce (target), kterou by měl algoritmus aproximovat
    # Použijeme kombinaci operací z našeho registru
    y_true = (X_df["temperature"] * X_df["pressure"]) + np.sin(X_df["temperature"]) - (X_df["humidity"] / 20.0)
    # Přidáme drobný šum, jak je v bencharcích zvykem
    y_arr = y_true + np.random.normal(0, 0.05, n_samples)

    print(f"-> Generated {n_samples} samples with features: {list(X_df.columns)}")

    # 2. INICIALIZACE REGRESORU
    print("\n[2/5] Initializing SMGPRegressor...")
    # Použijeme náš doporučený default set (Basic), hloubku 5 a rychlého Taylora (5 elementů)
    regressor = SMGPRegressor(
        random_state=42,
        max_time=30.0,            # Časový limit 30 sekund
        population_size=150,
        generations=100,          # Pro rychlý test stačí 100 generací
        depth=5,                  # Doporučená hloubka pro benchmarky
        mutation_rate=0.05,
        random_individual_rate=0.1,
        variable_probability=0.4,
        taylor_sum_elements=5,
        use_triangle_fval=True,
        verbose=True              # Chceme vidět průběh výpočtu
    )

    # 3. TRÉNOVÁNÍ (FIT)
    print("\n[3/5] Training the model via .fit()...")
    try:
        regressor.fit(X_df, y_arr)
        print("-> Training finished successfully!")
        print(f"-> Best Fitness achieved: {regressor.best_fitness_:.6g}")
    except Exception as e:
        print(f"!!! CRITICAL ERROR DURING FIT: {e}")
        return

    # 4. PŘEDPOVĚĎ (PREDICT)
    print("\n[4/5] Testing predictions via .predict()...")
    try:
        predictions = regressor.predict(X_df)
        print(f"-> Predictions shape: {predictions.shape}")
        mse = np.mean((predictions - y_arr) ** 2)
        print(f"-> Calculated Mean Squared Error on training data: {mse:.6g}")
        
        if np.isnan(predictions).any() or np.isinf(predictions).any():
            print("!!! WARNING: Predictions contain NaN or Inf values!")
        else:
            print("-> Prediction sanity check: PASSED (No NaNs/Infs)")
    except Exception as e:
        print(f"!!! CRITICAL ERROR DURING PREDICT: {e}")
        return

    # 5. EXPORT MODELU (SYM PY STRING)
    print("\n[5/5] Exporting model string for SymPy compatibility...")
    try:
        # Volání přesně tak, jak to dělá SRBench (předává model a DataFrame)
        raw_model_str = model(regressor)
        final_model_str = model(regressor, X=X_df)
        
        print(f"-> Raw model string (internal names): {raw_model_str}")
        print(f"-> Final SymPy model string (mapped names): {final_model_str}")
        
        # Kontrola, zda se správně nahradily proměnné
        if "x_0" in final_model_str or "x_1" in final_model_str:
            print("!!! WARNING: Variable mapping failed! Internal 'x_i' names are still present.")
        else:
            print("-> Variable mapping check: PASSED (Real column names successfully applied)")

        # Pokus o parsování v SymPy (pokud ho máš nainstalovaný)
        try:
            import sympy as sp
            parsed_expr = sp.parse_expr(final_model_str)
            print("-> SymPy Parsing: PASSED")
            print(f"-> SymPy Simplified: {sp.simplify(parsed_expr)}")
        except ImportError:
            print("-> SymPy library not installed locally, skipping text mathematical parsing validation.")
        except Exception as sympy_err:
            print(f"!!! SymPy Parsing FAILED: {sympy_err}")
            print("Make sure your operators are strictly written in python standard (e.g. log, sin, **).")

    except Exception as e:
        print(f"!!! CRITICAL ERROR DURING MODEL EXPORT: {e}")
        return

    print("\n" + "=" * 60)
    print("INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":

    run_srbench_integration_test()

    """
    dataSource = MysqlDataSource(host="localhost", user="root", password="krtek", database="smoothmultifunctiongeneticprogrammingtestschema", 
                                 table="symbolicregressiontable1",  columnNames=["x", "y", "z"], primaryKey="id", targetColumn="result")
    variableList = dataSource.createVariableList()
    f = SmoothMultifunctionSet.createClassicMultifunctionSet()
    fset = [f] 
    indexes = list(range(1,101))

 
    i1 = Individual.createRandomIndividual(depth=3, variableList=variableList, variableProbability=0.4, functionList=fset, minTerminalNodeVal=0, maxTerminalNodeVal=10)
    i2 = Individual.createRandomIndividual(depth=3, variableList=variableList, variableProbability=0.4, functionList=fset, minTerminalNodeVal=0, maxTerminalNodeVal=10)

    print(i1)
    print(i2)

    i3 = Crossover.swappingPointCrossover(ind1=i1, ind2=i2, fset=fset, varlist=variableList, minTerminalVal=0, maxTerminalVal=10)
    
    print(i3)

    Mutation.vectorMutation(mutationRate=0.2, individual=i3, fset=fset, varlist=variableList, minTerminalVal=0, maxTerminalVal=10 )
    print(i3)

    fitness = MeanSquaredErrorFitnessFunctionVector().evaluateFitness(individual=i1, dataSource=dataSource, dataIndexes=indexes, fList=fset, variableList=variableList)
    print("Fitness: " + str(fitness))

    
    dataSource = MysqlDataSource(host="localhost", user="root", password="krtek", database="smoothmultifunctiongeneticprogrammingtestschema", 
                                 table="symbolicregressiontable1",  columnNames=["x", "y", "z"], primaryKey="id", targetColumn="result")
    variableList = dataSource.createVariableList()
    dataSource.saveRowsLocally()
    f = SmoothMultifunctionSet.createClassicMultifunctionSet()
    fset = [f] 
    indexes = list(range(0,100))


    random_seed = 42
    evolution = VectorEvolutionAlgorithm(fList=fset, dataSource=dataSource, fitnessFunction=MeanSquaredErrorFitnessFunctionVector(), 
                                   dataIndexes=indexes, mutationFunc=Mutation.vectorMutation, crossoverFunc=Crossover.betweenPointCrossover,
                                   rng=random.Random(random_seed), taylorSumElements=5, useTriangleFval=True)
    evolution.runEvolution(maxGenerations = 1000, populationSize = 200, depth = 5, mutationRate = 0.05, 
                           randomIndividualRate = 0.1, variableProbability = 0.4,
                           minTerminalNodeVal = 0, maxTerminalNodeVal = 10)

    exit(0)

    evolution = EvolutionAlgorithm(smoothMultifunctionSet=fset, dataSource=dataSource, fitnessFunction=MeanSquaredErrorFitnessFunction(), 
                                   dataIndexes=indexes, selectionMethod=Selection.rouletteSelection, mutationFunc=Mutation.nodeMutation, 
                                   crossoverFunc=Crossover.onePointCrossover)
    evolution.initEvolutionStrategy(fitnessFunction=MeanSquaredErrorFitnessFunction(), selectionMethod=Selection.rouletteSelection, 
                                   mutationMethod=Mutation.vectorOnePointMutation, crossoverMethod=Crossover.vectorOnePointCrossover)
    evolution.runEvolution(maxGenerations = 1000, populationSize = 100, depth = 4, 
                           crossoverRate = 0.4, mutationRate = 0.025, 
                           randomIndividualRate = 0.1, variableProbability = 0.4,
                           minTerminalNodeVal = 0, maxTerminalNodeVal = 10, tuneConstants = True)
"""