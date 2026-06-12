# Crossover operators for recombining parent individuals.
# These functions generate new child solutions by swapping genetic material.

from __future__ import annotations

from individual import Individual, PrintableIndividual
from helperfunc import get_declining_random_fast
import random
from copy import deepcopy

class Crossover():
    """Crossover operators for recombining GP individuals."""

    @staticmethod
    def swappingPointCrossover(ind1: Individual, ind2: Individual, fset: list, varlist: list, minTerminalVal: float, maxTerminalVal: float, rng: random.Random = None) -> Individual:
        """Create a new individual by swapping vector entries at random points."""
        rng = rng if rng is not None else random
        newInd = deepcopy(ind1)
        nodeCnt = ind1.vector.size
        pointCnt = get_declining_random_fast(n=nodeCnt, rate=18, rng=rng)
        indexes = list(range(0, nodeCnt))
        points = []

        while(pointCnt > 0):    
            seed = rng.randint(0, len(indexes) - 1)
            points.append(indexes[seed])
            indexes.pop(seed)
            pointCnt -= 1

        for point in points:
            newVal = ind2.vector[point]
            if point < ((nodeCnt / 2) - 1): #Function node
                if (point % 2) == 0: #Function value
                    funcIdx = int(newInd.vector[point + 1])
                    f = fset[funcIdx]
                    if newVal >= f.minVal and newVal <= f.maxVal:
                        newInd.vector[point] = newVal
                    else:
                        newInd.vector[point] = rng.uniform(f.minVal, f.maxVal)

                elif (point % 2) == 1: #Function type
                    f = fset[int(newVal)]
                    newInd.vector[point] = newVal
                    fval = newInd.vector[point - 1]
                    if fval < f.minVal or fval > f.maxVal:
                        newInd.vector[point - 1] = rng.uniform(f.minVal, f.maxVal)

            else: # Terminal node
                if (point % 2) == 0: # Terminal value
                    newType = ind2.vector[point + 1]
                    originType = newInd.vector[point + 1]
                    if newType == originType:
                        newInd.vector[point] = newVal
                    else:
                        if originType == 0: # Constant
                            newInd.vector[point] = rng.uniform(minTerminalVal, maxTerminalVal)
                        if originType == 1: # Varible
                            newInd.vector[point] = rng.randint(0, len(varlist) - 1)

                elif (point % 2) == 1: # Terminal type
                    originType = newInd.vector[point]
                    if originType != newVal:
                        newInd.vector[point] = newVal
                        if newVal == 0: #constant
                            newInd.vector[point - 1] = rng.uniform(minTerminalVal, maxTerminalVal)
                        elif newVal == 1: #variable 
                            newInd.vector[point - 1] = rng.randint(0, len(varlist) - 1)

        return newInd
    
    @staticmethod
    def betweenPointCrossover(ind1: Individual, ind2: Individual, fset: list, varlist: list, minTerminalVal: float, maxTerminalVal: float, rng: random.Random = None, percent: float = 10.0) -> Individual:
        """Create a new individual by swapping vector entries at random points."""
        rng = rng if rng is not None else random
        newInd = deepcopy(ind1)
        nodeCnt = ind1.vector.size
        pointCnt = get_declining_random_fast(n=nodeCnt, rate=67, rng=rng)
        indexes = list(range(0, nodeCnt))
        points = []

        def sample_in_interval(origin_val: float, candidate_val: float, lower_bound: float, upper_bound: float) -> float:
            lower = min(origin_val, candidate_val) * (1 - (percent / 100.0))
            upper = max(origin_val, candidate_val) * (1 + (percent / 100.0))
            lower = max(lower_bound, lower)
            upper = min(upper_bound, upper)
            if lower >= upper:
                return max(lower_bound, min(upper_bound, (origin_val + candidate_val) / 2.0))
            return rng.uniform(lower, upper)

        while(pointCnt > 0):    
            seed = rng.randint(0, len(indexes) - 1)
            points.append(indexes[seed])
            indexes.pop(seed)
            pointCnt -= 1

        for point in points:
            newVal = ind2.vector[point]
            if point < ((nodeCnt / 2) - 1): #Function node
                if (point % 2) == 0: #Function value
                    funcIdx = int(newInd.vector[point + 1])
                    originFuncIdx = int(ind2.vector[point + 1])
                    f = fset[funcIdx]
                    originVal = float(newInd.vector[point])
                    candidateVal = float(newVal)
                    if funcIdx == originFuncIdx:
                        newInd.vector[point] = sample_in_interval(originVal, candidateVal, f.minVal, f.maxVal)
                    else:
                        newInd.vector[point] = sample_in_interval(originVal, candidateVal, f.minVal, f.maxVal)

                elif (point % 2) == 1: #Function type
                    f = fset[int(newVal)]
                    newInd.vector[point] = newVal
                    fval = newInd.vector[point - 1]
                    if fval < f.minVal or fval > f.maxVal:
                        newInd.vector[point - 1] = rng.uniform(f.minVal, f.maxVal)

            else: # Terminal node
                if (point % 2) == 0: # Terminal value
                    newType = ind2.vector[point + 1]
                    originType = newInd.vector[point + 1]
                    if newType == originType:
                        if originType == 0: # Constant:
                            originVal = float(newInd.vector[point])
                            candidateVal = float(newVal)
                            newInd.vector[point] = sample_in_interval(originVal, candidateVal, minTerminalVal, maxTerminalVal)
                        else: # Variable
                            newInd.vector[point] = newVal
                    else:
                        if originType == 0: # Constant
                            newInd.vector[point] = rng.uniform(minTerminalVal, maxTerminalVal)
                        if originType == 1: # Varible
                            newInd.vector[point] = rng.randint(0, len(varlist) - 1)

                elif (point % 2) == 1: # Terminal type
                    originType = newInd.vector[point]
                    if originType != newVal:
                        newInd.vector[point] = newVal
                        if newVal == 0: #constant
                            newInd.vector[point - 1] = rng.uniform(minTerminalVal, maxTerminalVal)
                        elif newVal == 1: #variable 
                            newInd.vector[point - 1] = rng.randint(0, len(varlist) - 1)

        return newInd
    
    @staticmethod
    def onePointCrossover(ind1: PrintableIndividual, ind2: PrintableIndividual, rng: random.Random = None) -> tuple[Individual, Individual]:
        """Perform one-point subtree crossover on two printable individuals."""
        rng = rng if rng is not None else random
        if not ind1.nodes or not ind2.nodes:
            return ind1, ind2

        min_nodes = min(len(ind1.nodes), len(ind2.nodes))
        first_leaf_index = (min_nodes - 1) // 2
        if first_leaf_index <= 1:
            crossover_root = 0
        else:
            crossover_root = rng.randint(1, first_leaf_index - 1)

        def subtree_indices(nodes_length: int, root_index: int) -> list[int]:
            indices = [root_index]
            i = 0
            while i < len(indices):
                current = indices[i]
                left = 2 * current + 1
                right = 2 * current + 2
                if left < nodes_length:
                    indices.append(left)
                if right < nodes_length:
                    indices.append(right)
                i += 1
            return indices

        swap_indices = subtree_indices(min_nodes, crossover_root)

        new_ind1 = deepcopy(ind1)
        new_ind2 = deepcopy(ind2)

        # Obnovit originální variableList aby se zachovaly reference na Variable objekty
        # (deepcopy je zkopíroval, ale chceme sdílené originály)
        new_ind1.variableList = ind1.variableList
        new_ind2.variableList = ind2.variableList
        
        # Aktualizovat všechny nody tak aby ukazovaly na originální Variable objekty
        for node in new_ind1.nodes:
            if node is not None and node.variable is not None and hasattr(node.variable, 'name'):
                # Najít odpovídající Variable v originálním variableList
                for orig_var in ind1.variableList:
                    if orig_var.name == node.variable.name:
                        node.variable = orig_var
                        break
        
        for node in new_ind2.nodes:
            if node is not None and node.variable is not None and hasattr(node.variable, 'name'):
                # Najít odpovídající Variable v originálním variableList
                for orig_var in ind2.variableList:
                    if orig_var.name == node.variable.name:
                        node.variable = orig_var
                        break

        for idx in swap_indices:
            new_ind1.nodes[idx], new_ind2.nodes[idx] = new_ind2.nodes[idx], new_ind1.nodes[idx]

        return new_ind1, new_ind2
    
    @staticmethod
    def vectorOnePointCrossover(parent1: list, parent2: list, rng: random.Random = None) -> tuple[list, list]:
        """Perform a one-point crossover on two flat vector parents."""
        rng = rng if rng is not None else random
        if len(parent1) != len(parent2):
            raise ValueError("Parents must be of the same length for one-point crossover.")
        
        crossover_point = rng.randint(0, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2
    
