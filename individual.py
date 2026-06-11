# Individual representation and evaluation logic for GP individuals.
# Supports both vector and tree-based individuals with function and terminal nodes.

from __future__ import annotations

from smoothMultifunctionSet import SmoothMultifunctionSet
from node import Node
from nodeType import NodeType
import numpy as np
import random

class Individual:
    """A flat vector GP individual for evaluation and genetic operators."""

    def __init__(self, depth: int = 0, variableList: list = None, funcList: list = None):
        """Create a vector-based GP individual with a given depth."""
        self.depth = depth
        self.vector = np.array([], dtype=float)
        self.variableList = variableList
        self.funcList = funcList

    def transformToPrintableIndividual(self) -> PrintableIndividual:
        """Convert the vector individual into a tree-like PrintableIndividual."""
        pInd = PrintableIndividual(depth=self.depth, randomInit=False, variableList=self.variableList)
        pInd.nodes = []
        for i in range(pow(2, self.depth - 1) - 1):
            fname = "F" + str(int(self.vector[(2*i)+1]) + 1)
            pInd.nodes.append(Node(type=NodeType.FUNCTIONNODE, value=self.vector[2*i], funcName=fname))
        for i in range(pow(2, self.depth - 1) - 1, pow(2, self.depth) - 1):
            if int(self.vector[(2 * i) + 1]) == 1: # Variable
                var = self.variableList[int(self.vector[(2*i)])]
                pInd.nodes.append(Node(type=NodeType.TERMINALNODE, variable=var))
            elif int(self.vector[(2 * i) + 1]) == 0: # Constant
                val = self.vector[2*i]
                pInd.nodes.append(Node(type=NodeType.TERMINALNODE, value=val))
            else:
                print(self.vector.at((2 * i) + 1))
                raise ValueError()
        return pInd
            
    def __str__(self):
        """Return a human-readable tree representation of the individual."""
        pInd = self.transformToPrintableIndividual()
        return pInd.getVerticalTreeString()
    
    def printTree(self):
        """Print the tree form of the individual to standard output."""
        pInd = self.transformToPrintableIndividual()
        pInd.printVerticalTree()



    @staticmethod
    def createRandomIndividual(depth: int = 0, variableList: list = None, variableProbability: float = 0.5, functionList: list = None,
                            minTerminalNodeVal: float = 0, maxTerminalNodeVal: float = 10, rng: random.Random = None) -> Individual:
        """Create a new random flat-vector individual for the given depth and variables."""
        rng = rng if rng is not None else random
        newIndividual = Individual(depth=depth, variableList=variableList, funcList=functionList) 
        
        # Dočasný plochý seznam
        temp_list = []
        
        for _ in range(pow(2, depth - 1) - 1):
            fidx = rng.randint(0, len(functionList) - 1)
            fval = rng.uniform(functionList[fidx].minVal, functionList[fidx].maxVal)
            # .extend() přidá prvky z nuly/seznamu jednotlivě za sebou
            temp_list.extend([fval, fidx])  
            
        for _ in range(pow(2, depth - 1)):
            seed = rng.random()
            if seed < variableProbability:
                varIdx = rng.randint(0, len(variableList) - 1)
                temp_list.extend([varIdx, 1.0]) # 1.0 aby seděl datový typ float
            else:
                tval = rng.uniform(minTerminalNodeVal, maxTerminalNodeVal)
                temp_list.extend([tval, 0.0])

        # Výsledkem bude jedno dlouhé ploché 1D NumPy pole
        
        newIndividual.vector = np.array(temp_list, dtype=float)
        return newIndividual
    
    def evaluate(self, fList: list, varList: list) -> float:
        """Evaluate the individual recursively using the given function and variable lists."""
        return self.evaluateRec(fList, varList, 0, 1)
    
    def evaluateRec(self, fList: list, varList: list, idx: int, depth: int) -> float:
        """Recursive helper for evaluate, traversing the flat vector representation."""
        if depth >= self.depth: # Terminal node
            val = self.vector[2*idx]
            type = self.vector[(2*idx)+1]
            if type == 0: # Constant
                return val
            else: # Variable
                return varList[int(val)].value

        else:
            left = self.evaluateRec(fList, varList, 2 * idx + 1, depth + 1)
            right = self.evaluateRec(fList, varList, 2 * idx + 2, depth + 1)

            fIdx = int(self.vector[(2 * idx) + 1])
            fVal = self.vector[2 * idx]
            f = fList[fIdx]

            return f.calculateResult(left, right, fVal)



class PrintableIndividual:
    """A tree-like individual representation used for printing and tree-based mutation."""

    def __init__(self, depth: int = 0, randomInit: bool = False, 
                 variableList: list = None, variableProbability: float = 0.5,  
                 minTerminalNodeVal: int = 0, maxTerminalNodeVal: int = 10,
                 minFunctionNodeVal: int = 0, maxFunctionNodeVal: int = 2, rng: random.Random = None):
        """Initialize a printable individual, optionally randomly filled."""
        self.depth = depth
        self.nodes = []
        self.variableList = variableList  # Uchováme si odkaz na variableList
        rng = rng if rng is not None else random
        
        if randomInit:
            for i in range(pow(2, depth - 1) - 1):
                self.nodes.append(Node.createRandomFunctionNode(minFunctionNodeVal, maxFunctionNodeVal, rng=rng))
            for i in range(pow(2, depth - 1)):
                self.nodes.append(Node.createRandomTerminalNode(minTerminalNodeVal, maxTerminalNodeVal, variableList, variableProbability, rng=rng))
        else:
            for i in range(pow(2, depth) - 1):
                self.nodes.append(None)

    @staticmethod
    def createRandomIndividual(smoothMultifunctionSet: SmoothMultifunctionSet, depth: int = 0, 
                               variableList: list = None, variableProbability: float = 0.5,
                               minTerminalNodeVal = 0, maxTerminalNodeVal = 10) -> Individual:
        """Create a random tree-based individual using the provided function set."""
        newIndividual = Individual(depth, randomInit = True, 
                                   variableList = variableList, variableProbability = variableProbability, 
                                   minTerminalNodeVal = minTerminalNodeVal, maxTerminalNodeVal = maxTerminalNodeVal,
                                   minFunctionNodeVal = smoothMultifunctionSet.minVal, maxFunctionNodeVal = smoothMultifunctionSet.maxVal)
        return newIndividual

    def getTreeString(self, index: int = 0, prefix: str = "", isLast: bool = True) -> str:
        """Return an ASCII tree representation of the individual."""
        if index >= len(self.nodes):
            return ""

        node = self.nodes[index]
        if node is None:
            node_label = "<empty>"
        else:
            node_label = str(node)

        branch = "└── " if isLast else "├── "
        result = prefix + branch + node_label + "\n"

        left_index = 2 * index + 1
        right_index = 2 * index + 2
        if left_index < len(self.nodes) or right_index < len(self.nodes):
            next_prefix = prefix + ("    " if isLast else "│   ")
            if left_index < len(self.nodes):
                result += self.getTreeString(left_index, next_prefix, right_index >= len(self.nodes))
            if right_index < len(self.nodes):
                result += self.getTreeString(right_index, next_prefix, True)

        return result

    def printTree(self) -> None:
        """Print the current tree structure to stdout."""
        if not self.nodes:
            print("<empty individual>")
            return
        if self.nodes[0] is None:
            print("<empty root>")
            return
        print(self.getTreeString())

    def getVerticalTreeString(self) -> str:
        """Construct a vertical ASCII rendering of the tree."""
        if not self.nodes:
            return "<empty individual>\n"

        node_texts = [str(node) if node is not None else "" for node in self.nodes]
        node_widths = [max(3, len(text)) for text in node_texts]
        
        leaf_count = 2 ** max(0, self.depth - 1)
        leaf_start = 2 ** (self.depth - 1) - 1

        # Dynamický výpočet minimální mezery (gap) na základě hloubky stromu.
        # Čím je strom hlubší, tím větší mezery mezi listy dole potřebujeme,
        # aby se nad ně bezpečně vešly texty větších funkčních uzlů.
        base_gap = 1
        if self.depth > 2:
            # Zjistíme maximální šířku jakéhokoliv funkčního uzlu nad listy
            max_fnode_width = max(node_widths[:leaf_start]) if leaf_start > 0 else 3
            # Mezera se odvíjí od velikosti funkčních uzlů a hloubky
            base_gap = max(1, (max_fnode_width // 2) + 1)

        leaf_centers = []
        current_x = 0
        for i in range(leaf_count):
            idx = leaf_start + i
            width = node_widths[idx] if idx < len(node_widths) else 3
            leaf_centers.append(current_x + width // 2)
            
            if i < leaf_count - 1:
                next_idx = leaf_start + i + 1
                next_width = node_widths[next_idx] if next_idx < len(node_widths) else 3
                
                # Mezera se dynamicky přizpůsobuje šířce sousedních listů a vypočtené base_gap
                gap = base_gap + max(0, (width + next_width) // 4)
                current_x += width + gap
            else:
                current_x += width

        total_width = current_x
        centers_by_level = [leaf_centers]

        while len(centers_by_level[0]) > 1:
            current = centers_by_level[0]
            parents = []
            for i in range(0, len(current), 2):
                parents.append((current[i] + current[i + 1]) // 2)
            centers_by_level.insert(0, parents)

        lines = []
        for level in range(self.depth):
            centers = centers_by_level[level]
            node_line = [" "] * total_width

            for j, center in enumerate(centers):
                index = 2 ** level - 1 + j
                text = node_texts[index] if index < len(node_texts) else ""
                width = node_widths[index] if index < len(node_widths) else 3
                text = text.center(width)
                
                # BEZPEČNOSTNÍ POJISTKA: Výpočet levého okraje nesmí klesnout pod nulu
                left = max(0, int(center - width // 2))
                
                for k, ch in enumerate(text):
                    # Zabrání zápisu mimo vyhrazenou šířku řádku
                    if left + k < total_width:
                        node_line[left + k] = ch

            if level > 0:
                indicator_line = [" "] * total_width
                for center in centers:
                    if center < total_width:
                        indicator_line[center] = "|"
                lines.append("".join(indicator_line))

            lines.append("".join(node_line))

            if level < self.depth - 1:
                branch_line = [" "] * total_width
                connector_line = [" "] * total_width
                next_centers = centers_by_level[level + 1]

                for j, center in enumerate(centers):
                    if center < total_width:
                        branch_line[center] = "|"
                    
                    if 2 * j + 1 < len(next_centers):
                        left_child = next_centers[2 * j]
                        right_child = next_centers[2 * j + 1]
                        # Zafixování rozsahu, aby spojovníky nepřetekly šířku plátna
                        start_x = min(total_width - 1, left_child)
                        end_x = min(total_width - 1, right_child)
                        for x in range(start_x, end_x + 1):
                            connector_line[x] = "-"

                lines.append("".join(branch_line))
                lines.append("".join(connector_line))

        return "\n".join(lines) + "\n"

    def printVerticalTree(self) -> None:
        """Print the vertical tree rendering without adding an extra newline."""
        print(self.getVerticalTreeString(), end="")

    def __str__(self) -> str:
        """Return a compact tree-string representation."""
        return self.getTreeString()
    
    def evaluate(self, smoothMultifunctionSet: SmoothMultifunctionSet) -> float:
        """Evaluate the printable individual using the provided smooth multifunction set."""
        if not self.nodes or self.nodes[0] is None:
            raise ValueError("Cannot evaluate an empty individual")

        return self.evaluateRec(smoothMultifunctionSet, 0, 1)
    
    def evaluateRec(self, smoothMultifunctionSet: SmoothMultifunctionSet, idx: int, depth: int) -> float:
        """Recursive evaluation helper for printable individuals."""
        if depth > self.depth - 1:
            node = self.nodes[idx]
            if node is None:
                raise ValueError("Terminal node is None")
            if node.variable:
                if not node.variable.initialized:
                    raise ValueError(f"Variable '{node.variable}' not initialized with a value")
                return node.variable.value
            else:
                return node.value
        else:
            left = self.evaluateRec(smoothMultifunctionSet, 2 * idx + 1, depth + 1)
            right = self.evaluateRec(smoothMultifunctionSet, 2 * idx + 2, depth + 1)
            return smoothMultifunctionSet.calculateResult(left, right, self.nodes[idx].value)
        
    def getConstantNodesVector(self) -> list:
        """Return a list of constant terminal nodes in the individual."""
        constantNodesVector = []
        for node in self.nodes[2 ** (self.depth - 1) - 1:]:
            if node is not None and not node.variable:
                constantNodesVector.append(node)
        return constantNodesVector