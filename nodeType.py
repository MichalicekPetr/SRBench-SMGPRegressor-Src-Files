# Enumeration of node types used in GP trees.
# Defines function nodes, terminal nodes, and variable nodes.

from __future__ import annotations

from enum import Enum

class NodeType(Enum):
    FUNCTIONNODE = 1
    TERMINALNODE = 2