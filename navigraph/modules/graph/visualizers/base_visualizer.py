# graph_visualizer.py
from abc import ABC, abstractmethod
from typing import Any
import networkx as nx

class GraphVisualizer(ABC):
    """
    Abstract base class for graph visualizers.
    """
    @abstractmethod
    def visualize(self, graph: nx.Graph, **kwargs: Any) -> Any:
        """
        Visualize the provided graph.
        """
        pass

