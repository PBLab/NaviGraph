import random
import logging
from typing import List, Tuple, Union, Callable, Optional
import numpy as np
import networkx as nx
from omegaconf import DictConfig
from pyvis.network import Network

from session_module_base import SessionModule, register_module
from modules.graph.graph_tile_dictionary import graph_dict  # mapping from tile id to node/edge


class GraphBase(SessionModule):
    """
    Base class for all graph modules in Navigraph.

    Provides common graph operations (shortest path, random walk) and visualization
    using PyVis. Subclasses must implement the build_graph() method.
    """

    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None, **kwargs) -> None:
        super().__init__(logger=logger, **kwargs)
        self.cfg: DictConfig = cfg
        self.graph: nx.Graph = nx.Graph()  # the internal NetworkX graph

    def build_graph(self) -> None:
        """
        Build the graph and store it in self.graph.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build_graph()")

    def get_tree_location(self, key: int) -> Union[Tuple, int]:
        """
        Given a tile id (key), return the corresponding node (or edge) from the graph.
        Uses a pre-defined dictionary mapping.
        """
        return graph_dict.get(key, None)

    def get_shortest_path(self, source: int, target: int, weight: Optional[str] = None, method: str = 'dijkstra') -> \
    List[int]:
        """
        Compute the shortest path between source and target nodes.
        """
        return nx.shortest_path(self.graph, source=source, target=target, weight=weight, method=method)

    def get_random_walk(self, source: int, target: int, disable_backtrack: bool = False) -> List[int]:
        """
        Compute a random walk path from source to target.
        """
        path = [source]
        current_parent = None
        while source != target:
            neighbors = list(self.graph.adj[source])
            if disable_backtrack and current_parent in neighbors and len(neighbors) > 1:
                neighbors.remove(current_parent)
            current_parent = source
            source = random.choice(neighbors)
            path.append(source)
        return path

    def visualize(self, output_file: str = "graph.html", notebook: bool = False) -> None:
        """
        Visualize the graph using PyVis. Nodes are added with their positions (if available)
        and edges with optional weight information.

        Args:
            output_file (str): File path to save the HTML visualization.
            notebook (bool): If True, show additional UI controls (e.g. for physics) in a notebook.
        """
        net = Network(height="800px", width="100%", directed=False)
        pos = nx.get_node_attributes(self.graph, 'pos')
        for node in self.graph.nodes():
            x, y = pos.get(node, (0, 0))
            net.add_node(node, label=str(node), x=x, y=y)
        for u, v, data in self.graph.edges(data=True):
            net.add_edge(u, v, value=data.get('weight', 1))
        # Set some basic options for interactivity.
        net.set_options('''
        var options = {
          "nodes": {
            "shape": "dot",
            "size": 10
          },
          "edges": {
            "smooth": false
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 1000
            }
          }
        }
        ''')
        net.show(output_file)
        if notebook:
            net.show_buttons(filter_=['physics'])

    def initialize(self) -> None:
        """
        Default initialization: log the module details.
        Subclasses may call build_graph() during initialization.
        """
        self.logger.info(f"Initializing graph module: {self.__class__.__name__}")
        # Optionally, a subclass could call self.build_graph() here.


@register_module
class BinaryTreeGraph(GraphBase):
    """
    A binary tree graph implementation.

    Builds a binary tree graph based on a given height. This implementation supports both
    balanced and unbalanced trees (by adjusting the generation logic, if needed).
    """

    def __init__(self, cfg: DictConfig, weight_func: Optional[Callable[[int, int], float]] = None, **kwargs) -> None:
        """
        Initialize BinaryTreeGraph.

        Args:
            cfg (DictConfig): Configuration with graph parameters.
            weight_func (Callable, optional): Function to determine edge weights (default constant 1).
        """
        super().__init__(cfg, **kwargs)
        self.height: int = self.cfg.graph.height
        self.weight_func: Callable[[int, int], float] = weight_func if weight_func is not None else (lambda p, c: 1)
        self.build_graph()
        self.logger.info("BinaryTreeGraph built with height %d", self.height)

    def build_graph(self) -> None:
        """
        Build a binary tree graph.

        Node naming convention: concatenation of level and node index.
        Positions are computed using a rolling average for x-coordinates.
        """
        g = nx.Graph()
        height = self.height
        # For bottom level, initial positions are sequential integers.
        x_pos = np.arange(2 ** height)
        # Function to compute rolling mean with a window of 2.
        get_x_pos = lambda x: np.convolve(x, np.ones(2), 'valid') / 2
        # Build levels from bottom (highest index) to top (level 0)
        for level in range(height)[::-1]:
            num_nodes = 2 ** level
            if level == height - 1:
                x_positions = np.arange(2 ** height)
            else:
                x_positions = get_x_pos(x_pos)[::2]
                x_pos = x_positions  # update for next level
            for i in range(num_nodes):
                node = int(f"{level}{i}")
                pos = (x_positions[i], height - level)
                g.add_node(node, pos=pos)
                if level == height - 1:
                    continue
                left_child = int(f"{level + 1}{2 * i}")
                right_child = int(f"{level + 1}{2 * i + 1}")
                g.add_edge(node, left_child, weight=self.weight_func(node, left_child))
                g.add_edge(node, right_child, weight=self.weight_func(node, right_child))
        self.graph = g

    @classmethod
    def from_config(cls, config: dict) -> "BinaryTreeGraph":
        """
        Instantiate a BinaryTreeGraph from a configuration dictionary.

        Expected configuration keys under "graph" include:
          - height: the height of the binary tree.

        Args:
            config (dict): Configuration parameters.

        Returns:
            BinaryTreeGraph: An instance of BinaryTreeGraph.
        """
        height = config.get("graph", {}).get("height", 3)
        return cls(cfg=config)


def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("GraphModule")
    logger.setLevel(logging.DEBUG)
    # Create a BinaryTreeGraph from configuration.
    graph_module = BinaryTreeGraph(cfg)

    # Example usage: compute a random walk and log it.
    random_walk = graph_module.get_random_walk(0, 637, disable_backtrack=False)
    logger.info("Random walk from 0 to 637: %s", random_walk)

    # Visualize the graph using PyVis.
    graph_module.visualize(output_file="binary_tree_graph.html", notebook=False)


if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig


    @hydra.main(config_path="../../configs", config_name="maze_master_basic")
    def hydra_main(cfg: DictConfig) -> None:
        main(cfg)


    hydra_main()
