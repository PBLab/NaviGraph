# graph_datasource.py
import random
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import networkx as nx
from omegaconf import DictConfig
from session_datasource import SessionDatasource  # our abstract datasource base class
from graph_node import Node, TileNode


class GraphDatasource(SessionDatasource, ABC):
    """
    Abstract base class for graph datasources.

    This class builds a NetworkX graph representing the maze topology and
    provides common operations such as shortest path, random walk, and augmentation of the session DataFrame.
    """

    def __init__(self, cfg: DictConfig, logger: Optional[logging.Logger] = None, **kwargs: Any) -> None:
        super().__init__(logger=logger, **kwargs)
        self.cfg: DictConfig = cfg
        self.graph: nx.Graph = nx.Graph()  # Underlying graph
        self.logger.info(f"Initializing graph datasource: {self.__class__.__name__}")

    @abstractmethod
    def build_graph(self) -> None:
        """
        Build the graph (nodes and edges) and store it in self.graph.
        """
        pass

    def load_graph_from_mapping(self, mapping: Dict[Any, Any]) -> None:
        """
        Load graph nodes and edges from a mapping dictionary.

        The mapping keys are tile ids and values can be:
          - int: mapping to a physical node id.
          - tuple: mapping to an edge (node1, node2).
          - frozenset: containing both an int and a tuple, meaning the tile maps to both a node and an edge.

        Nodes are created as TileNodes (tile_id is required) and edges are added with attributes:
          'tile_id': the mapping tile id, and 'virtual': True if the edge is virtual.
        """
        nodes: Dict[int, TileNode] = {}
        edges: List[Tuple[int, int, Dict[str, Any]]] = []

        for tile_id, value in mapping.items():
            if isinstance(value, int):
                # Physical node
                node = TileNode(node_id=value, tile_id=tile_id)
                nodes[value] = node
            elif isinstance(value, tuple):
                # Edge between two nodes
                n1, n2 = value
                edges.append((n1, n2, {"tile_id": tile_id, "virtual": False}))
            elif isinstance(value, frozenset):
                node_id = None
                edge_tuple = None
                for item in value:
                    if isinstance(item, int):
                        node_id = item
                    elif isinstance(item, tuple):
                        edge_tuple = item
                if node_id is not None:
                    # For nodes that also map to an edge, mark the edge attribute as virtual
                    node = TileNode(node_id=node_id, tile_id=tile_id)
                    nodes[node_id] = node
                if edge_tuple is not None:
                    n1, n2 = edge_tuple
                    edges.append((n1, n2, {"tile_id": tile_id, "virtual": True}))
            else:
                self.logger.warning("Unrecognized mapping for tile %s: %s", tile_id, value)

        for node in nodes.values():
            self.graph.add_node(node.node_id, data=node)
        for u, v, attr in edges:
            self.graph.add_edge(u, v, **attr)
        self.logger.info("Graph loaded from mapping with %d nodes and %d edges", self.graph.number_of_nodes(),
                         self.graph.number_of_edges())

    def get_shortest_path(self, source: int, target: int, weight: Optional[str] = None, method: str = "dijkstra") -> \
    List[int]:
        return nx.shortest_path(self.graph, source=source, target=target, weight=weight, method=method)

    def get_random_walk(self, source: int, target: int, disable_backtrack: bool = False) -> List[int]:
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

    def get_graph_location(self, tile_id: int) -> Optional[int]:
        """
        Given a tile id, return the node id from the graph whose associated TileNode has that tile_id.
        If multiple nodes match, return the first one found.
        """
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id].get("data")
            if node_data and hasattr(node_data, "tile_id") and node_data.tile_id == tile_id:
                return node_id
        return None

    def augment_dataframe(self, df: Any) -> Any:
        """
        Augment the provided DataFrame with graph-related information.

        For example, if df has a column 'tile_id', add a new column 'graph_node' that maps each tile_id to a node id.
        """
        if "tile_id" in df.columns:
            df["graph_node"] = df["tile_id"].apply(lambda tid: self.get_graph_location(tid))
            self.logger.info("Augmented dataframe with 'graph_node' column.")
        else:
            self.logger.warning("DataFrame does not contain 'tile_id' column; skipping augmentation.")
        return df

    @abstractmethod
    def get_visualizer(self) -> "GraphVisualizer":
        """
        Return an instance of a GraphVisualizer to visualize this graph.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "GraphDatasource":
        """
        Create an instance from a configuration dictionary.
        """
        pass

    def initialize(self) -> None:
        """
        Default initialization: log the datasource details.
        """
        super().initialize()  # uses the default logging from SessionDatasource
