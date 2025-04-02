# pyvis_visualizer.py
from typing import Any
import networkx as nx
from pyvis.network import Network
from graph_visualizer import GraphVisualizer

class PyVisVisualizer(GraphVisualizer):
    """
    Graph visualizer using PyVis for interactive HTML-based visualization.
    """
    def visualize(self, graph: nx.Graph, output_file: str = "graph.html", **kwargs: Any) -> None:
        net = Network(height="800px", width="100%", directed=False)
        pos = nx.get_node_attributes(graph, "pos")
        for node in graph.nodes():
            node_data = graph.nodes[node].get("data")
            label = node_data.name if node_data and hasattr(node_data, "name") else str(node)
            x, y = pos.get(node, (0, 0))
            net.add_node(node, label=label, x=x, y=y)
        for u, v, data in graph.edges(data=True):
            net.add_edge(u, v, value=data.get("weight", 1), title=f"virtual: {data.get('virtual', False)}")
        net.set_options("""
        var options = {
          "nodes": {"shape": "dot", "size": 10},
          "edges": {"smooth": false},
          "physics": {"enabled": true, "stabilization": {"iterations": 1000}}
        }
        """)
        net.show(output_file)
