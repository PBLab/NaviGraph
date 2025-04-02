import io
import cv2
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from typing import Any
from base_frame_visualizer import BaseFrameVisualizer


class GraphVisualizer(BaseFrameVisualizer):
    """
    Visualizer for graph data. This visualizer creates a static image from a NetworkX graph.

    Expects the DataFrame row to include a key "graph" containing a NetworkX graph.
    This visualizer ignores the input frame and returns an image generated from the graph.
    """

    def __init__(self, node_color: str = "blue", edge_color: str = "gray", node_size: int = 300, **kwargs) -> None:
        super().__init__(**kwargs)
        self.node_color = node_color
        self.edge_color = edge_color
        self.node_size = node_size

    def visualize(self, frame: np.ndarray, data_row: pd.Series) -> np.ndarray:
        """
        Generate a graph visualization and return it as an image.

        Args:
            frame (np.ndarray): The base frame (ignored).
            data_row (pd.Series): Must contain "graph" as a NetworkX graph.

        Returns:
            np.ndarray: An image (numpy array) of the graph.
        """
        graph = data_row.get("graph", None)
        if graph is None or not isinstance(graph, nx.Graph):
            return frame  # Fallback: return original frame.

        plt.figure(figsize=(6, 4))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_color=self.node_color, edge_color=self.edge_color,
                node_size=self.node_size)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img = np.array(Image.open(buf))
        buf.close()
        return img
