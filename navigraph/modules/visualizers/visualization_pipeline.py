import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from visualizer_registry import VisualizerRegistry


class VisualizationPipeline:
    """
    Pipeline that applies registered frame visualizers to a video frame using data from a DataFrame row.

    The pipeline processes the frame sequentially through visualizers specified in an ordered list.
    Each visualizer retrieves its needed data from a provided dictionary (e.g. a DataFrame row).
    """

    def __init__(self, order: List[str]):
        """
        Args:
            order (List[str]): List of keys corresponding to registered visualizers.
        """
        self.order = order

    def visualize(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        out = frame.copy()
        for key in self.order:
            vis = VisualizerRegistry.get_visualizer(key)
            if vis is not None:
                vis_data = data.get(key)
                out = vis.visualize(out, vis_data)
        return out
