from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd


class BaseFrameVisualizer(ABC):
    """
    Abstract base class for visualizing overlays on a video frame using data from a DataFrame row.

    Subclasses must implement the visualize() method.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the visualizer with arbitrary arguments.

        Subclasses may use these parameters as needed.
        """
        super().__init__()

    @abstractmethod
    def visualize(self, frame: np.ndarray, data_row: pd.Series) -> np.ndarray:
        """
        Draw overlays on the provided frame based on data in the DataFrame row.

        Args:
            frame (np.ndarray): The base video frame.
            data_row (pd.Series): A row from the session DataFrame containing overlay information.

        Returns:
            np.ndarray: The frame with overlays.
        """
        pass
