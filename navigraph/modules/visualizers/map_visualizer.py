import cv2
import numpy as np
import pandas as pd
from typing import Any, Tuple, Optional
from base_frame_visualizer import BaseFrameVisualizer


class MapVisualizer(BaseFrameVisualizer):
    """
    Visualizer for drawing map tile information on a video frame.

    Expects the DataFrame row to include:
      - "tile_bbox": a list [x, y, width, height] for the tile bounding box,
      - "tile_id": the identifier for the tile.
    """

    def __init__(self, tile_color: Tuple[int, int, int] = (255, 0, 0), **kwargs) -> None:
        super().__init__(**kwargs)
        self.tile_color = tile_color

    def visualize(self, frame: np.ndarray, data_row: pd.Series) -> np.ndarray:
        """
        Draw a rectangle for the tile and add text showing the tile id.

        Args:
            frame (np.ndarray): The video frame.
            data_row (pd.Series): Must contain "tile_bbox" and "tile_id".

        Returns:
            np.ndarray: The frame with the tile overlay.
        """
        out_frame = frame.copy()
        if "tile_bbox" in data_row and data_row["tile_bbox"] is not None:
            bbox = data_row["tile_bbox"]
            tile_id = data_row.get("tile_id", "")
            x, y, w, h = bbox
            cv2.rectangle(out_frame, (x, y), (x + w, y + h), self.tile_color, 2)
            cv2.putText(out_frame, str(tile_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.tile_color, 2)
        return out_frame
