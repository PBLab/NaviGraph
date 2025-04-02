import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple
from base_frame_visualizer import BaseFrameVisualizer


class PoseVisualizer(BaseFrameVisualizer):
    """
    Visualizer for drawing pose keypoints on a video frame.

    Expects the DataFrame row to include a key "keypoints" containing a list of (x, y) coordinates.
    """

    def __init__(self, keypoint_color: Tuple[int, int, int] = (0, 255, 0), keypoint_radius: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.keypoint_color = keypoint_color
        self.keypoint_radius = keypoint_radius

    def visualize(self, frame: np.ndarray, data_row: pd.Series) -> np.ndarray:
        """
        Draw keypoints on the frame.

        Args:
            frame (np.ndarray): The base video frame.
            data_row (pd.Series): Should contain "keypoints": List[Tuple[int,int]].

        Returns:
            np.ndarray: Frame with keypoints drawn.
        """
        out_frame = frame.copy()
        if "keypoints" in data_row and data_row["keypoints"] is not None:
            for pt in data_row["keypoints"]:
                cv2.circle(out_frame, (int(pt[0]), int(pt[1])), self.keypoint_radius, self.keypoint_color, -1)
        return out_frame
