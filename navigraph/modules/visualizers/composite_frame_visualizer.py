import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from base_frame_visualizer import BaseFrameVisualizer
from visualizer_registry import VisualizerRegistry


def overlay_img(background_img: np.ndarray,
                overlay_img: np.ndarray,
                resize_factor: float = 0.3,
                opacity: float = 0.5,
                frame_location: str = 'bottom_right',
                method: str = 'on_top') -> np.ndarray:
    """
    Overlays overlay_img onto background_img according to the specified method.

    Args:
        background_img (np.ndarray): The base image.
        overlay_img (np.ndarray): The image to overlay.
        resize_factor (float): Factor to resize the overlay image.
        opacity (float): Opacity of the overlay.
        frame_location (str): Location to place the overlay ("bottom_right" or "bottom_left").
        method (str): "on_top" to overlay, "side_by_side" to concatenate.

    Returns:
        np.ndarray: The resulting image.
    """
    if method == 'side_by_side':
        height_ratio = background_img.shape[0] / overlay_img.shape[0]
        height = int(background_img.shape[0])
        width = int(background_img.shape[1] * height_ratio * 2)
        resized_overlay = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)
        return np.concatenate((resized_overlay, background_img), axis=1)
    elif method == 'on_top':
        bg_copy = background_img.copy()
        ov_copy = overlay_img.copy()
        # Compute new size for overlay.
        if overlay_img.shape[0] > overlay_img.shape[1]:
            height = int(background_img.shape[0] * resize_factor)
            height_ratio = height / overlay_img.shape[0]
            width = int(overlay_img.shape[1] * height_ratio)
        else:
            width = int(background_img.shape[1] * resize_factor)
            width_ratio = width / overlay_img.shape[1]
            height = int(overlay_img.shape[0] * width_ratio)
        resized_overlay = cv2.resize(ov_copy, (height, width), interpolation=cv2.INTER_AREA)
        if frame_location == 'bottom_right':
            bg_copy[-resized_overlay.shape[0]:, -resized_overlay.shape[1]:] = resized_overlay
        elif frame_location == 'bottom_left':
            bg_copy[-resized_overlay.shape[0]:, :resized_overlay.shape[1]] = resized_overlay
        else:
            raise ValueError(f"Unsupported frame_location: {frame_location}")
        cv2.addWeighted(bg_copy, opacity, background_img, 1 - opacity, 0, background_img)
        return background_img
    else:
        raise NotImplementedError("Overlay method not supported.")


def get_default_args(func: Any) -> Dict[str, Any]:
    import inspect
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class CompositeFrameVisualizer(BaseFrameVisualizer):
    """
    Composite visualizer that applies a series of frame visualizers in sequence.

    It retrieves child visualizers from the VisualizerRegistry based on a specified order.
    Additionally, it supports a "custom" overlay mode that uses a sophisticated overlay_img function.

    Input:
        - frame: a video frame (np.ndarray)
        - data: a dict representing a row from the session DataFrame (with keys like "keypoints", "tile_bbox", "tile_id", "graph", etc.)
    """

    def __init__(self, order: List[str], mode: str = "overlay", opacity: float = 0.5,
                 custom_params: Dict[str, Any] = None) -> None:
        """
        Args:
            order (List[str]): List of keys corresponding to registered visualizers.
            mode (str): One of "overlay", "side_by_side", or "custom".
            opacity (float): Opacity for overlay modes.
            custom_params (Dict[str, Any]): Additional parameters for the custom overlay function.
        """
        assert mode in (
        "overlay", "side_by_side", "custom"), "Mode must be one of 'overlay', 'side_by_side', or 'custom'"
        self.order = order
        self.mode = mode
        self.opacity = opacity
        self.custom_params = custom_params if custom_params is not None else get_default_args(overlay_img)

    def visualize(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        images = []
        # Retrieve and apply each registered visualizer.
        for key in self.order:
            vis = VisualizerRegistry.get_visualizer(key)
            if vis is not None:
                images.append(vis.visualize(frame, data))
        if self.mode == "overlay":
            out = frame.copy()
            for img in images:
                if img.shape[:2] != out.shape[:2]:
                    img = cv2.resize(img, (out.shape[1], out.shape[0]))
                out = cv2.addWeighted(out, 1 - self.opacity, img, self.opacity, 0)
            return out
        elif self.mode == "side_by_side":
            h = frame.shape[0]
            resized = [cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)) for img in images]
            return cv2.hconcat(resized)
        elif self.mode == "custom":
            # Use our overlay_img function with provided custom_params.
            out = frame.copy()
            for img in images:
                out = overlay_img(out, img, **self.custom_params)
            return out
        else:
            return frame
