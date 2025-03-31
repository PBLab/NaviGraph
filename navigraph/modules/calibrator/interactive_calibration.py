import cv2
from omegaconf import DictConfig
from typing import List, Tuple, Optional
import logging


class PointCapture:
    """
    Utility class for capturing calibration points interactively from an image using OpenCV.

    This class displays an image and allows the user to click points on it.
    The clicked points are drawn on the image and stored for further processing.
    The behavior (e.g., circle radius, color, thickness) is configurable.
    """

    def __init__(self, cfg: DictConfig, window_name: str = "Capture") -> None:
        """
        Initialize PointCapture with configuration and optional window name.

        Args:
            cfg (DictConfig): Configuration object containing calibrator parameters.
            window_name (str): The name of the window used for point capture.
        """
        self._cfg: DictConfig = cfg
        self.points: List[Tuple[int, int]] = []
        self.frame: Optional = None
        self.window_name: str = window_name
        self.logger = logging.getLogger(__name__)
        # TODO: Add counter and on-screen text functionality if needed.

    @property
    def cfg(self) -> DictConfig:
        """Return the configuration."""
        return self._cfg

    def click_event(self, event: int, x: int, y: int, flags, param) -> None:
        """
        Mouse callback function for capturing points.

        Draws a circle on the image at the clicked point and saves the (x, y) coordinates.

        Args:
            event (int): The type of mouse event.
            x (int): The x-coordinate of the mouse event.
            y (int): The y-coordinate of the mouse event.
            flags: Any flags passed by OpenCV.
            param: Additional parameters (unused).
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            radius = self.cfg.calibrator_parameters.points_capture_parameters.radius
            color = eval(self.cfg.calibrator_parameters.points_capture_parameters.color)
            thickness = self.cfg.calibrator_parameters.points_capture_parameters.thickness
            cv2.circle(self.frame, (x, y), radius, color, thickness)
            self.points.append((x, y))
            self.logger.debug(f"Captured point: {(x, y)}")

    def capture_points(self, frame) -> List[Tuple[int, int]]:
        """
        Display the frame and allow the user to capture points interactively.

        The user should click on the image to select calibration points. Press 'Enter' to finish.

        Args:
            frame: The image frame from which to capture points.

        Returns:
            List[Tuple[int, int]]: A list of (x, y) coordinates for the captured points.
        """
        self.frame = frame.copy()
        self.points = []
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.click_event)

        print("Select at least 4 points on the image. Press 'Enter' to finish capturing.")

        while True:
            cv2.imshow(self.window_name, self.frame)
            key = cv2.waitKey(1)
            # 13 is the Enter key.
            if key & 0xFF == 13:
                break

        cv2.destroyAllWindows()
        return self.points
