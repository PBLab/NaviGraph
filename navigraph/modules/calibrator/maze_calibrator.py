import os
import numpy as np
import cv2
import logging
from typing import Union, Tuple
from omegaconf import DictConfig
from session_module_base import SessionModule, register_module
from modules.calibrator.utils import PointCapture  # Adjust the import based on your package structure


@register_module
class MazeCalibrator(SessionModule):
    """
    A session module responsible for calibrating the maze.

    This module captures calibration points from a video frame and a map,
    computes the transformation matrix using the configured registration method,
    and can test the calibration by overlaying transformed coordinates on the map.

    Each session can have its own calibration matrix, which may be used by other modules (e.g., the map).
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        """
        Initialize the MazeCalibrator.

        Args:
            cfg (DictConfig): Configuration settings for calibration.
            **kwargs: Additional keyword arguments passed to the base SessionModule.
        """
        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.transform_matrix: Union[np.ndarray, None] = None
        self.transform_matrix_path: Union[str, None] = None
        self.points_catcher = PointCapture(self.cfg)
        self.logger.debug("MazeCalibrator initialized with config: %s", cfg)

    def capture_points(self, path: str) -> Tuple[list, Tuple[int, int, int]]:
        """
        Capture calibration points from a video frame.

        Args:
            path (str): Path to the video file.

        Returns:
            Tuple: A tuple containing the list of captured points and the shape of the frame.
        """
        cap = cv2.VideoCapture(path)
        if not cap or not cap.isOpened():
            raise ValueError(f"Problem loading video from path: {path}")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Problem reading frame from path: {path}")

        self.logger.info("Please select at least 4 points. Press 'Enter' when done.")
        points_captured = np.round(np.array(self.points_catcher.capture_points(frame)))
        self.logger.info("Points captured: %s", points_captured)
        return points_captured, frame.shape

    def find_transform_matrix(self, path: str, map_path: str) -> np.ndarray:
        """
        Compute and return the transformation matrix using calibration points from both the video frame and the map.

        Args:
            path (str): Path to the video file for calibration.
            map_path (str): Path to the map image.

        Returns:
            np.ndarray: The computed transformation matrix.
        """
        frame_points, _ = self.capture_points(path)
        map_points, _ = self.capture_points(map_path)

        method = self.cfg.calibrator_parameters.registration_method
        if method == 'affine':
            self.transform_matrix = cv2.getPerspectiveTransform(
                frame_points.astype(np.float32), map_points.astype(np.float32))
        elif method == 'homography':
            self.transform_matrix, _ = cv2.findHomography(
                frame_points.astype(np.float32), map_points.astype(np.float32), method=0)
        elif method == 'homography&ransac':
            self.transform_matrix, _ = cv2.findHomography(
                frame_points.astype(np.float32), map_points.astype(np.float32),
                method=cv2.RANSAC, ransacReprojThreshold=30)
        else:
            raise ValueError("Registration method not supported.")

        self.logger.info("Calculated transform matrix: %s", self.transform_matrix)

        if self.cfg.calibrator_parameters.save_transform_matrix:
            calibration_path = os.path.join(
                self.cfg.calibrator_parameters.path_to_save_calibration_files, 'calibration_files')
            os.makedirs(calibration_path, exist_ok=True)
            self.transform_matrix_path = os.path.join(calibration_path, 'transform_matrix.npy')
            np.save(self.transform_matrix_path, self.transform_matrix)
            self.logger.info("Transform matrix saved at: %s", self.transform_matrix_path)
        return self.transform_matrix

    def test_calibration(self, path: str, map_path: str, matrix_path_or_array: Union[str, np.ndarray]) -> None:
        """
        Test the calibration by capturing new test points and overlaying the transformed coordinates on the map.

        Args:
            path (str): Path to the video file for testing.
            map_path (str): Path to the map image.
            matrix_path_or_array (Union[str, np.ndarray]): Either the path to a saved transform matrix or the matrix itself.
        """
        map_img = cv2.imread(map_path)
        cap = cv2.VideoCapture(path)
        if isinstance(matrix_path_or_array, np.ndarray):
            matrix = matrix_path_or_array
        else:
            matrix = np.load(matrix_path_or_array)

        if not cap or not cap.isOpened():
            raise ValueError(f"Problem loading video from path: {path}")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Problem reading frame from path: {path}")

        self.logger.info("Please select new test points. Press 'Enter' when done.")
        points = self.points_catcher.capture_points(frame)
        points_on_map = cv2.perspectiveTransform(np.array([points], dtype='float32'), matrix)[0]
        for x, y in points_on_map:
            cv2.circle(map_img, (int(x), int(y)),
                       self.cfg.calibrator_parameters.map_test_parameters.radius,
                       eval(self.cfg.calibrator_parameters.map_test_parameters.color),
                       self.cfg.calibrator_parameters.map_test_parameters.thickness)
        self.logger.info("Press Enter to close the test window.")
        while True:
            cv2.imshow('Test Calibration', map_img)
            key = cv2.waitKey(1)
            if key & 0xFF == 13:  # Enter key pressed
                cv2.destroyAllWindows()
                break

    @classmethod
    def from_config(cls, config: dict) -> "MazeCalibrator":
        """
        Factory method to instantiate a MazeCalibrator from a configuration dictionary.

        Expected configuration should include keys under 'calibrator_parameters'.

        Args:
            config (dict): Configuration parameters.

        Returns:
            MazeCalibrator: An instance of MazeCalibrator.
        """
        return cls(cfg=config)
