"""Interactive calibrator for camera-to-map spatial transformation."""

from typing import Union, Optional, Tuple
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

from .point_selector import PointSelector, Point
from .transform_calculator import TransformCalculator, TransformMethod, CalibrationResult


class InteractiveCalibrator:
    """Orchestrates interactive camera-to-map calibration workflow."""

    # Seconds to skip at the start of video files to avoid IR calibration burn frames
    VIDEO_SKIP_SECONDS = 10

    def __init__(self, logger_instance: Optional = None):
        """Initialize calibrator.

        Args:
            logger_instance: Optional logger instance (uses global if None)
        """
        self.logger = logger_instance or logger
        self.point_selector = PointSelector()
        self.transform_calculator = TransformCalculator()

    def calibrate_camera_to_map(
        self,
        camera_source: Union[int, str, Path],
        map_image_path: Path,
        method: str = "homography_ransac",
        min_points: int = 4,
        show_preview: bool = True
    ) -> CalibrationResult:
        """Run interactive calibration workflow.

        Args:
            camera_source: Camera index (int) or video file path
            map_image_path: Path to map image
            method: Transformation method ("affine", "homography", "homography_ransac")
            min_points: Minimum number of points required
            show_preview: Whether to show correspondence preview

        Returns:
            CalibrationResult with transformation matrix and metrics

        Raises:
            ValueError: If calibration fails or user cancels
            FileNotFoundError: If files don't exist
        """
        self.logger.info("Starting interactive camera-to-map calibration")

        # Load map image
        map_image = self._load_image(map_image_path)

        # Get camera frame
        camera_frame = self._capture_camera_frame(camera_source)

        # Parse transformation method
        transform_method = self._parse_transform_method(method)

        # Interactive point selection with retry loop
        while True:
            try:
                source_points, target_points = self.point_selector.select_corresponding_points(
                    source_image=camera_frame,
                    target_image=map_image,
                    min_points=min_points,
                    window_title="NaviGraph Calibration"
                )

                if len(source_points) < min_points:
                    raise ValueError(f"Insufficient points selected: {len(source_points)} < {min_points}")

                # Show point correspondence preview if requested
                if show_preview:
                    confirmed = self.point_selector.show_correspondence_preview(
                        camera_frame, map_image, source_points, target_points
                    )
                    if not confirmed:
                        self.logger.info("User requested to redo point selection")
                        continue

                break

            except ValueError as e:
                if "cancelled" in str(e).lower():
                    self.logger.warning("Calibration cancelled by user")
                    raise
                else:
                    self.logger.error(f"Point selection failed: {e}")
                    raise

        # Convert points to numpy arrays
        src_array = np.array([[p.x, p.y] for p in source_points], dtype=np.float32)
        dst_array = np.array([[p.x, p.y] for p in target_points], dtype=np.float32)

        # Calculate transformation
        self.logger.info(f"Computing {method} transformation from {len(source_points)} point pairs")

        try:
            calibration_result = self.transform_calculator.calculate_transform(
                source_points=src_array,
                target_points=dst_array,
                method=transform_method
            )

            self.logger.info(f"Calibration computed. Reprojection error: {calibration_result.reprojection_error:.2f} pixels")

        except Exception as e:
            self.logger.error(f"Transformation calculation failed: {e}")
            raise ValueError(f"Failed to calculate transformation: {e}")

        # Show overlay preview before returning (user can redo or cancel)
        if show_preview:
            action = self._show_overlay_preview(
                camera_frame, map_image, calibration_result.transform_matrix
            )
            if action == "redo":
                self.logger.info("User requested redo from overlay preview, restarting calibration")
                return self.calibrate_camera_to_map(
                    camera_source, map_image_path, method, min_points, show_preview
                )
            elif action == "cancel":
                raise ValueError("Calibration cancelled by user at overlay preview")

        self.logger.info("Calibration accepted by user")
        return calibration_result

    def _show_overlay_preview(
        self,
        camera_frame: np.ndarray,
        map_image: np.ndarray,
        transform_matrix: np.ndarray
    ) -> str:
        """Show overlay of warped video on map for visual quality check.

        Warps the camera frame onto the map using the computed homography and
        blends them (40% map, 60% warped video) so the user can judge alignment.

        Args:
            camera_frame: The source video/camera frame
            map_image: The target map image
            transform_matrix: The computed homography matrix

        Returns:
            "accept" if user presses Enter, "redo" if R, "cancel" if ESC
        """
        h, w = map_image.shape[:2]

        # Warp the camera frame onto the map coordinate space
        warped = cv2.warpPerspective(camera_frame, transform_matrix, (w, h))

        # Blend: 40% map, 60% warped video
        blended = cv2.addWeighted(map_image, 0.4, warped, 0.6, 0)

        # Add instruction text
        fs = max(0.5, 0.8 * h / 800)
        ft = max(1, int(2 * h / 800))
        text = "Calibration preview -- Enter to accept, R to redo, ESC to cancel"
        tsz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0]

        # Semi-transparent background for text
        overlay = blended.copy()
        cv2.rectangle(overlay, (0, 0), (w, tsz[1] + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, blended, 0.3, 0, blended)
        cv2.putText(blended, text, (10, tsz[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), ft)

        cv2.namedWindow("Calibration Overlay Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Overlay Preview", blended)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):  # Enter
                cv2.destroyAllWindows()
                return "accept"
            elif key in (ord('r'), ord('R')):
                cv2.destroyAllWindows()
                return "redo"
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                return "cancel"

    def test_calibration(
        self,
        camera_source: Union[int, str, Path],
        map_image_path: Path,
        transform_matrix: np.ndarray,
        method: str = "homography_ransac"
    ) -> None:
        """Test existing calibration by selecting test points.

        Args:
            camera_source: Camera index or video file path
            map_image_path: Path to map image
            transform_matrix: Previously calculated transformation matrix
            method: Method used to calculate the matrix
        """
        self.logger.info("Starting calibration test")

        # Load images
        map_image = self._load_image(map_image_path)
        camera_frame = self._capture_camera_frame(camera_source)

        # Parse method
        transform_method = self._parse_transform_method(method)

        # Select test points on camera
        self.logger.info("Select test points on camera image")
        test_selector = PointSelector()
        test_points, _ = test_selector.select_corresponding_points(
            source_image=camera_frame,
            target_image=np.zeros_like(map_image),  # Dummy target
            min_points=1,
            window_title="Test Points Selection"
        )

        if not test_points:
            self.logger.warning("No test points selected")
            return

        # Transform test points
        test_array = np.array([[p.x, p.y] for p in test_points], dtype=np.float32)
        transformed_points = self.transform_calculator.test_transform(
            transform_matrix=transform_matrix,
            test_points=test_array,
            method=transform_method
        )

        # Show results on map
        self._show_test_results(map_image, test_points, transformed_points)

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and validate image file.

        Args:
            image_path: Path to image file

        Returns:
            Loaded image as numpy array

        Raises:
            FileNotFoundError: If image doesn't exist or can't be loaded
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")

        self.logger.debug(f"Loaded image: {image_path} ({image.shape[1]}x{image.shape[0]})")
        return image

    def _capture_camera_frame(self, source_path: Union[int, str, Path]) -> np.ndarray:
        """Load spatial image from file (image or video).

        For video files, skips ahead past the first few seconds to avoid
        IR calibration burn frames that appear at the start of recordings.

        Args:
            source_path: Path to image file or video file

        Returns:
            Spatial image as numpy array

        Raises:
            ValueError: If file can't be opened or no frame captured
            FileNotFoundError: If file doesn't exist
        """
        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Spatial source not found: {source_path}")

        self.logger.info(f"Loading spatial image from: {source_path}")

        # Check if it's an image file first
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if source_path.suffix.lower() in image_extensions:
            # Load as image
            frame = cv2.imread(str(source_path))
            if frame is None:
                raise ValueError(f"Failed to load image: {source_path}")
            self.logger.debug(f"Loaded image: {frame.shape[1]}x{frame.shape[0]}")
            return frame

        # Try as video file
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {source_path}")

        # Skip ahead to avoid burned/white IR calibration frames at the start.
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_to = min(int(fps * self.VIDEO_SKIP_SECONDS), max(total_frames - 1, 0))
        if skip_to > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_to)
            self.logger.debug(
                f"Skipped to frame {skip_to} ({skip_to / fps:.1f}s) to avoid IR calibration frames"
            )

        ret, frame = cap.read()
        if not ret:
            # Fall back to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Failed to read frame from video: {source_path}")

        cap.release()

        if frame is None:
            raise ValueError("No frame captured from video")

        self.logger.debug(f"Captured frame from video: {frame.shape[1]}x{frame.shape[0]}")
        return frame

    def _parse_transform_method(self, method_str: str) -> TransformMethod:
        """Parse string method to TransformMethod enum."""
        method_map = {
            "affine": TransformMethod.AFFINE,
            "homography": TransformMethod.HOMOGRAPHY,
            "homography_ransac": TransformMethod.HOMOGRAPHY_RANSAC,
        }

        if method_str not in method_map:
            available = ", ".join(method_map.keys())
            raise ValueError(f"Unknown method '{method_str}'. Available: {available}")

        return method_map[method_str]

    def _show_test_results(
        self,
        map_image: np.ndarray,
        original_points: list,
        transformed_points: np.ndarray
    ) -> None:
        """Show test results on map image.

        Args:
            map_image: Map image to draw on
            original_points: Original test points from camera
            transformed_points: Points transformed to map coordinates
        """
        display = map_image.copy()

        # Draw transformed points on map
        for i, (orig_pt, trans_pt) in enumerate(zip(original_points, transformed_points)):
            pos = (int(trans_pt[0]), int(trans_pt[1]))
            cv2.circle(display, pos, 12, (0, 255, 255), 3)  # Yellow circle
            cv2.putText(display, str(i + 1), (pos[0] + 15, pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add instructions
        cv2.putText(display, f"Test Results: {len(original_points)} points transformed",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "Press any key to close", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.namedWindow("Calibration Test Results", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Test Results", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.logger.info("Test completed")
