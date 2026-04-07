"""Interactive point selection UI for camera-to-map calibration.

Side-by-side display: click a point on the video (left), then the
corresponding point on the map (right), alternating. Right-click to
undo the last point. Press Enter when done (minimum 4 pairs),
'r' to reset all, ESC to cancel.
"""

from typing import List, Tuple, Optional, NamedTuple
import cv2
import numpy as np
from loguru import logger


class Point(NamedTuple):
    """Represents a calibration point with coordinates."""
    x: float
    y: float

    def as_tuple(self) -> Tuple[int, int]:
        """Return point as integer tuple for OpenCV."""
        return (int(self.x), int(self.y))


class PointSelector:
    """Side-by-side point selection: video on left, map on right.

    Click alternates between source (video) and target (map).
    Points are stored in original image coordinates, not display coordinates.
    """

    # Colors (BGR)
    COLOR_SOURCE = (255, 100, 0)      # Blue-ish
    COLOR_TARGET = (0, 200, 0)        # Green
    COLOR_ACTIVE = (0, 200, 255)      # Yellow -- waiting for click
    COLOR_LINE = (255, 255, 0)        # Cyan
    COLOR_TEXT = (255, 255, 255)       # White
    COLOR_TEXT_BG = (0, 0, 0)         # Black

    def __init__(self):
        self.source_points: List[Point] = []
        self.target_points: List[Point] = []
        self._waiting_for = "source"  # "source" or "target"

        # Layout info (set in _build_layout)
        self._display_h = 0
        self._display_w = 0
        self._src_x0 = 0      # source image region in display
        self._src_scale = 1.0
        self._tgt_x0 = 0      # target image region in display
        self._tgt_scale = 1.0
        self._gap = 10         # pixel gap between images

    def select_corresponding_points(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        min_points: int = 4,
        window_title: str = "NaviGraph Calibration"
    ) -> Tuple[List[Point], List[Point]]:
        """Select corresponding points on side-by-side images.

        Returns:
            Tuple of (source_points, target_points) in original image coords.

        Raises:
            ValueError: If user cancels or insufficient points.
        """
        self.source_points = []
        self.target_points = []
        self._waiting_for = "source"
        self._src_orig = source_image.copy()
        self._tgt_orig = target_image.copy()

        self._build_layout(source_image, target_image)

        window = window_title
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self._mouse_callback)

        logger.info(
            f"Side-by-side calibration: click matching points alternating left/right. "
            f"Min {min_points} pairs. Enter=confirm, R=reset, ESC=cancel, Right-click=undo"
        )

        while True:
            display = self._render()
            n_pairs = min(len(self.source_points), len(self.target_points))
            self._draw_status(display, n_pairs, min_points)
            cv2.imshow(window, display)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                raise ValueError("Calibration cancelled by user")

            elif key in (13, 10):  # Enter
                if n_pairs >= min_points:
                    # Trim to equal length
                    n = min(len(self.source_points), len(self.target_points))
                    self.source_points = self.source_points[:n]
                    self.target_points = self.target_points[:n]
                    cv2.destroyAllWindows()
                    logger.info(f"Confirmed {n} point pairs")
                    return self.source_points, self.target_points

            elif key in (ord('r'), ord('R')):
                self.source_points.clear()
                self.target_points.clear()
                self._waiting_for = "source"
                logger.info("Reset all points")

        cv2.destroyAllWindows()
        return self.source_points, self.target_points

    # ---- layout ----

    def _build_layout(self, src: np.ndarray, tgt: np.ndarray) -> None:
        """Compute how to place both images side-by-side at matched height."""
        sh, sw = src.shape[:2]
        th, tw = tgt.shape[:2]

        # Scale both to the same height (use the larger)
        target_h = max(sh, th)
        self._src_scale = target_h / sh
        self._tgt_scale = target_h / th

        src_disp_w = int(sw * self._src_scale)
        tgt_disp_w = int(tw * self._tgt_scale)

        self._display_h = target_h
        self._display_w = src_disp_w + self._gap + tgt_disp_w
        self._src_x0 = 0
        self._src_disp_w = src_disp_w
        self._tgt_x0 = src_disp_w + self._gap
        self._tgt_disp_w = tgt_disp_w

    def _render(self) -> np.ndarray:
        """Build the side-by-side display with all points drawn."""
        canvas = np.zeros((self._display_h, self._display_w, 3), dtype=np.uint8)

        # Resize and place source
        src_resized = cv2.resize(self._src_orig, (self._src_disp_w, self._display_h))
        canvas[:, self._src_x0:self._src_x0 + self._src_disp_w] = src_resized

        # Resize and place target
        tgt_resized = cv2.resize(self._tgt_orig, (self._tgt_disp_w, self._display_h))
        canvas[:, self._tgt_x0:self._tgt_x0 + self._tgt_disp_w] = tgt_resized

        # Draw separator line
        sep_x = self._src_disp_w + self._gap // 2
        cv2.line(canvas, (sep_x, 0), (sep_x, self._display_h), (80, 80, 80), 2)

        # Marker size scaled to display
        r = max(4, int(8 * self._display_h / 800))
        t = max(1, int(2 * self._display_h / 800))
        fs = max(0.4, 0.7 * self._display_h / 800)
        ft = max(1, int(2 * self._display_h / 800))

        # Draw completed pairs with connecting lines
        n_pairs = min(len(self.source_points), len(self.target_points))
        for i in range(n_pairs):
            sp = self.source_points[i]
            tp = self.target_points[i]
            sd = self._src_to_display(sp)
            td = self._tgt_to_display(tp)

            cv2.circle(canvas, sd, r, self.COLOR_SOURCE, t)
            cv2.circle(canvas, td, r, self.COLOR_TARGET, t)
            cv2.line(canvas, sd, td, self.COLOR_LINE, max(1, t // 2))
            self._draw_number(canvas, sd, i + 1, fs, ft)
            self._draw_number(canvas, td, i + 1, fs, ft)

        # Draw unpaired source point (if we clicked source but not yet target)
        if len(self.source_points) > len(self.target_points):
            sp = self.source_points[-1]
            sd = self._src_to_display(sp)
            cv2.circle(canvas, sd, r, self.COLOR_ACTIVE, t + 1)
            self._draw_number(canvas, sd, len(self.source_points), fs, ft)

        # Highlight which side is active
        if self._waiting_for == "source":
            label, lx = "Click VIDEO", 10
        else:
            label, lx = "Click MAP", self._tgt_x0 + 10
        cv2.putText(canvas, label, (lx, self._display_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 1.2, self.COLOR_ACTIVE, ft + 1)

        # Labels
        cv2.putText(canvas, "VIDEO", (10, int(30 * self._display_h / 800)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, self.COLOR_TEXT, ft)
        cv2.putText(canvas, "MAP", (self._tgt_x0 + 10, int(30 * self._display_h / 800)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, self.COLOR_TEXT, ft)

        return canvas

    def _draw_number(self, img, pos, num, fs, ft):
        text = str(num)
        tsz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0]
        tx = pos[0] - tsz[0] // 2
        ty = pos[1] - int(12 * self._display_h / 800)
        cv2.rectangle(img, (tx - 2, ty - tsz[1] - 2), (tx + tsz[0] + 2, ty + 2),
                      self.COLOR_TEXT_BG, -1)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, self.COLOR_TEXT, ft)

    def _draw_status(self, img, n_pairs, min_points):
        """Draw status bar at top."""
        if n_pairs >= min_points:
            status = f"{n_pairs} pairs -- press Enter to confirm (or keep adding)"
            color = self.COLOR_TARGET
        else:
            status = f"{n_pairs}/{min_points} pairs -- need {min_points - n_pairs} more"
            color = self.COLOR_TEXT

        fs = max(0.4, 0.6 * self._display_h / 800)
        ft = max(1, int(2 * self._display_h / 800))
        tsz = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)[0]

        # Background bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], tsz[1] + 16), self.COLOR_TEXT_BG, -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.putText(img, status, (10, tsz[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, fs, color, ft)

    # ---- coordinate transforms ----

    def _src_to_display(self, pt: Point) -> Tuple[int, int]:
        return (int(pt.x * self._src_scale) + self._src_x0,
                int(pt.y * self._src_scale))

    def _tgt_to_display(self, pt: Point) -> Tuple[int, int]:
        return (int(pt.x * self._tgt_scale) + self._tgt_x0,
                int(pt.y * self._tgt_scale))

    def _display_to_src(self, dx: int, dy: int) -> Optional[Point]:
        """Convert display coords to source image coords, or None if outside."""
        if self._src_x0 <= dx < self._src_x0 + self._src_disp_w:
            return Point((dx - self._src_x0) / self._src_scale,
                         dy / self._src_scale)
        return None

    def _display_to_tgt(self, dx: int, dy: int) -> Optional[Point]:
        """Convert display coords to target image coords, or None if outside."""
        if self._tgt_x0 <= dx < self._tgt_x0 + self._tgt_disp_w:
            return Point((dx - self._tgt_x0) / self._tgt_scale,
                         dy / self._tgt_scale)
        return None

    # ---- mouse ----

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._waiting_for == "source":
                pt = self._display_to_src(x, y)
                if pt is not None:
                    self.source_points.append(pt)
                    self._waiting_for = "target"
                    logger.debug(f"Source point {len(self.source_points)}: ({pt.x:.0f}, {pt.y:.0f})")

            elif self._waiting_for == "target":
                pt = self._display_to_tgt(x, y)
                if pt is not None:
                    self.target_points.append(pt)
                    self._waiting_for = "source"
                    logger.debug(f"Target point {len(self.target_points)}: ({pt.x:.0f}, {pt.y:.0f})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Undo last point
            if self._waiting_for == "target" and len(self.source_points) > len(self.target_points):
                # We just placed a source point, undo it
                removed = self.source_points.pop()
                self._waiting_for = "source"
                logger.debug(f"Undid source point: ({removed.x:.0f}, {removed.y:.0f})")
            elif self._waiting_for == "source" and self.target_points:
                # Undo the last completed pair (remove both target and source)
                removed_t = self.target_points.pop()
                removed_s = self.source_points.pop()
                # Stay waiting for source (to redo the pair)
                logger.debug(f"Undid pair: src=({removed_s.x:.0f},{removed_s.y:.0f}), "
                             f"tgt=({removed_t.x:.0f},{removed_t.y:.0f})")

    # ---- preview (called by InteractiveCalibrator after confirmation) ----

    def show_correspondence_preview(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_points: List[Point],
        target_points: List[Point]
    ) -> bool:
        """Show preview of correspondences. Enter=confirm, r=redo, ESC=cancel."""
        if len(source_points) != len(target_points):
            return True

        self._src_orig = source_image
        self._tgt_orig = target_image
        self.source_points = list(source_points)
        self.target_points = list(target_points)
        self._build_layout(source_image, target_image)

        display = self._render()

        # Add confirmation text
        fs = max(0.4, 0.6 * self._display_h / 800)
        ft = max(1, int(2 * self._display_h / 800))
        cv2.putText(display, "Preview -- Enter to confirm, 'r' to redo, ESC to cancel",
                    (10, self._display_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, self.COLOR_TEXT, ft)

        cv2.namedWindow("Calibration Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Calibration Preview", display)

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 10):
                cv2.destroyAllWindows()
                return True
            elif key in (ord('r'), ord('R'), 27):
                cv2.destroyAllWindows()
                return False
