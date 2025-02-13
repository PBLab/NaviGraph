import cv2
from omegaconf import DictConfig


class PointCapture(object):

    def __init__(self, cfg: DictConfig):

        self._cfg = cfg
        self.points = []
        self.frame = []
        # TODO: add counter and put text in click_event function
        # self.count = 0

    @property
    def cfg(self):
        return self._cfg

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y),
                       self.cfg.calibrator_parameters.points_capture_parameters.radius,
                       eval(self.cfg.calibrator_parameters.points_capture_parameters.color),
                       self.cfg.calibrator_parameters.points_capture_parameters.thickness)
            self.points.append((x, y))

    def capture_points(self, frame):
        # TODO: add window name options and move destroy all windows
        self.frame = frame.copy()
        self.points = []
        cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Capture', self.click_event)
        cv2.imshow('Capture', self.frame)

        while True:
            cv2.imshow('Capture', self.frame)
            key = cv2.waitKey(1)
            if key & 0xFF == 13:  # Enter button
                cv2.destroyAllWindows()
                break

        return self.points
