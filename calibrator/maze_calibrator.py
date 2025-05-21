import os
import numpy as np
import logging as lg
import hydra
import cv2
from omegaconf import DictConfig
from calibrator.utils import PointCapture

log = lg.getLogger(__name__)


# ToDo : Change prints to log DEBUG messages...
class MazeCalibrator(object):

    def __init__(self, cfg: DictConfig):
        if cfg.verbose:
            log.setLevel(lg.DEBUG)

        self._cfg = cfg
        self.transform_matrix = None
        self.transform_matrix_path = None
        self.points_catcher = PointCapture(self.cfg)

    @property
    def cfg(self):
        return self._cfg

    def capture_points(self, path) -> (list, tuple):
        cap = cv2.VideoCapture(path)
        if not cap:
            raise ValueError(f"Problem loading path: {path}")

        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Problem loading path: {path}")

        print("Hello. Please choose a minimum of 4 points. Press 'Enter' when done:")
        points_captured = np.round(np.array(self.points_catcher.capture_points(frame)))
        print(f"Points captured: {points_captured}")

        return points_captured, frame.shape

# ToDO: make possible to move/delete points, number points, show both windows at the same time

    def find_transform_matrix(self, path, map_path) -> np.ndarray:

        frame_points, frame_shape = self.capture_points(path)
        map_points, map_shape = self.capture_points(map_path)

        if self.cfg.calibrator_parameters.registration_method == 'affine':
            self.transform_matrix = cv2.getPerspectiveTransform(frame_points.astype(np.float32),
                                                                map_points.astype(np.float32))

        elif self.cfg.calibrator_parameters.registration_method == 'homography':
            self.transform_matrix, _ = cv2.findHomography(frame_points.astype(np.float32),
                                                          map_points.astype(np.float32),
                                                          method=0)

        elif self.cfg.calibrator_parameters.registration_method == 'homography&ransac':
            self.transform_matrix, _ = cv2.findHomography(frame_points.astype(np.float32),
                                                          map_points.astype(np.float32),
                                                          method=cv2.RANSAC,
                                                          ransacReprojThreshold=30)
        else:
            raise ValueError('Registration method not supported . . .')

        print(f'Calculated transform matrix: {self.transform_matrix}')

        if self.cfg.calibrator_parameters.save_transform_matrix:
            calibration_path = os.path.join(self.cfg.calibrator_parameters.path_to_save_calibration_files,
                                            'calibration_files')
            os.makedirs(calibration_path, exist_ok=True)
            self.transform_matrix_path = os.path.join(calibration_path, 'transform_matrix.npy')
            np.save(self.transform_matrix_path, self.transform_matrix)
            print(f'Transform matrix saved at: {self.transform_matrix_path}')

        return self.transform_matrix

    def test_calibration(self, path, map_path, matrix_path_or_array):

        map_img = cv2.imread(map_path)
        cap = cv2.VideoCapture(path)
        if isinstance(matrix_path_or_array, np.ndarray):
            matrix = matrix_path_or_array
        else:
            matrix = np.load(matrix_path_or_array)

        if not cap:
            raise ValueError(f"Problem loading path: {path}")

        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Problem loading path: {path}")

        print("Hello, Maze-Master. Please choose new points to test. Press 'Enter' when done:")

        points = self.points_catcher.capture_points(frame)
        points_on_map = cv2.perspectiveTransform(np.array([points], dtype='float32'), matrix)[0]

        for x, y in points_on_map:
            cv2.circle(map_img, (x.astype(int), y.astype(int)),
                       self.cfg.calibrator_parameters.map_test_parameters.radius,
                       eval(self.cfg.calibrator_parameters.map_test_parameters.color),
                       self.cfg.calibrator_parameters.map_test_parameters.thickness)

        print('Press Enter to close window . . . ')
        while True:
            cv2.imshow('test', map_img)
            key = cv2.waitKey(1)
            if key & 0xFF == 13:  # Enter
                cv2.destroyAllWindows()
                break


@hydra.main(config_path="../configs", config_name="maze_master_basic")
def main(cfg: DictConfig):

    maze_calibrator = MazeCalibrator(cfg)
    if 'calibrate' in cfg.system_running_mode:
        maze_calibrator.find_transform_matrix(cfg.stream_path, cfg.map_path)
        if 'test' in cfg.system_running_mode:
            maze_calibrator.test_calibration(cfg.stream_path,
                                             cfg.map_path,
                                             maze_calibrator.transform_matrix)
            print('Finished calibrating and testing . . .')
        else:
            print('Finished calibrating . . .')

    elif 'test' in cfg.system_running_mode:
        maze_calibrator.test_calibration(cfg.stream_path,
                                         cfg.map_path,
                                         cfg.calibrator_parameters.pre_calculated_transform_matrix_path)
        print('Finished testing . . .')

    else:
        print('Skipping calibration . . .')


if __name__ == '__main__':
    main()
