import os
import glob
from typing import List, Tuple

import pandas as pd
from omegaconf import DictConfig
import logging as lg

from session.session import Session
from calibrator.maze_calibrator import MazeCalibrator
from analyzer.session_analyzer import SessionAnalyzer
from plot_generator.plot_generator import PlotGenerator
from visualizer.visualizer import Visualizer

DEFAULT_RUNNING_MODE = 'analyze'
STREAM_PATH = 'stream_path'
DETECTION_PATH = 'keypoint_detection_file_path'
LOGGER_NAME = 'maze_master_log'
OUTPUT_PATH = 'experiment_output_path'

# System Modes
SYSTEM_RUNNING_MODE_KEY = 'system_running_mode'
CALIBRATE_MODE = 'calibrate'
TEST_MODE = 'test'
VISUALIZE_MODE = 'visualize'
ANALYZE_MODE = 'analyze'
SUPPORTED_VIDEO_FORMATS = ['*.mp4', '*.avi']


class Manager(object):
    # TODO: Pass an output directory to every component
    # TODO: Pass the logger to all components
    def __init__(self, cfg: DictConfig):
        output_path = cfg.get(OUTPUT_PATH, '.')
        self.logger = lg.getLogger(__name__)
        self.logger.addHandler(lg.FileHandler(os.path.join(output_path, 'maze_master.log')))

        if cfg.verbose:
            self.logger.setLevel(lg.DEBUG)

        self._cfg = cfg
        self._system_mode = self.cfg.get(SYSTEM_RUNNING_MODE_KEY, DEFAULT_RUNNING_MODE)
        self._analysis_df = None

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    @property
    def system_mode(self) -> str:
        return self._system_mode

    @property
    def analysis_df(self) -> pd.DataFrame:
        return self._analysis_df

    def process_inputs(self) -> Tuple[List, List]:

        path_to_streams_dir = self.cfg.get(STREAM_PATH, None)
        if not os.path.isdir(path_to_streams_dir):
            raise NotImplementedError('stream path must be a path to a directory containing desired .mp4 or .avi videos')
        input_streams = []
        for supported_type in SUPPORTED_VIDEO_FORMATS:
            input_streams.extend(glob.glob(os.path.join(path_to_streams_dir, supported_type), recursive=True))

        path_to_detection_files = self.cfg.get(DETECTION_PATH, None)
        if not os.path.isdir(path_to_detection_files):
            raise NotImplementedError('path to detection files must be a path to a directory containing desired .h5 files')
        detection_paths = glob.glob(os.path.join(path_to_streams_dir, "*.h5"), recursive=True)

        if len(input_streams) != len(detection_paths):
            raise AssertionError('Number of streams is not equal to number of detection files')

        # Match detection files to stream files by name - Note stream basename must by unique and
        # contained within detection file name basename
        sorted_detection_paths = []
        for input_stream in input_streams:
            for detection_path in detection_paths:
                stream_basename = os.path.basename(input_stream).split('.')[0]
                detection_basename = os.path.basename(detection_path)
                if stream_basename in detection_basename:
                    sorted_detection_paths.append(detection_path)
                    break

        return input_streams, sorted_detection_paths

    def run_new_calibrate_mode(self, input_stream_paths, detection_files_paths) -> None:
        print('Calibrating . . .')
        calibration_video_path = input_stream_paths[0]  # calibrate according to first video
        calibrator = MazeCalibrator(self.cfg)
        calibrator.find_transform_matrix(calibration_video_path, self.cfg.map_path)
        self.cfg.calibrator_parameters.pre_calculated_transform_matrix_path = calibrator.transform_matrix_path

        if TEST_MODE in self.system_mode:
            print('Testing calibration . . .')
            calibrator.test_calibration(calibration_video_path,
                                        self.cfg.map_path,
                                        calibrator.transform_matrix)
        print('Finished . . .')

    def test_saved_calibration(self, input_stream_paths, detection_files_paths) -> None:

        print('Testing calibration loaded from disk . . .')
        calibration_video_path = input_stream_paths[0]  # calibrate according to first video
        calibrator = MazeCalibrator(self.cfg)
        calibrator.test_calibration(calibration_video_path,
                                    self.cfg.map_path,
                                    self.cfg.calibrator_parameters.pre_calculated_transform_matrix_path)
        print('Finished . . .')

    def run(self):

        input_streams, sorted_detection_paths = self.process_inputs()

        print(f'Mode: {self.system_mode}')
        if CALIBRATE_MODE in self.system_mode:
            self.run_new_calibrate_mode(input_streams, sorted_detection_paths)

        elif TEST_MODE in self.system_mode:
            self.test_saved_calibration(input_streams, sorted_detection_paths)

        session_list = [Session(self.cfg,
                                stream_path=stream_path,
                                keypoint_detection_file_path=detection_file_path) for stream_path, detection_file_path
                        in zip(input_streams, sorted_detection_paths)]

        if ANALYZE_MODE in self.system_mode:
            analyzer = SessionAnalyzer(self.cfg, session_list)
            df = analyzer.run_analysis()
            self.logger.info(df)
            self._analysis_df = df
            PlotGenerator(self.cfg, data=df).generate()

        if VISUALIZE_MODE in self.system_mode:
            for input_stream, session in zip(input_streams, session_list):
                visualizer = Visualizer(cfg=self.cfg)
                visualizer.visualize_session(input_stream, session)

