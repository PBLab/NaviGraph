# session_manager.py

import os
import glob
import logging
from typing import List, Tuple
import pandas as pd

from omegaconf import DictConfig
from session.session import Session
from navigraph.modules.calibrator import MazeCalibrator
from analysis.session_analyzer import SessionAnalyzer  # updated to your new code
from navigraph.modules.visualizers.visualization_pipeline import VisualizationPipeline

# Constants / config keys
DEFAULT_RUNNING_MODE = 'analyze'
STREAM_PATH = 'stream_path'
DETECTION_PATH = 'keypoint_detection_file_path'
OUTPUT_PATH = 'experiment_output_path'
SYSTEM_RUNNING_MODE_KEY = 'system_running_mode'

CALIBRATE_MODE = 'calibrate'
TEST_MODE = 'test'
VISUALIZE_MODE = 'visualize'
ANALYZE_MODE = 'analyze'
SUPPORTED_VIDEO_FORMATS = ['*.mp4', '*.avi']


class SessionManager:
    """
    Orchestrates the experiment pipeline:
      - Finds and matches input files.
      - Triggers calibration (if applicable).
      - Creates Session objects.
      - Runs session-level and cross-session analyses (no PlotGenerator).
      - Optionally performs visualization.
    """

    def __init__(self, cfg: DictConfig, logger: logging.Logger = None) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)

        # Set up file-based logging if needed
        output_dir = cfg.get(OUTPUT_PATH, '.')
        log_file = os.path.join(output_dir, 'session_manager.log')
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_file)
            self.logger.addHandler(file_handler)
        if cfg.verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        self._system_mode: str = cfg.get(SYSTEM_RUNNING_MODE_KEY, DEFAULT_RUNNING_MODE)
        self._analysis_df: pd.DataFrame = pd.DataFrame()  # store session-level results here

    @property
    def system_mode(self) -> str:
        return self._system_mode

    @property
    def analysis_df(self) -> pd.DataFrame:
        return self._analysis_df

    def process_inputs(self) -> Tuple[List[str], List[str]]:
        """
        Locate and match video files and detection files by shared basename.
        Returns a tuple: (video_files, detection_files).
        """
        # 1) Gather candidate videos
        video_dir = self.cfg.get(STREAM_PATH, None)
        if not os.path.isdir(video_dir):
            raise NotImplementedError("Stream path must be a directory of .mp4/.avi files.")
        video_files: List[str] = []
        for pattern in SUPPORTED_VIDEO_FORMATS:
            video_files.extend(glob.glob(os.path.join(video_dir, pattern), recursive=True))

        # 2) Gather candidate detection files
        det_dir = self.cfg.get(DETECTION_PATH, None)
        if not os.path.isdir(det_dir):
            raise NotImplementedError("Detection path must be a directory of .h5 files.")
        detection_files = glob.glob(os.path.join(det_dir, '*.h5'), recursive=True)

        # 3) Optional: Enforce same count
        if len(video_files) != len(detection_files):
            raise AssertionError("Mismatch in number of video files vs. detection files.")

        # 4) Match detection to video by shared basename
        matched_detection_files: List[str] = []
        for vid in video_files:
            vid_basename = os.path.splitext(os.path.basename(vid))[0]
            for det in detection_files:
                if vid_basename in os.path.basename(det):
                    matched_detection_files.append(det)
                    break

        self.logger.info("Found %d video files and %d matching detection files.",
                         len(video_files), len(matched_detection_files))
        return video_files, matched_detection_files

    def run_calibration(self, video_files: List[str]) -> None:
        """
        Calibrate using the first video file, then (optionally) test calibration.
        """
        self.logger.info("Running calibration using: %s", video_files[0])
        calibrator = MazeCalibrator(self.cfg)
        calibrator.find_transform_matrix(video_files[0], self.cfg.map_path)
        self.cfg.calibrator_parameters.pre_calculated_transform_matrix_path = calibrator.transform_matrix_path

        # If mode includes 'test', run a test using the same calibrator instance
        if TEST_MODE in self.system_mode:
            self.logger.info("Testing calibration after creation.")
            calibrator.test_calibration(video_files[0], self.cfg.map_path, calibrator.transform_matrix)

        self.logger.info("Calibration complete.")

    def create_sessions(self, video_files: List[str], detection_files: List[str]) -> List[Session]:
        """
        Instantiate a Session object for each (video, detection) pair.
        """
        sessions = []
        for vid, det in zip(video_files, detection_files):
            self.logger.info("Creating session for: %s", vid)
            session = Session(self.cfg, stream_path=vid, keypoint_detection_file_path=det, logger=self.logger)
            sessions.append(session)
        return sessions

    def run_analysis(self, sessions: List[Session]) -> None:
        """
        Run session-level + cross-session analysis using the new registry-based approach.
        Saves the session-level DataFrame in self._analysis_df, logs cross-session results.
        """
        self.logger.info("Running analysis on %d session(s).", len(sessions))
        analyzer = SessionAnalyzer(self.cfg, sessions)

        # 1) Session-level analysis
        session_metrics_df = analyzer.run_session_level_analysis()
        self.logger.info("Session-level metrics:\n%s", session_metrics_df)
        self._analysis_df = session_metrics_df

        # 2) Cross-session analysis
        cross_result = analyzer.run_cross_session_analysis(session_metrics_df)
        self.logger.info("Cross-session analysis result for metric '%s':\n%s",
                         cross_result.metric_name, cross_result.value)

    def run_visualization(self, video_files: List[str], sessions: List[Session]) -> None:
        """
        If needed, run your Visualizer on each session. This can remain unchanged.
        """
        self.logger.info("Running visualization on %d sessions.", len(sessions))
        for vid, sess in zip(video_files, sessions):
            self.logger.info("Visualizing session for: %s", vid)
            vis = Visualizer(cfg=self.cfg, logger=self.logger)
            vis.visualize_session(vid, sess)

    def run(self) -> None:
        """
        The main entry point. Steps:
          1) Process inputs
          2) Possibly calibrate
          3) Create sessions
          4) Run session-level + cross-session analysis
          5) Optionally visualize
        """
        self.logger.info("Starting SessionManager in mode: %s", self.system_mode)
        video_files, detection_files = self.process_inputs()

        # Possibly calibrate
        if CALIBRATE_MODE in self.system_mode:
            self.run_calibration(video_files)
        elif TEST_MODE in self.system_mode:
            self.logger.info("Testing existing calibration from disk.")
            calibrator = MazeCalibrator(self.cfg)
            calibrator.test_calibration(
                video_files[0],
                self.cfg.map_path,
                self.cfg.calibrator_parameters.pre_calculated_transform_matrix_path
            )

        # Create Session objects
        sessions = self.create_sessions(video_files, detection_files)

        # Run analysis if configured
        if ANALYZE_MODE in self.system_mode:
            self.run_analysis(sessions)

        # Run visualization if configured
        if VISUALIZE_MODE in self.system_mode:
            self.run_visualization(video_files, sessions)
