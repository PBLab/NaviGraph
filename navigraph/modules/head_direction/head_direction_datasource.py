"""
HeadDirectionDatasource provides head direction (pose orientation) data from a CSV file.
It loads quaternion data from the CSV and converts them to Euler angles using general conversion utilities.
Class‐specific processing—such as adjusting sampling to match frame rates—is handled by class methods.
"""

import os
import logging
from typing import Any, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from session_datasource import SessionDataSource
from navigraph.utils.conversion_utils import quaternions_to_euler

logger = logging.getLogger(__name__)


class HeadDirectionDatasource(SessionDataSource):
    """
    Datasource for head direction data.

    This datasource loads head direction data from a CSV file and converts quaternions to Euler angles.
    The general conversion functions (such as quaternions_to_euler and wrap_angle) are located in
    general_conversion_utils.py, while class-specific adjustments (e.g. aligning sampling rate)
    are implemented as methods.

    It provides an augment_dataframe() method to add yaw, pitch, and roll for each frame,
    and a method to get head direction for a specific frame.
    """

    def __init__(self, cfg: DictConfig, head_direction_path: Optional[str] = None,
                 yaw_offset: float = -167, positive_direction: float = -1, **kwargs: Any) -> None:
        """
        Initialize the HeadDirectionDatasource.

        Args:
            cfg (DictConfig): Configuration containing head direction settings.
            head_direction_path (Optional[str]): Path to the head direction CSV file.
            yaw_offset (float): Yaw offset for conversion.
            positive_direction (float): Multiplier for yaw.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.head_direction_path: Optional[str] = head_direction_path or cfg.get("head_direction_path", None)
        self.yaw_offset: float = yaw_offset
        self.positive_direction: float = positive_direction
        self.euler_angles: Optional[np.ndarray] = None

        if self.head_direction_path is not None:
            if not os.path.isfile(self.head_direction_path):
                raise ValueError(f"Head direction file not found: {self.head_direction_path}")
            self.logger.info("Loading head direction data from %s", self.head_direction_path)
            self.head_direction_raw_data: pd.DataFrame = pd.read_csv(self.head_direction_path)
            self.euler_angles = quaternions_to_euler(self.head_direction_raw_data, yaw_offset, positive_direction)
        else:
            self.logger.warning("No head direction file provided; head direction data will be unavailable.")

    def adjust_head_direction(self, skip_index: int = 2) -> np.ndarray:
        """
        Adjust the sampling of head direction data by selecting every nth entry.
        This method is class-specific and aligns head direction data with frame indices.

        Args:
            skip_index (int): The skip multiplier (e.g., if head direction is sampled at a slower rate).

        Returns:
            np.ndarray: Adjusted Euler angles (subset of the full array).
        """
        if self.euler_angles is None:
            self.logger.warning("No Euler angles available for adjustment.")
            return np.array([])
        return self.euler_angles[::skip_index]

    def get_yaw_pitch_roll(self, frame_idx: int, skip_index: int = 2) -> Tuple[float, float, float]:
        """
        Retrieve (yaw, pitch, roll) for a given frame index.

        Uses adjust_head_direction() internally to align with frame indices.

        Args:
            frame_idx (int): Frame index.
            skip_index (int): Skip multiplier to adjust sampling.

        Returns:
            Tuple[float, float, float]: (yaw, pitch, roll) angles. Returns (np.nan, np.nan, np.nan)
              if data is not available.
        """
        adjusted = self.adjust_head_direction(skip_index)
        if adjusted.size == 0 or frame_idx >= adjusted.shape[0]:
            return (np.nan, np.nan, np.nan)
        return tuple(adjusted[frame_idx])

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the provided DataFrame with head direction data.

        This method assumes the DataFrame index corresponds to frame numbers.
        It adds three new columns: 'yaw', 'pitch', and 'roll', computed for each frame.

        Args:
            df (pd.DataFrame): The session DataFrame.

        Returns:
            pd.DataFrame: The augmented DataFrame.
        """
        self.logger.info("Augmenting DataFrame with head direction data.")
        df_aug = df.copy()
        yaw_list, pitch_list, roll_list = [], [], []
        for frame_idx in df_aug.index:
            yaw, pitch, roll = self.get_yaw_pitch_roll(frame_idx)
            yaw_list.append(yaw)
            pitch_list.append(pitch)
            roll_list.append(roll)
        df_aug["yaw"] = yaw_list
        df_aug["pitch"] = pitch_list
        df_aug["roll"] = roll_list
        return df_aug

    def load_data(self, file_path: str) -> Any:
        """
        HeadDirectionDatasource does not load data via load_data() since it loads during initialization.
        """
        raise NotImplementedError("HeadDirectionDatasource does not support load_data().")

    @classmethod
    def from_config(cls, config: dict) -> "HeadDirectionDatasource":
        """
        Instantiate a HeadDirectionDatasource from a configuration dictionary.

        Expected keys include:
          - head_direction_path: Path to the CSV file with head direction data.
          - Optionally, head_direction_yaw_offset and head_direction_positive_direction.

        Returns:
            HeadDirectionDatasource: An instance.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(config)
        head_direction_path = cfg.get("head_direction_path", None)
        yaw_offset = cfg.get("head_direction_yaw_offset", -167)
        positive_direction = cfg.get("head_direction_positive_direction", -1)
        return cls(cfg, head_direction_path=head_direction_path, yaw_offset=yaw_offset,
                   positive_direction=positive_direction)

    def initialize(self) -> None:
        """
        Default initialization: log datasource details and number of Euler entries (if available).
        """
        super().initialize()
        if self.euler_angles is not None:
            self.logger.info("HeadDirectionDatasource initialized with %d Euler angle entries.",
                             self.euler_angles.shape[0])
        else:
            self.logger.info("HeadDirectionDatasource initialized without head direction data.")
