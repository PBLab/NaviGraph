# neural_datasource.py

import os
import logging
from typing import Any, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darr
from scipy.spatial.transform import Rotation
from omegaconf import DictConfig
from navigraph.session.session_datasource import SessionDataSource

logger = logging.getLogger(__name__)

UNIT_ID = 'unit_id'
FRAME = 'frame'


class NeuralActivityDataSource(SessionDataSource):
    """
    Datasource for neural data (e.g. df/f values for neurons) loaded from a zarr/xarray format
    similar to the Minian package.

    This datasource loads neural data from a given file/directory, optionally loads head direction data,
    computes grouping by frame and neuron unit id, and creates a DataFrame (neural_df_over_f)
    where each row corresponds to a frame and each column to a neuron’s df/f value.

    It also provides convenient query methods:
      - get_df_over_f(unit_id): Return the df/f (activity) of a given neuron over time.
      - get_spatial_footprint(unit_id): Return the spatial footprint of a neuron.
      - get_df_over_f_per_frame(frame_id): Return neural data for a specific frame.
      - get_mean_df_over_f(): Compute mean df/f per neuron over frames.

    Finally, augment_dataframe() merges the neural data with the session DataFrame (assuming the session's
    index corresponds to frame identifiers).

    This class is designed to be extended if you later wish to add further neural data properties.
    """

    def __init__(self, cfg: DictConfig, **kwargs: Any) -> None:
        """
        Initialize the NeuralDataDatasource by loading neural data from a zarr/xarray source.

        Args:
            cfg (DictConfig): Configuration including:
                - minian_path: path to the neural data (zarr/xarray format)
                - head_direction_path (optional): CSV file with head direction data.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.cfg: DictConfig = cfg
        self.logger.info("Initializing NeuralDataDatasource")

        minian_path = cfg.get("minian_path", None)
        if minian_path is None:
            raise ValueError("Configuration must provide 'minian_path' for neural data.")
        self.minian_data = self._load_minian_data(minian_path)
        self.logger.info("Neural data loaded from %s", minian_path)

        head_direction_path = cfg.get("head_direction_path", None)
        if head_direction_path is not None:
            self.head_direction_raw_data = pd.read_csv(head_direction_path)
            self.euler_angles = self.quaternions_to_euler(self.head_direction_raw_data)
        else:
            self.euler_angles = None

        # Group neural data by frame and by neuron unit id.
        self._frame_groups = self.minian_data.groupby(FRAME).groups
        self._unit_id_groups = self.minian_data.groupby(UNIT_ID).groups

        # Create a DataFrame of df/f values: rows=frames, columns=neuron unit ids.
        self.neural_df_over_f = pd.DataFrame(
            data=self.minian_data.C.values.T,
            index=list(self._frame_groups.keys()),
            columns=list(self._unit_id_groups.keys())
        )

    @staticmethod
    def quaternions_to_euler(data: pd.DataFrame, yaw_offset: float = -167,
                             positive_direction: float = -1) -> np.ndarray:
        """
        Convert quaternions to Euler angles (yaw, pitch, roll) using ZYX convention.

        Args:
            data (pd.DataFrame): DataFrame with quaternion columns: 'qw', 'qx', 'qy', 'qz'.
            yaw_offset (float): Yaw offset for calibration.
            positive_direction (float): Multiplier to correct yaw direction.

        Returns:
            np.ndarray: Array of Euler angles.
        """
        quaternions = data[['qw', 'qx', 'qy', 'qz']].dropna().values
        euler_angles = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_euler('zyx', degrees=True)
        # Wrap yaw to [-180, 180]
        euler_angles[:, 0] = (euler_angles[:, 0] - yaw_offset + 180) % 360 - 180
        euler_angles[:, 0] *= positive_direction
        return euler_angles

    @staticmethod
    def _load_minian_data(dpath: str, return_dict: bool = False) -> pd.DataFrame:
        """
        Load neural data from a file or directory in zarr/xarray format.

        Args:
            dpath (str): Path to the neural data.
            return_dict (bool): If True, return a dictionary; otherwise merge arrays.

        Returns:
            pd.DataFrame: DataFrame representation of the neural data.
        """
        if os.path.isfile(dpath):
            ds = xr.open_dataset(dpath).chunk()
        elif os.path.isdir(dpath):
            dslist = []
            for d in os.listdir(dpath):
                arr_path = os.path.join(dpath, d)
                if os.path.isdir(arr_path):
                    arr = list(xr.open_zarr(arr_path).values())[0]
                    arr.data = darr.from_zarr(os.path.join(arr_path, arr.name), inline_array=True)
                    dslist.append(arr)
            if return_dict:
                ds = {d.name: d for d in dslist}
            else:
                ds = xr.merge(dslist, compat="no_conflicts")
        else:
            raise ValueError("The provided minian_path does not exist.")
        # Convert to a DataFrame.
        return ds.to_dataframe()

    def get_df_over_f(self, unit_id: Any) -> Optional[np.ndarray]:
        """
        Get the df/f values for a given neuron (unit_id) over all frames.

        Args:
            unit_id (Any): The neuron identifier.

        Returns:
            Optional[np.ndarray]: 1D array of df/f values, or None if not found.
        """
        if unit_id not in self._unit_id_groups:
            self.logger.warning("Unit id %s not found.", unit_id)
            return None
        return self.neural_df_over_f[unit_id].values

    def get_spatial_footprint(self, unit_id: Any) -> Optional[np.ndarray]:
        """
        Get the spatial footprint for a given neuron (unit_id).

        Args:
            unit_id (Any): The neuron identifier.

        Returns:
            Optional[np.ndarray]: Array of spatial footprint data, or None on error.
        """
        try:
            return self.minian_data.groupby(UNIT_ID)[unit_id].A.values
        except Exception as e:
            self.logger.error("Error retrieving spatial footprint for unit %s: %s", unit_id, e)
            return None

    def get_df_over_f_per_frame(self, frame_id: Any) -> Optional[pd.Series]:
        """
        Get the neural data (df/f) for a specific frame.

        Args:
            frame_id (Any): The frame identifier.

        Returns:
            Optional[pd.Series]: Neural data for that frame, or None if frame_id not found.
        """
        if frame_id not in self._frame_groups:
            return None
        return self.neural_df_over_f.loc[frame_id]

    def get_mean_df_over_f(self) -> pd.Series:
        """
        Compute the mean df/f for each neuron over all frames.

        Returns:
            pd.Series: Mean df/f values indexed by neuron id.
        """
        return self.neural_df_over_f.mean(axis=0)

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the provided session DataFrame with neural data.

        Merges the neural_df_over_f DataFrame (indexed by frame) with the given DataFrame.
        Assumes that the given DataFrame's index corresponds to frame identifiers.

        Returns:
            pd.DataFrame: The augmented DataFrame.
        """
        self.logger.info("Augmenting DataFrame with neural data.")
        df_aug = df.copy()
        df_aug = df_aug.merge(self.neural_df_over_f, left_index=True, right_index=True, how="left",
                              suffixes=("", "_neural"))
        return df_aug

    def load_data(self, file_path: str) -> Any:
        """
        NeuralDataDatasource does not load data from file directly.
        Instead, it loads data from the provided minian_path during initialization.
        """
        raise NotImplementedError("NeuralDataDatasource does not support loading data from a separate file.")

    @classmethod
    def from_config(cls, config: dict) -> "NeuralDataDatasource":
        """
        Instantiate a NeuralDataDatasource from a configuration dictionary.

        Expected configuration keys include:
          - minian_path: Path to the neural data in zarr/xarray format.
          - head_direction_path (optional): Path to head direction CSV file.

        Returns:
            NeuralDataDatasource: An instance of the datasource.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(config)
        return cls(cfg)

    def initialize(self) -> None:
        """
        Default initialization: logs datasource details.
        """
        super().initialize()
        self.logger.info("NeuralDataDatasource initialized with minian data and head direction data (if provided).")
