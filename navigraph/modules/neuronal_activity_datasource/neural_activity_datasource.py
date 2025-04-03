# neural_datasource.py

import os
import logging
from typing import Any, List, Tuple, Union, Optional
import numpy as np
import pandas as pd
import xarray as xr
import dask.array as darr
# Removed: from scipy.spatial.transform import Rotation (no longer needed directly here)
from omegaconf import DictConfig
from navigraph.session.session_datasource import SessionDataSource
# --- Import the utility function ---
from navigraph.utils.conversion_utils import quaternions_to_euler

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
                - head_direction_yaw_offset (optional): Yaw offset for quaternion conversion (default: -167).
                - head_direction_positive_direction (optional): Multiplier for yaw (default: -1).
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
        self.euler_angles: Optional[np.ndarray] = None # Initialize attribute
        if head_direction_path is not None:
            if not os.path.isfile(head_direction_path):
                 # Use logger instead of print for warnings/errors
                self.logger.warning(f"Head direction file not found: {head_direction_path}. Skipping.")
            else:
                try:
                    self.head_direction_raw_data = pd.read_csv(head_direction_path)
                    # --- Use the imported utility function ---
                    # Get optional parameters from config, using defaults from the utility if not present
                    yaw_offset = cfg.get("head_direction_yaw_offset", -167)
                    positive_direction = cfg.get("head_direction_positive_direction", -1)
                    self.euler_angles = quaternions_to_euler(
                        self.head_direction_raw_data,
                        yaw_offset=yaw_offset,
                        positive_direction=positive_direction
                    )
                except Exception as e:
                    self.logger.error(f"Error processing head direction file {head_direction_path}: {e}")
                    self.euler_angles = None # Ensure it's None on error

        # Ensure minian_data is valid before proceeding
        if self.minian_data is None or self.minian_data.empty:
             raise ValueError("Failed to load valid Minian data.")

        # Check if required columns/variables exist before accessing
        if 'frame' not in self.minian_data:
             raise ValueError("Minian data DataFrame missing 'frame' column.")
        if 'unit_id' not in self.minian_data:
             raise ValueError("Minian data DataFrame missing 'unit_id' column.")
        if 'C' not in self.minian_data:
             raise ValueError("Minian data DataFrame missing 'C' (temporal components) column.")

        # Group neural data by frame and by neuron unit id.
        # Use .index.unique() for potentially faster unique values if index is frame/unit_id
        self._frame_groups = self.minian_data.groupby(FRAME).groups
        self._unit_id_groups = self.minian_data.groupby(UNIT_ID).groups

        # Create a DataFrame of df/f values: rows=frames, columns=neuron unit ids.
        # Ensure keys exist before accessing
        frame_keys = list(self._frame_groups.keys()) if self._frame_groups else []
        unit_id_keys = list(self._unit_id_groups.keys()) if self._unit_id_groups else []

        if not frame_keys or not unit_id_keys:
             self.logger.warning("Could not group Minian data by frame or unit_id. neural_df_over_f will be empty.")
             self.neural_df_over_f = pd.DataFrame()
        else:
            try:
                 # Pivot might be more robust if data isn't perfectly aligned
                 # self.neural_df_over_f = self.minian_data.pivot(index=FRAME, columns=UNIT_ID, values='C')
                 # Or stick to original if C is already structured correctly (N_units x N_frames)
                 self.neural_df_over_f = pd.DataFrame(
                      data=self.minian_data.C.values.T, # Assuming C is (units, frames)
                      index=frame_keys, # Use the actual frame keys
                      columns=unit_id_keys # Use the actual unit_id keys
                 )
            except Exception as e:
                 self.logger.error(f"Error creating neural_df_over_f DataFrame: {e}")
                 self.neural_df_over_f = pd.DataFrame()


    # --- Removed the staticmethod quaternions_to_euler ---

    @staticmethod
    def _load_minian_data(dpath: str, return_dict: bool = False) -> Optional[pd.DataFrame]:
        """
        Load neural data from a file or directory in zarr/xarray format.

        Args:
            dpath (str): Path to the neural data.
            return_dict (bool): If True, return a dictionary; otherwise merge arrays.

        Returns:
            Optional[pd.DataFrame]: DataFrame representation of the neural data, or None on error.
        """
        ds = None # Initialize ds
        try:
            if os.path.isfile(dpath):
                ds = xr.open_dataset(dpath).chunk()
            elif os.path.isdir(dpath):
                dslist = []
                for d in os.listdir(dpath):
                    arr_path = os.path.join(dpath, d)
                    if os.path.isdir(arr_path):
                        try:
                            # Attempt to open zarr group; list(..)[0] might fail if empty/no DataArray
                            zarr_store = xr.open_zarr(arr_path)
                            data_arrays = list(zarr_store.values())
                            if not data_arrays:
                                logger.warning(f"Zarr group at {arr_path} contains no DataArrays. Skipping.")
                                continue
                            arr = data_arrays[0] # Take the first DataArray found
                            # Ensure arr.name exists before using it
                            if hasattr(arr, 'name') and arr.name:
                                arr.data = darr.from_zarr(os.path.join(arr_path, arr.name), inline_array=True)
                                dslist.append(arr)
                            else:
                                logger.warning(f"DataArray in {arr_path} has no name. Skipping.")
                        except Exception as e_zarr:
                             logger.error(f"Error processing Zarr directory {arr_path}: {e_zarr}")

                if not dslist:
                     logger.warning(f"No valid Zarr arrays found in directory {dpath}.")
                     return None

                if return_dict:
                     # Ensure items have names before creating dict
                    ds = {d.name: d for d in dslist if hasattr(d, 'name') and d.name}
                else:
                     try:
                          ds = xr.merge(dslist, compat="no_conflicts")
                     except Exception as e_merge:
                          logger.error(f"Error merging xarray DataArrays from {dpath}: {e_merge}")
                          return None # Return None if merge fails
            else:
                # Use logger
                logger.error(f"The provided minian_path '{dpath}' is not a valid file or directory.")
                return None

            # Convert to a DataFrame, handle potential errors
            if ds is not None:
                if isinstance(ds, dict): # If return_dict was True
                     # Decide how to handle dict -> DataFrame conversion
                     # Maybe combine them? For now, log warning and return None
                     logger.warning("Cannot convert dictionary of xarray datasets to single DataFrame in _load_minian_data.")
                     return None
                return ds.to_dataframe()
            else:
                 return None # ds was never assigned or merge failed

        except Exception as e_load:
            logger.error(f"Failed to load Minian data from {dpath}: {e_load}")
            return None # Return None on any loading error

    def get_df_over_f(self, unit_id: Any) -> Optional[np.ndarray]:
        """
        Get the df/f values for a given neuron (unit_id) over all frames.

        Args:
            unit_id (Any): The neuron identifier.

        Returns:
            Optional[np.ndarray]: 1D array of df/f values, or None if not found or df is empty.
        """
        if self.neural_df_over_f.empty:
             self.logger.warning("neural_df_over_f is empty. Cannot get df/f for unit %s.", unit_id)
             return None
        if unit_id not in self.neural_df_over_f.columns: # Check columns instead of _unit_id_groups
            self.logger.warning("Unit id %s not found in neural_df_over_f columns.", unit_id)
            return None
        return self.neural_df_over_f[unit_id].values

    def get_spatial_footprint(self, unit_id: Any) -> Optional[np.ndarray]:
        """
        Get the spatial footprint for a given neuron (unit_id). Requires 'A' in minian_data.

        Args:
            unit_id (Any): The neuron identifier.

        Returns:
            Optional[np.ndarray]: Array of spatial footprint data, or None on error.
        """
        if self.minian_data is None or 'A' not in self.minian_data:
             self.logger.warning("Minian data does not contain 'A' (spatial footprints).")
             return None
        try:
            # Grouping and accessing specific unit
            # Ensure unit_id exists in the 'unit_id' dimension/coord of 'A'
            if unit_id not in self.minian_data['unit_id']:
                 self.logger.warning(f"Unit id {unit_id} not found in spatial footprints ('A').")
                 return None
            # Select directly using xarray's sel or isel if unit_id is coordinate/dimension
            footprint = self.minian_data['A'].sel(unit_id=unit_id).values
            return footprint
        except Exception as e:
            self.logger.error("Error retrieving spatial footprint for unit %s: %s", unit_id, e)
            return None

    def get_df_over_f_per_frame(self, frame_id: Any) -> Optional[pd.Series]:
        """
        Get the neural data (df/f) for a specific frame.

        Args:
            frame_id (Any): The frame identifier.

        Returns:
            Optional[pd.Series]: Neural data for that frame, or None if frame_id not found or df is empty.
        """
        if self.neural_df_over_f.empty:
            self.logger.warning("neural_df_over_f is empty. Cannot get df/f for frame %s.", frame_id)
            return None
        if frame_id not in self.neural_df_over_f.index: # Check index
            self.logger.warning("Frame id %s not found in neural_df_over_f index.", frame_id)
            return None
        return self.neural_df_over_f.loc[frame_id]

    def get_mean_df_over_f(self) -> pd.Series:
        """
        Compute the mean df/f for each neuron over all frames.

        Returns:
            pd.Series: Mean df/f values indexed by neuron id. Returns empty Series if df is empty.
        """
        if self.neural_df_over_f.empty:
            self.logger.warning("neural_df_over_f is empty. Returning empty Series for mean df/f.")
            return pd.Series(dtype=float)
        return self.neural_df_over_f.mean(axis=0)

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the provided session DataFrame with neural data.

        Merges the neural_df_over_f DataFrame (indexed by frame) with the given DataFrame.
        Assumes that the given DataFrame's index corresponds to frame identifiers.

        Returns:
            pd.DataFrame: The augmented DataFrame. Returns original df if neural data is empty.
        """
        if self.neural_df_over_f.empty:
            self.logger.warning("Neural df/f data is empty. Skipping augmentation.")
            return df

        self.logger.info("Augmenting DataFrame with neural data.")
        df_aug = df.copy()
        try:
            # Ensure indices are compatible (e.g., both numeric or same dtype)
            # df_aug.index = pd.to_numeric(df_aug.index, errors='coerce')
            # self.neural_df_over_f.index = pd.to_numeric(self.neural_df_over_f.index, errors='coerce')
            # Consider checking index types before merge
            df_aug = df_aug.merge(self.neural_df_over_f, left_index=True, right_index=True, how="left",
                                suffixes=("", "_neural"))
        except Exception as e:
             self.logger.error(f"Error merging neural data into main DataFrame: {e}. Returning original DataFrame.")
             return df # Return original df on merge error
        return df_aug

    def load_data(self, file_path: str) -> Any:
        """
        NeuralDataDatasource does not load data from file directly.
        Instead, it loads data from the provided minian_path during initialization.
        """
        raise NotImplementedError("NeuralDataDatasource does not support loading data from a separate file via load_data().")

    @classmethod
    def from_config(cls, config: dict) -> "NeuralActivityDataSource":
        """
        Instantiate a NeuralActivityDataSource from a configuration dictionary.

        Expected configuration keys include:
          - minian_path: Path to the neural data in zarr/xarray format.
          - head_direction_path (optional): Path to head direction CSV file.
          - head_direction_yaw_offset (optional): Yaw offset for conversion.
          - head_direction_positive_direction (optional): Multiplier for yaw.

        Returns:
            NeuralActivityDataSource: An instance of the datasource.
        """
        from omegaconf import OmegaConf
        # Ensure config is a DictConfig for consistent access
        if not isinstance(config, DictConfig):
             cfg = OmegaConf.create(config)
        else:
             cfg = config
        # Pass the whole cfg to constructor, it will extract needed keys
        return cls(cfg=cfg)

    def initialize(self) -> None:
        """
        Default initialization: logs datasource details.
        """
        super().initialize()
        if not self.neural_df_over_f.empty:
            self.logger.info("NeuralActivityDataSource initialized with %d neurons and %d frames.",
                             self.neural_df_over_f.shape[1], self.neural_df_over_f.shape[0])
        else:
             self.logger.warning("NeuralActivityDataSource initialized, but neural_df_over_f DataFrame is empty.")
        if self.euler_angles is not None:
            self.logger.info("Head direction data loaded with %d entries.", self.euler_angles.shape[0])
        else:
            self.logger.info("Head direction data was not loaded.")