import logging
import os
from typing import Union, Any, List, Optional

import pandas as pd

from navigraph.session.session_datasource import SessionDataSource
from navigraph.utils.logging import get_logger


class PoseDataSource(SessionDataSource):
    """
    A generic datasource for managing keypoints or behavioral location data.

    This class is responsible for loading keypoints data from a file (defaulting to HDF5),
    extracting metadata (session id, body parts, coordinate labels, etc.), and augmenting a DataFrame.

    To support a different input format, subclass and override the `load_data` method.
    """

    def __init__(self,
                 file_path: str,
                 likelihood: Union[float, None] = None,
                 logger: Optional[logging.Logger]=get_logger(),
                 **kwargs: Any) -> None:
        """
        Initialize the KeypointsDataSource by loading data from the provided file.

        Args:
            file_path (str): Path to the keypoints data file (default format is HDF5).
            likelihood (float, optional): Likelihood threshold for filtering keypoints.
            logger (logging.Logger, optional): Logger for logging messages.
            **kwargs: Additional keyword arguments (if needed).
        """
        self.file_path = file_path
        self.likelihood = likelihood
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.logger.debug("Loading keypoints data from %s", file_path)

        # Load data using the default loader (HDF5). Override load_data() for other formats.
        self.df: pd.DataFrame = self.load_data(file_path)

        # Extract common metadata from the DataFrame.
        self.session_id: Union[str, List[str]] = self._extract_session_id()
        self.session_name: str = os.path.basename(file_path).split('.')[0]
        self.bodyparts: List[str] = self._extract_bodyparts()
        self.coords: List[str] = self._extract_coords()

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load the keypoints data from the given file.

        Default implementation uses pandas.read_hdf. Override this method to support other file formats.

        Args:
            file_path (str): Path to the data file.

        Returns:
            pd.DataFrame: Loaded DataFrame containing keypoints data.
        """
        self.logger.debug("Using HDF5 loader for file: %s", file_path)
        return pd.read_hdf(file_path)

    def _extract_session_id(self) -> Union[str, List[str]]:
        """
        Extract session ID(s) from the DataFrame's multi-index (assumes level 0 holds session IDs).

        Returns:
            Union[str, List[str]]: A single session ID or list of session IDs.
        """
        session_ids = list(self.df.columns.levels[0])
        return session_ids if len(session_ids) > 1 else session_ids[0]

    def _extract_bodyparts(self) -> List[str]:
        """
        Extract body parts from the DataFrame's multi-index (assumes level 1 holds body part names).

        Returns:
            List[str]: List of body parts.
        """
        return list(self.df.columns.levels[1])

    def _extract_coords(self) -> List[str]:
        """
        Extract coordinate labels from the DataFrame's multi-index (assumes level 2 holds coordinate names).

        Returns:
            List[str]: List of coordinate labels.
        """
        return list(self.df.columns.levels[2])

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the provided DataFrame with keypoints data.

        This default implementation simply returns the loaded keypoints DataFrame.
        In practice, you might merge this data with an existing session DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to augment.

        Returns:
            pd.DataFrame: The augmented DataFrame.
        """
        self.logger.debug("Augmenting dataframe with keypoints data from session: %s", self.session_name)
        # Example: merge on an index or concatenate columns as needed.
        # Here, we simply return self.df for demonstration purposes.
        return self.df

    def get_new_columns(self) -> List[str]:
        """
        Return a list of column names that this datasource adds.

        Returns:
            List[str]: The list of new column names.
        """
        return list(self.df.columns)

    @classmethod
    def from_config(cls, config: dict) -> "KeypointsDataSource":
        """
        Factory method to instantiate a KeypointsDataSource from configuration.

        Expected configuration keys:
            - file_path: Path to the keypoints data file.
            - likelihood: (Optional) Likelihood threshold for filtering.
            - Additional parameters may be provided.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            KeypointsDataSource: An instance of KeypointsDataSource.
        """
        file_path = config.get("file_path")
        if file_path is None:
            raise ValueError("Configuration must include a 'file_path' key.")
        likelihood = config.get("likelihood")
        return cls(file_path=file_path, likelihood=likelihood)
