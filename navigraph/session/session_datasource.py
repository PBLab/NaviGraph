# session_datasource.py

from abc import abstractmethod
from typing import List

import pandas as pd

from session_module_base import SessionModule


class SessionDataSource(SessionModule):
    """
    Abstract base class for session datasources that augment a session DataFrame.
    Inherits from SessionModule to get common behavior (e.g., logging).
    """

    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from the specified file.
        Must be implemented by subclasses.
        """
        pass

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the provided DataFrame with data from this datasource.
        Subclasses can override this method as needed.
        """
        return df

    def get_new_columns(self) -> List[str]:
        """
        Return a list of new column names that this datasource adds.
        """
        return []

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "SessionDataSource":
        """
        Instantiate a datasource from a configuration dictionary.
        Must be implemented by subclasses.
        """
        pass
