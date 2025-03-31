import logging
from typing import Union, Dict, Optional
import pandas as pd
from omegaconf import DictConfig
from session_module_base import get_module_class, SessionModule
from navigraph.utils.logging import get_logger


class Session:
    """
    A Session represents a single experiment's analysis run.

    It holds the main DataFrame with keypoints/behavioral data and dynamically loads
    session modules (datasources) that augment the DataFrame.

    Attributes:
        cfg (Union[DictConfig, dict]): The configuration for the session.
        df (pd.DataFrame): The main DataFrame holding the session data.
        modules (Dict[str, SessionModule]): Registered modules that augment the session data.
    """

    def __init__(self,
                 cfg: Union[DictConfig, dict],
                 logger: Optional[logging.Logger]=get_logger(),
                 initial_dataframe: Optional[pd.DataFrame]=None) -> None:
        """
        Initialize the Session with a configuration and an initial DataFrame.

        Args:
            cfg (Union[DictConfig, dict]): Configuration settings for the session.
            initial_dataframe (pd.DataFrame): Initial DataFrame containing session data.
        """
        self.cfg: Union[DictConfig, dict] = cfg
        self.logger: logging.Logger = logger
        # Create a copy to avoid unintended modifications in case an initial_dataframe is provided.
        self.df: pd.DataFrame = initial_dataframe.copy() if initial_dataframe is not None else pd.DataFrame()
        self.modules: Dict[str, SessionModule] = {}
        self.load_modules()

    def load_modules(self) -> None:
        """
        Dynamically load and initialize session modules based on the configuration.

        Expects the configuration to have a 'datasources' section with keys matching
        registered module class names. Each module is instantiated using its from_config
        factory method and then initialized.

        Raises:
            ValueError: If a configured module is not found in the registry or if the
                        'datasources' section is not a dictionary.
        """
        modules_config = self.cfg.get("datasources", {})
        if not isinstance(modules_config, dict):
            raise ValueError("The 'datasources' configuration must be a dictionary.")

        for module_name, module_conf in modules_config.items():
            module_cls = get_module_class(module_name)
            if module_cls is None:
                raise ValueError(f"Module '{module_name}' not found in registry.")

            # Instantiate the module from its configuration and initialize it.
            module_instance = module_cls.from_config(module_conf)
            module_instance.initialize()
            self.modules[module_name] = module_instance

    def augment_dataframe_with_modules(self) -> pd.DataFrame:
        """
        Augment the session DataFrame by applying each registered module's
        augment_dataframe method (if available).

        Returns:
            pd.DataFrame: The augmented DataFrame with new columns or modifications added
                          by the session modules.
        """
        for module in self.modules.values():
            # Check if the module implements the augment_dataframe method.
            if hasattr(module, "augment_dataframe") and callable(module.augment_dataframe):
                self.df = module.augment_dataframe(self.df)
        return self.df

