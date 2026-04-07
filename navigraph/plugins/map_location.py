"""Map location plugin for NaviGraph unified architecture.

Transforms pose coordinates to map coordinates using calibration matrix.
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from pathlib import Path

from ..core.navigraph_plugin import NaviGraphPlugin
from ..core.exceptions import NavigraphError
from ..core.coordinate_transform import apply_coordinate_transform_to_bodyparts
from ..core.registry import register_data_source_plugin


@register_data_source_plugin("map_location")
class MapLocationPlugin(NaviGraphPlugin):
    """Provides map image and adds map coordinate columns for tracked bodyparts."""
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'map_location'),
            'type': 'map_location',
            'description': 'Transforms pose coordinates to map coordinates using calibration',
            'provides': ['map_image', 'map_metadata'],
            'augments': 'bodypart_map_x, bodypart_map_y columns'
        }
    
    def provide(self, shared_resources: Dict[str, Any]) -> None:
        """Load map image and add to shared resources.
        
        Args:
            shared_resources: Dictionary to store map resources
        """
        # Validate we have discovered files
        if not self.discovered_files:
            raise NavigraphError(
                f"MapLocationPlugin requires map image file but none found. "
                f"Check file_pattern in config: {self.config.get('file_pattern')}"
            )
        
        map_file = self.discovered_files[0]  # Use first discovered map
        self.logger.info(f"Loading map image from: {map_file.name}")
        
        try:
            # Load map image
            map_image = cv2.imread(str(map_file))
            
            if map_image is None:
                raise NavigraphError(f"Failed to load map image: {map_file}")
            
            # Create map metadata
            map_metadata = {
                'width': map_image.shape[1],
                'height': map_image.shape[0],
                'channels': map_image.shape[2] if len(map_image.shape) > 2 else 1,
                'map_path': str(map_file)
            }
            
            # Add to shared resources
            shared_resources['map_image'] = map_image
            shared_resources['map_metadata'] = map_metadata
            
            self.logger.info(
                f"✓ Map loaded: {map_metadata['width']}x{map_metadata['height']} pixels"
            )
            
        except Exception as e:
            raise NavigraphError(
                f"Failed to load map image from {map_file}: {str(e)}"
            ) from e
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Add map coordinate columns for tracked bodyparts.
        
        Args:
            dataframe: DataFrame with pose tracking data
            shared_resources: Available shared resources
            
        Returns:
            DataFrame with map coordinate columns added
        """
        # Validate calibration matrix is available
        if 'calibration_matrix' not in shared_resources:
            raise NavigraphError(
                "MapLocationPlugin requires calibration matrix. "
                f"Available resources: {list(shared_resources.keys())}"
            )
        
        calibration_matrix = shared_resources['calibration_matrix']
        
        # Detect available bodyparts from pose data
        available_bodyparts = self._detect_pose_bodyparts(dataframe)
        
        if not available_bodyparts:
            raise NavigraphError(
                "MapLocationPlugin requires pose tracking data. "
                f"No bodypart columns found. Available columns: {list(dataframe.columns)}"
            )
        
        # Get bodyparts configuration
        config_bodyparts = self.config.get('bodyparts', [])
        
        # Determine which bodyparts to process
        if config_bodyparts:
            # Filter to only requested bodyparts
            bodyparts = [bp for bp in available_bodyparts if bp in config_bodyparts]
            if not bodyparts:
                self.logger.warning(
                    f"None of requested bodyparts {config_bodyparts} found in pose data. "
                    f"Available: {available_bodyparts}"
                )
                return dataframe
        else:
            # Process all available bodyparts
            bodyparts = available_bodyparts
        
        self.logger.info(f"Processing map coordinates for {len(bodyparts)} bodyparts")
        
        try:
            # Use utility function to transform coordinates
            result_df = apply_coordinate_transform_to_bodyparts(
                dataframe, 
                bodyparts, 
                calibration_matrix, 
                output_suffix="map"
            )
            
            self.logger.info(f"Map location complete: {len(bodyparts)} bodyparts processed")
            return result_df
            
        except Exception as e:
            raise NavigraphError(f"Failed to transform coordinates: {str(e)}") from e
    
    def _detect_pose_bodyparts(self, dataframe: pd.DataFrame) -> List[str]:
        """Detect available bodyparts from pose tracking columns.
        
        Args:
            dataframe: DataFrame to analyze
            
        Returns:
            List of bodypart names found
        """
        bodyparts = set()
        
        # Look for _x columns and extract bodypart names
        for col in dataframe.columns:
            if col.endswith('_x'):
                bodypart = col[:-2]  # strip '_x'
                # Skip derived columns from other plugins (e.g. nose_map_x)
                if '_map' in bodypart or '_calibrated' in bodypart or '_graph' in bodypart:
                    continue
                if f'{bodypart}_y' in dataframe.columns:
                    bodyparts.add(bodypart)
        
        return sorted(list(bodyparts))
    
    def get_expected_columns(self) -> List[str]:
        """Generate expected column names based on configuration.
        
        Returns:
            List of column names this plugin will create
        """
        columns = []
        
        # Get configured bodyparts
        bodyparts = self.config.get('bodyparts', [])
        
        # If empty, we don't know until runtime
        if not bodyparts:
            return []
        
        # Add map coordinate columns for each bodypart
        for bp in bodyparts:
            columns.extend([f"{bp}_map_x", f"{bp}_map_y"])
        
        return columns