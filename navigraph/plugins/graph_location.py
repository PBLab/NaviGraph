"""Graph location plugin for NaviGraph unified architecture.

Maps pose coordinates to graph nodes and edges using spatial mapping.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple

from ..core.navigraph_plugin import NaviGraphPlugin
from ..core.exceptions import NavigraphError
from ..core.coordinate_transform import apply_coordinate_transform_to_bodyparts
from ..core.registry import register_data_source_plugin


@register_data_source_plugin("graph_location")
class GraphLocationPlugin(NaviGraphPlugin):
    """Maps bodypart coordinates to graph nodes and edges."""
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            'name': self.config.get('name', 'graph_location'),
            'type': 'graph_location',
            'description': 'Maps pose coordinates to graph nodes and edges using spatial mapping',
            'provides': [],
            'augments': 'bodypart_graph_node, bodypart_graph_edge columns'
        }
    
    def augment_data(self, dataframe: pd.DataFrame, shared_resources: Dict[str, Any]) -> pd.DataFrame:
        """Add graph location columns for tracked bodyparts.
        
        Process: pose → calibration → map coordinates → graph mapping → graph location
        
        Args:
            dataframe: DataFrame with pose tracking data
            shared_resources: Available shared resources
            
        Returns:
            DataFrame with graph location columns added
        """
        # Validate required resources
        required_resources = ['calibration_matrix', 'graph', 'graph_mapping']
        missing = [r for r in required_resources if r not in shared_resources]
        if missing:
            raise NavigraphError(
                f"GraphLocationPlugin requires: {missing}. "
                f"Available resources: {list(shared_resources.keys())}"
            )
        
        calibration_matrix = shared_resources['calibration_matrix']
        graph = shared_resources['graph']
        graph_mapping = shared_resources['graph_mapping']
        
        # Validate graph mapping is not None
        if graph_mapping is None:
            raise NavigraphError(
                "Graph mapping is None. Check that spatial mapping was loaded correctly."
            )
        
        # Detect available bodyparts from pose data
        available_bodyparts = self._detect_pose_bodyparts(dataframe)
        
        if not available_bodyparts:
            raise NavigraphError(
                "GraphLocationPlugin requires pose tracking data. "
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
        
        self.logger.info(f"Processing graph locations for {len(bodyparts)} bodyparts")
        
        try:
            # Step 1: Transform coordinates to map space using calibration
            transformed_df = apply_coordinate_transform_to_bodyparts(
                dataframe, 
                bodyparts, 
                calibration_matrix, 
                output_suffix="calibrated"  # Temporary columns for mapping
            )
            
            # Step 2: Map coordinates to graph elements for each bodypart
            for bodypart in bodyparts:
                calibrated_x_col = f'{bodypart}_calibrated_x'
                calibrated_y_col = f'{bodypart}_calibrated_y'
                
                # Get calibrated coordinates
                x_coords = transformed_df[calibrated_x_col].values
                y_coords = transformed_df[calibrated_y_col].values
                
                # Map to graph elements
                nodes, edges = self._map_coordinates_to_graph(
                    x_coords, y_coords, graph_mapping
                )
                
                # Add graph location columns
                transformed_df[f'{bodypart}_graph_node'] = nodes
                transformed_df[f'{bodypart}_graph_edge'] = edges
                
                # Remove temporary calibrated columns
                transformed_df.drop([calibrated_x_col, calibrated_y_col], axis=1, inplace=True)
                
                self.logger.debug(f"Mapped {bodypart} to graph locations")
            
            self.logger.info(f"Graph location complete: {len(bodyparts)} bodyparts processed")
            return transformed_df
            
        except Exception as e:
            raise NavigraphError(f"Failed to map to graph locations: {str(e)}") from e
    
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
                # Skip derived columns from other plugins (e.g. nose_map_x from map_location)
                if '_map' in bodypart or '_calibrated' in bodypart:
                    continue
                if f'{bodypart}_y' in dataframe.columns:
                    bodyparts.add(bodypart)
        
        return sorted(list(bodyparts))
    
    def _map_coordinates_to_graph(
        self, 
        x_coords: np.ndarray, 
        y_coords: np.ndarray, 
        graph_mapping: Any
    ) -> Tuple[List[Optional[int]], List[Optional[Tuple[int, int]]]]:
        """Map coordinates to graph nodes and edges using spatial mapping.
        
        Args:
            x_coords: Array of x coordinates in map space
            y_coords: Array of y coordinates in map space
            graph_mapping: SpatialMapping instance
            
        Returns:
            Tuple of (node_list, edge_list) where each list has same length as input coordinates
        """
        nodes = []
        edges = []
        
        for x, y in zip(x_coords, y_coords):
            if np.isnan(x) or np.isnan(y):
                # Invalid coordinates
                nodes.append(None)
                edges.append(None)
                continue
            
            try:
                # Use graph mapping to find node and edge
                node, edge = graph_mapping.map_point_to_elements(float(x), float(y))
                
                # Store results
                nodes.append(node)
                edges.append(edge)
                
            except Exception as e:
                # Mapping failed for this point
                self.logger.debug(f"Mapping failed for point ({x}, {y}): {str(e)}")
                nodes.append(None)
                edges.append(None)
        
        return nodes, edges
    
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
        
        # Add graph location columns for each bodypart
        for bp in bodyparts:
            columns.extend([f"{bp}_graph_node", f"{bp}_graph_edge"])
        
        return columns