# map_datasource.py

import cv2
import numpy as np
import os
import logging
import pandas as pd
from typing import Tuple, Optional, Any
from omegaconf import DictConfig
from navigraph.session.session_datasource import SessionDataSource

logger = logging.getLogger(__name__)


class TileMapper(SessionDataSource):
    """
    A datasource that augments the session DataFrame with map tile information.

    This class assumes that a valid transformation matrix and map image are provided
    (by the Session or calibrator module). It converts pixel coordinates to map coordinates,
    then determines the tile (region) in which the coordinates lie based on configuration.

    The augmentation adds two columns:
      - 'tile_id': The computed tile identifier.
      - 'tile_bbox': The bounding box [x, y, width, height] of the tile.
    """

    def __init__(self,
                 cfg: DictConfig,
                 transform_matrix: np.ndarray,
                 map_img: np.ndarray,
                 coord_columns: Tuple[str, str] = ("x", "y"),
                 logger: Optional[logging.Logger] = None,
                 **kwargs: Any) -> None:
        """
        Initialize the MapTileDatasource.

        Args:
            cfg (DictConfig): Map and calibration settings.
            transform_matrix (np.ndarray): Transformation matrix mapping pixel to map coordinates.
            map_img (np.ndarray): Loaded map image.
            coord_columns (Tuple[str, str], optional): Names of DataFrame columns containing pixel coordinates.
            logger (Optional[logging.Logger]): Logger instance. If not provided, a default is used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(logger=logger, **kwargs)
        self.cfg: DictConfig = cfg
        self.coord_columns: Tuple[str, str] = coord_columns
        self.transform_matrix: np.ndarray = transform_matrix
        self.map_img: np.ndarray = map_img

        # Map settings from configuration.
        self.pixel_to_meter: Optional[float] = cfg.map_settings.get("pixel_to_meter", None)
        self.origin: Tuple[int, int] = eval(cfg.map_settings.origin)  # expects string like "(0, 0)"
        self.grid_size: Tuple[int, int] = tuple(np.array(eval(cfg.map_settings.grid_size)) + 1)
        self.segment_length: int = cfg.map_settings.get("segment_length")
        self.logger.info("MapTileDatasource initialized: origin=%s, grid_size=%s, segment_length=%d",
                         self.origin, self.grid_size, self.segment_length)

    def get_map_coords(self, row: int, col: int) -> Tuple[int, int]:
        """
        Convert pixel coordinates (row, col) to map coordinates using the transformation matrix.

        Args:
            row (int): Pixel row.
            col (int): Pixel column.

        Returns:
            Tuple[int, int]: Map coordinates (map_col, map_row), or (-1, -1) if outside map bounds.
        """
        pts = np.array([[(col, row)]], dtype="float32")
        transformed = cv2.perspectiveTransform(pts, self.transform_matrix)
        map_coords = tuple(transformed.ravel().astype(int))
        if (0 <= map_coords[1] < self.map_img.shape[0]) and (0 <= map_coords[0] < self.map_img.shape[1]):
            return map_coords
        else:
            return (-1, -1)

    def get_tile_by_map_coords(self, map_col: int, map_row: int) -> Tuple[Optional[list], int]:
        """
        Given map coordinates, compute the tile bounding box and tile id.

        Args:
            map_col (int): Map column coordinate.
            map_row (int): Map row coordinate.

        Returns:
            Tuple[Optional[list], int]: The tile bounding box [x, y, width, height] and tile id,
            or (None, -1) if the coordinates fall outside the defined grid.
        """
        origin_row, origin_col = self.origin
        max_grid_row = self.grid_size[1] * self.segment_length
        max_grid_col = self.grid_size[0] * self.segment_length

        if map_row < origin_row or map_col < origin_col or map_row > max_grid_row or map_col > max_grid_col:
            return None, -1

        row_offset = map_row - origin_row
        col_offset = map_col - origin_col
        row_seg = row_offset // self.segment_length
        col_seg = col_offset // self.segment_length
        x_min = origin_col + (col_seg * self.segment_length)
        y_min = origin_row + (row_seg * self.segment_length)
        width = self.segment_length
        height = self.segment_length
        tile_id = (self.grid_size[0] * row_seg) + col_seg
        return ([x_min, y_min, width, height], tile_id)

    def get_tile_by_img_coords(self, row: int, col: int) -> Tuple[Optional[list], int]:
        """
        Convert pixel coordinates to map coordinates and determine the corresponding tile.

        Args:
            row (int): Pixel row.
            col (int): Pixel column.

        Returns:
            Tuple[Optional[list], int]: Tile bounding box and tile id.
        """
        map_coords = self.get_map_coords(row, col)
        return self.get_tile_by_map_coords(*map_coords)

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the DataFrame with tile mapping information.

        Expects the DataFrame to contain columns specified by self.coord_columns (default: "x" and "y").
        Two new columns are added:
          - 'tile_id': The computed tile identifier.
          - 'tile_bbox': The bounding box [x, y, width, height] of the tile.

        Returns:
            pd.DataFrame: The augmented DataFrame.
        """
        x_col, y_col = self.coord_columns

        def compute_tile(row: pd.Series) -> pd.Series:
            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except (TypeError, ValueError):
                return pd.Series({"tile_id": -1, "tile_bbox": None})
            map_coords = self.get_map_coords(int(y), int(x))
            bbox, tile_id = self.get_tile_by_map_coords(*map_coords)
            return pd.Series({"tile_id": tile_id, "tile_bbox": bbox})

        self.logger.info("Augmenting DataFrame with tile mapping using columns: %s and %s", x_col, y_col)
        df_aug = df.copy()
        tile_info = df_aug.apply(compute_tile, axis=1)
        df_aug = pd.concat([df_aug, tile_info], axis=1)
        self.logger.info("DataFrame augmented with tile mapping.")
        return df_aug

    def load_data(self, file_path: str) -> Any:
        raise NotImplementedError("MapTileDatasource does not load data from file directly.")

    @classmethod
    def from_config(cls, config: dict) -> "MapTileDatasource":
        """
        Create a MapTileDatasource from a configuration dictionary.

        Expected keys:
          - map_settings: with 'pixel_to_meter', 'origin', 'grid_size', 'segment_length'
          - map_path: path to the map image.
          - calibrator_parameters.pre_calculated_transform_matrix_path: path to the transform matrix file.
          - Optional: coord_columns (default: ["x", "y"])

        Returns:
            MapTileDatasource: An instance of the datasource.
        """
        from omegaconf import OmegaConf
        cfg = OmegaConf.create(config)
        map_path = cfg.get("map_path")
        map_img = cv2.imread(map_path)
        if map_img is None:
            raise ValueError(f"Could not load map image from {map_path}")
        tm_path = cfg.calibrator_parameters.pre_calculated_transform_matrix_path
        if not os.path.isfile(tm_path):
            raise ValueError(f"Transformation matrix file not found: {tm_path}")
        transform_matrix = np.load(tm_path)
        coord_columns = tuple(cfg.get("coord_columns", ("x", "y")))
        return cls(cfg, transform_matrix, map_img, coord_columns=coord_columns)

    def initialize(self) -> None:
        super().initialize()
        self.logger.info("MapTileDatasource initialized with transformation matrix and map image.")
