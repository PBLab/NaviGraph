import cv2
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import hydra
import logging as lg
import os


log = lg.getLogger(__name__)


class MapLabeler(object):
    def __init__(self, cfg: DictConfig, transform_matrix=None, path_to_img=None):

        if cfg.verbose:
            log.setLevel(lg.DEBUG)

        self._cfg = cfg
        # TODO: set this as a proper attribute + handle cases where this is None
        self.pixel_to_meter = cfg.map_settings.get('pixel_to_meter', None)

        if path_to_img is not None:
            self.map_img = cv2.imread(path_to_img)
        else:
            self.map_img = cv2.imread(self.cfg.map_path)

        if transform_matrix is None:
            pre_calculated_transform_matrix_path = self.cfg.calibrator_parameters.pre_calculated_transform_matrix_path
            if pre_calculated_transform_matrix_path is None:
                raise ValueError('A valid transform matrix from either pre calculated path or as an argument is a must!')
            elif os.path.isfile(pre_calculated_transform_matrix_path):
                self.transform_matrix = np.load(self.cfg.calibrator_parameters.pre_calculated_transform_matrix_path)
            else:
                raise ValueError(f'Can not find calibration files at given path: {pre_calculated_transform_matrix_path}')
        else:
            self.transform_matrix = transform_matrix

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    def get_map_coords(self, row: int, col: int) -> tuple:

        map_col, map_row = cv2.perspectiveTransform(np.array([[(col, row)]], dtype='float32'), self.transform_matrix). \
            ravel().astype(int)
        if (map_row is not None and 0 <= map_row < self.map_img.shape[0]) and\
                (map_col is not None and 0 <= map_col < self.map_img.shape[1]):

            return map_col, map_row

        else:
            return -1, -1

    def get_tile_by_map_coords(self, map_col, map_row):
        # Handle edge cases
        origin_row, origin_col = eval(self.cfg.map_settings.origin)
        max_grid_row, max_grid_col = (np.array(eval(self.cfg.map_settings.grid_size)) + 1) * self.cfg.map_settings.segment_length
        if map_row < origin_row or map_col < origin_col or map_row > max_grid_row or map_col > max_grid_col:
            return None, -1

        row_origin_offset = map_row - origin_row
        col_origin_offset = map_col - origin_col

        row_segment_multiplier = row_origin_offset // self.cfg.map_settings.segment_length
        col_segment_multiplier = col_origin_offset // self.cfg.map_settings.segment_length

        # bbox format: x_min, y_min, x_max, y_max
        y_min = (self.cfg.map_settings.segment_length * row_segment_multiplier) + origin_row
        x_min = (self.cfg.map_settings.segment_length * col_segment_multiplier) + origin_col
        x_max = x_min + self.cfg.map_settings.segment_length - 1
        y_max = y_min + self.cfg.map_settings.segment_length - 1

        tile_id = (eval(self.cfg.map_settings.grid_size)[0] * row_segment_multiplier) + col_segment_multiplier

        return [x_min, y_min, x_max - x_min, y_max - y_min], tile_id  # return  [x, y, w, h] format

    def get_tile_by_img_coords(self, row: int, col: int) -> tuple:

        map_col, map_row = self.get_map_coords(row, col)
        return self.get_tile_by_map_coords(map_col, map_row)

    def draw_tiles_ids_on_map(self):
        """
        This function plots the tile_id on the map for verification & debugging purposes
        :return:
        """
        tile_id_history = []
        map_copy = self.map_img.copy()
        for col in range(0, self.map_img.shape[1], self.cfg.map_settings.segment_length//2):
            for row in range(0, self.map_img.shape[0], self.cfg.map_settings.segment_length//2):
                bbox, tile_id = self.get_tile_by_map_coords(map_col=col, map_row=row)
                if tile_id in tile_id_history or tile_id == -1:
                    continue
                else:
                    tile_id_history.append(tile_id)
                    [x, y, w, h] = bbox
                    map_copy = cv2.putText(map_copy, str(tile_id), (int(x + w/2), int(y + h/2)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        plt.imshow(map_copy)
        plt.show()


@hydra.main(config_path="../configs", config_name="maze_master_basic")
def main(cfg: DictConfig):
    transform_matrix = np.load('/home/elior/PycharmProjects/maze_analysis/demo_data/calibration_files/transform_matrix.npy')
    path_to_img = '/home/elior/PycharmProjects/maze_analysis/demo_data/maze_map.png'

    map_labler = MapLabeler(cfg, transform_matrix=transform_matrix, path_to_img=path_to_img)
    bbox, id = map_labler.get_tile_by_img_coords(220, 220)

    map_labler.draw_tiles_ids_on_map()


if __name__ == '__main__':
    main()


