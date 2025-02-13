import cv2
from omegaconf import DictConfig
import hydra
import numpy as np


def get_tile_bbox_id_by_row_col(cfg: DictConfig, row: int, col: int):
    segment_length = cfg.map_settings.segment_length
    origin_row, origin_col = eval(cfg.map_settings.origin)
    grid_size = eval(cfg.map_settings.grid_size)
    max_grid_row, max_grid_col = np.array(grid_size) * segment_length

    if row < origin_row or col < origin_col or row > max_grid_row or col > max_grid_col:
        return None, -1

    row_origin_offset = row - origin_row
    col_origin_offset = col - origin_col

    row_segment_multiplier = row_origin_offset // segment_length
    col_segment_multiplier = col_origin_offset // segment_length

    # bbox format: x_min, y_min, x_max, y_max
    y_min = (segment_length * row_segment_multiplier) + origin_row
    x_min = (segment_length * col_segment_multiplier) + origin_col
    x_max = x_min + segment_length - 1
    y_max = y_min + segment_length - 1
    x, y, w, h = [x_min, y_min, x_max - x_min, y_max - y_min]

    tile_id = (grid_size[0] * row_segment_multiplier) + col_segment_multiplier

    return [x, y, w, h], tile_id


def draw_tile_id_on_map(cfg: DictConfig):

    map_img = cv2.imread(cfg.map_path)
    printed_tile_ids = set()
    search_interval = int(cfg.map_settings.segment_length/2)
    for row in range(0, map_img.shape[0], search_interval):
        for col in range(0, map_img.shape[1], search_interval):

            bbox, tile_id = get_tile_bbox_id_by_row_col(cfg, row, col)

            if tile_id in printed_tile_ids or bbox is None:
                continue
            else:
                x, y, w, h = bbox
                printed_tile_ids.add(tile_id)

                # draw
                cv2.putText(map_img, f'{tile_id}', (int(x + (w/2)) - 10, int(y + (h/2)) + 10), cv2.FONT_HERSHEY_TRIPLEX,
                            1, (0, 0, 255), 2)

    cv2.imshow('tile id mapping', map_img)
    cv2.waitKey(0)


@hydra.main(config_path="../configs", config_name="maze_master_basic")
def main(cfg: DictConfig):
    draw_tile_id_on_map(cfg)


if __name__ == '__main__':
    main()
