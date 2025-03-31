from omegaconf import DictConfig
import os
from navigraph.session import Session
import logging as lg
import cv2
import numpy as np

from .utils import overlay_img, get_default_args

log = lg.getLogger(__name__)

VISUALIZATION_KEY = 'visualization'
RECORD_VISUALIZATION_KEY = 'record_visualization'
FPS_KEY = 'fps'
SHOW_VISUALIZATION_KEY = 'show_visualization'
SHOW_KEY = 'show'
RESIZE_FACTOR_KEY = 'resize_factor'
OUTPUT_RESIZE_KEY = 'resize'
OPACITY_KEY = 'opacity'
METHOD_KEY = 'method'
FRAME_LOCATION_KEY = 'frame_location'
DRAW_MAP_KEY = 'draw_map'
DRAW_TREE_KEY = 'draw_tree'
SHOW_TREE_ONLY = 'show_tree_only'


class Visualizer(object):
    def __init__(self, cfg: DictConfig
                 ):
        if cfg.verbose:
            log.setLevel(lg.DEBUG)

        self._cfg = cfg
        self._overlay_default_arguments = get_default_args(overlay_img)

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    def visualize_session(self, path_to_session_stream: str, session: Session) -> None:

        cap = cv2.VideoCapture(path_to_session_stream)
        resize = self.cfg.get(VISUALIZATION_KEY, {}).get(OUTPUT_RESIZE_KEY,
                                                         (cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        fps = self.cfg.get(VISUALIZATION_KEY, {}).get(FPS_KEY)
        if self.cfg.get(VISUALIZATION_KEY, {}).get(RECORD_VISUALIZATION_KEY, False):
            writer = cv2.VideoWriter(filename=os.path.join(self.cfg.experiment_output_path, 'processed__' +
                                                           os.path.basename(path_to_session_stream)),
                                     fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps=fps if fps is not None else cap.get(cv2.CAP_PROP_FPS),
                                     frameSize=resize)
        counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                cap.release()
                cv2.destroyAllWindows()
                break

            # retrieve subject coordinate and location confidence
            x, y, likelihood = session.get_coords(counter, self.cfg.location_settings.bodypart)

            if x is not None and y is not None:
                # Draw desired body-part location on image
                cv2.circle(frame, (int(x), int(y)),
                           self.cfg.calibrator_parameters.points_capture_parameters.radius,
                           eval(self.cfg.calibrator_parameters.points_capture_parameters.color),
                           self.cfg.calibrator_parameters.points_capture_parameters.thickness)

            # Draw body-part on map (currently not in use)
            # map_y, map_x = session.get_map_coords(counter, self.cfg.location_settings.bodypart)
            map_copy = session.map_labeler.map_img.copy()

            # Draw tile on map
            map_tile_bbox, tile_id = session.get_map_tile(counter, self.cfg.location_settings.bodypart)
            if map_tile_bbox is not None:
                map_tile = map_copy[map_tile_bbox[1]:map_tile_bbox[1] + map_tile_bbox[3],
                                    map_tile_bbox[0]:map_tile_bbox[0] + map_tile_bbox[2]]

                # Draw map location rectangle
                rect = np.ones(map_tile.shape, dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)
                res = cv2.addWeighted(map_tile, 0.5, rect, 0.5, 1.0)
                map_copy[map_tile_bbox[1]:map_tile_bbox[1] + map_tile_bbox[3],
                         map_tile_bbox[0]:map_tile_bbox[0] + map_tile_bbox[2]] = res

                # Draw tile_id as text above the map location
                cv2.putText(map_copy, f'{tile_id}', (map_tile_bbox[0], map_tile_bbox[1] - 5), cv2.FONT_HERSHEY_TRIPLEX,
                            3, (0, 0, 255), 2)

            # Draw map on image
            if self.cfg.get(VISUALIZATION_KEY, None):
                if self.cfg.visualization.get(DRAW_MAP_KEY, None) and self.cfg.visualization.draw_map.get(SHOW_KEY, None):
                    frame = overlay_img(frame,
                                        map_copy,
                                        resize_factor=self.cfg.visualization.draw_map.get(
                                            RESIZE_FACTOR_KEY, self._overlay_default_arguments[RESIZE_FACTOR_KEY]),
                                        opacity=self.cfg.visualization.draw_map.get(
                                            OPACITY_KEY, self._overlay_default_arguments[OPACITY_KEY]),
                                        method=self.cfg.visualization.draw_map.get(
                                            METHOD_KEY, self._overlay_default_arguments[METHOD_KEY]),
                                        frame_location=self.cfg.visualization.draw_map.get(
                                            FRAME_LOCATION_KEY, self._overlay_default_arguments[FRAME_LOCATION_KEY]))

                # Draw tree on image (This option is currently very inefficient and extremely slow thus disabled in
                # default configuration)
                if self.cfg.visualization.get(DRAW_TREE_KEY, None) and self.cfg.visualization.draw_tree.get(SHOW_KEY, None):
                    frame = overlay_img(frame,
                                        cv2.cvtColor(session.draw_tree(tile_id), cv2.COLOR_RGB2BGR),
                                        resize_factor=self.cfg.visualization.draw_tree.get(
                                            RESIZE_FACTOR_KEY, self._overlay_default_arguments[RESIZE_FACTOR_KEY]),
                                        opacity=self.cfg.visualization.draw_tree.get(
                                            OPACITY_KEY, self._overlay_default_arguments[OPACITY_KEY]),
                                        method=self.cfg.visualization.draw_tree.get(
                                            METHOD_KEY, self._overlay_default_arguments[METHOD_KEY]),
                                        frame_location=self.cfg.visualization.draw_tree.get(
                                            FRAME_LOCATION_KEY, self._overlay_default_arguments[FRAME_LOCATION_KEY]))

                    if self.cfg.visualization.draw_tree.get(SHOW_TREE_ONLY, None):
                        frame = cv2.cvtColor(session.draw_tree(tile_id), cv2.COLOR_RGB2BGR)

                # Resize for display
                if self.cfg.visualization.resize:
                    frame = cv2.resize(frame,
                                       self.cfg.visualization.resize,
                                       interpolation=cv2.INTER_AREA)

            if self.cfg.get(VISUALIZATION_KEY, {}).get(RECORD_VISUALIZATION_KEY, False):
                writer.write(frame)

            if self.cfg.get(VISUALIZATION_KEY, {}).get(SHOW_VISUALIZATION_KEY, False):
                cv2.imshow('frame', frame)

            counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if self.cfg.get(VISUALIZATION_KEY, {}).get(RECORD_VISUALIZATION_KEY, False):
            writer.release()
        cv2.destroyAllWindows()
