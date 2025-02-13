from __future__ import annotations

import os
import numpy as np
import cv2
from omegaconf import DictConfig
from typing import Optional, Tuple, Any
import pandas as pd
import hydra
import logging as lg
from map.map_labeler import MapLabeler
from graph.graph import Graph
import queue

TILE_BBOX = 'tile_box'
TILE_ID = 'tile_id'
LIKELIHOOD = 'likelihood'
TREE_POSITION = 'tree_position'
FPS = 'fps'
REWARD_TILE_ID = 'reward_tile_id'


log = lg.getLogger(__name__)


class Session(object):
    def __init__(self, cfg: DictConfig, stream_path: str, keypoint_detection_file_path: str, transform_matrix=None):
        if cfg.verbose:
            log.setLevel(lg.DEBUG)

        self._cfg = cfg
        self.data = pd.read_hdf(keypoint_detection_file_path)
        self._session_id = list(self.data.columns.levels[0])
        self._session_id = self._session_id if len(self._session_id) > 1 else self._session_id[0]
        self._session_name = os.path.basename(keypoint_detection_file_path).split('.')[0]
        self._bodyparts = list(self.data.columns.levels[1])
        self._likelihood = self.cfg.location_settings.get(LIKELIHOOD, None)
        self._coords = list(self.data.columns.levels[2])
        self.map_labeler = MapLabeler(self.cfg, transform_matrix)
        self.tree = Graph(self.cfg)

        # find path from source to reward give reward tile_id
        reward_tile_id = self.cfg.get(REWARD_TILE_ID, None)
        if reward_tile_id is not None:
            tree_loc = self.tree.get_tree_location(reward_tile_id)
            if isinstance(tree_loc, int):
                reward_node_id = tree_loc
            elif isinstance(tree_loc, tuple):
                raise ValueError('tile_id must be related to a node, not an edge, in order to calculate shortest path '
                                 'to reward')
            else:
                reward_node_id = [item for item in tree_loc if isinstance(item, int)][0]
            self.path_to_reward = self.tree.get_shortest_path(source=0, target=reward_node_id)

        self.session_stream_info = self._get_session_stream_info(stream_path)
        self.session_history = queue.Queue()

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def session_name(self) -> str:
        return self._session_name

    @property
    def bodyparts(self) -> list:
        return self._bodyparts

    @property
    def likelihood(self) -> float:
        return self._likelihood

    @property
    def coords(self) -> list:
        return self._coords

    def get_coords(self, frame_number, body_part) -> tuple[int, int, float] | Any:
        if body_part not in self.bodyparts:
            raise KeyError('Required bodypart not documneted in session. Use Session.bodyparts to see available '
                           'bodyparts')

        if frame_number not in self.data.index:
            return None, None, None

        x, y, likelihood = self.data.loc[frame_number, self.session_id][body_part]

        if self.likelihood is not None and likelihood < self.likelihood:
            return None, None, None
        else:
            return x, y, likelihood

    def get_map_coords(self,  frame_number, body_part):
        x, y, _ = self.get_coords(frame_number, body_part)
        return self.map_labeler.get_map_coords(row=y, col=x)

    def get_map_tile(self, frame_number, body_part):

        x, y, likelihood = self.get_coords(frame_number, body_part)
        if x is None or y is None:
            return None, None

        return self.map_labeler.get_tile_by_img_coords(row=y, col=x)

    def get_df(self, body_part) -> pd.DataFrame:
        # TODO: add tree node\edge here - after implementing the tile-node\edge dictionary (as data class ?)
        df = self.data[self.session_id][body_part]
        # Add tile data
        tile_data = pd.Series(df.apply(lambda row: self.map_labeler.get_tile_by_img_coords(row=row.y, col=row.x), axis=1))
        df[[TILE_BBOX, TILE_ID]] = pd.DataFrame(tile_data.tolist(), index=df.index)

        # Add node data
        df[TREE_POSITION] = pd.Series(df.apply(lambda row: None if row[TILE_ID] == -1 else self.tree.get_tree_location(row[TILE_ID]), axis=1))

        if self.likelihood is not None:
            df[df[LIKELIHOOD] < self.likelihood] = np.NaN

        return df

    def insert_data(self, body_part, col_name, col_values):
        try:
            self.data[(self.session_id, body_part, col_name)] = col_values
        except Exception as e:
            raise e


    def _get_session_stream_info(self, path):

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        return {'fps': fps, 'frame_count': frame_count, 'duration': duration}

    def draw_tree(self, tile_id, mode='current'):
        """
        This function gets a tile_id which can be a number or a list of numbers and returns the tree image with desired
        edges\nodes in the specified colors
        :return:
        """

        # recursively draw history
        if self.session_history.qsize() > 0:
            self.draw_tree(self.session_history.get(), mode='history')

        if not mode == 'history':
            self.session_history.put(tile_id)

        if tile_id == -1:
            node, edge = None, None
        else:

            tree_loc = self.tree.get_tree_location(tile_id)
            if isinstance(tree_loc, tuple):
                edge = [tree_loc]
                node = None
            elif isinstance(tree_loc, int):
                edge = None
                node = [tree_loc]
            elif isinstance(tree_loc, frozenset):
                for item in tree_loc:
                    if isinstance(item, tuple):
                        edge = [item]
                    elif isinstance(item, int):
                        node = [item]
            else:
                node, edge = None, None
        unique_path = self.path_to_reward + [(node_1, node_2) for node_1, node_2 in
                                             zip(self.path_to_reward, self.path_to_reward[1:])]
        self.tree.draw_tree(node_list=node, edge_list=edge, color_mode=mode, unique_path=unique_path)
        return self.tree.tree_fig_to_img()


@hydra.main(config_path="../configs", config_name="maze_master_basic")
def main(cfg: DictConfig):

    session = Session(cfg, cfg.keypoint_detection_file_path)
    session.get_map_tile(5, 'nose')
    session.get_df('nose')


if __name__ == '__main__':
    main()


