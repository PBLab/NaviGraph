import os
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import logging as lg
from navigraph.session import FPS, TILE_ID, TREE_POSITION
from navigraph.session import Session
from .utils import a_to_b, count_unique_type_specific_objects,\
    count_node_visits_eliminating_sequences, Condition
import operator
from functools import partial

log = lg.getLogger(__name__)

FUNC_NAME = 'func_name'
FUNC_ARGS = 'args'
ANALYZE_CFG_KEY = 'analyze'
METRICS_KEY = 'metrics'
SAVE_CSV = 'save_as_csv'
SAVE_PKL = 'save_as_pkl'
OUTPUT_PATH = 'experiment_output_path'
OUTPUT_FILE_NAME = 'analysis_results'
SAVE_RAW = 'save_raw_data_as_pkl'
DEFAULT_MIN_NODES_ON_PATH = 0
NODE_PATH_TOPOLOGICAL_MODE = 'topological_distance'
NODE_PATH_EXPLORATION_MODE = 'exploration'
AVG_NODE_TIME_BY_PATH = 'by_path'
AVG_NODE_TIME_OVERALL = 'overall'
TIME_FROM_SESSION_START = 'time_from_session_start'
TIME_FROM_ROUND_START = 'time_from_round_start'
DISTANCE_FROM_REWARD = 'distance_from_reward'

NODE_POSSIBLE_DTYPES = (int, frozenset)

# helper condition - being used by several methods
count_unique_nodes = partial(count_unique_type_specific_objects, dtype=NODE_POSSIBLE_DTYPES)
condition = Condition(column_name=TREE_POSITION,
                      func=count_unique_nodes,
                      threshold=DEFAULT_MIN_NODES_ON_PATH,
                      operator=operator.gt)


def analyze_session(func):
    def wrapper(*args, **kwargs):
        result_dict = {}
        session_list = args[0].session_list

        for sess in session_list:
            result_dict[sess.session_name] = func(*args, sess=sess, **kwargs)

        return result_dict

    return wrapper


class SessionAnalyzer(object):
    def __init__(self, cfg: DictConfig, session_list: list):
        if cfg.verbose:
            log.setLevel(lg.DEBUG)

        self._cfg = cfg
        self.session_list = session_list

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    @analyze_session
    def time_a_to_b(self, sess: Session, a: int, b: int, min_nodes_on_path: int = 0, insert_trial_number = True):
        """
        Return a list of time intervals in seconds between a and b. The output time intervals are ordered meaning, The
        first value for example is the time between t1: first arrival to a and t2: the first arrival to b  such that
        t2>t1. Additionally, time intervals can be filtered out using a minimal requirement on path length using
        min_nodes_on_path > 0

        :param sess:
        :param a:
        :param b:
        :param min_nodes_on_path:
        :param insert_trial_number:
        :return:
        """
        df = sess.get_df(self.cfg.location_settings.bodypart)
        condition.threshold = min_nodes_on_path
        a_to_b_indices = a_to_b(df, TILE_ID, a, b, condition)

        # Add trial number
        if insert_trial_number:
            col_name = 'trial'
            df[col_name] = np.nan
            for i, (start, end) in enumerate(a_to_b_indices, start=1):
                df.loc[start:end, col_name] = i

            sess.insert_data(self.cfg.location_settings.bodypart, col_name, df[col_name])

        times_between_a_to_b = [(ind_b + 1 - ind_a) * (1 / sess.session_stream_info[FPS]) for (ind_a, ind_b) in
                                a_to_b_indices]

        return times_between_a_to_b

    @analyze_session
    def velocity_a_to_b(self, sess: Session, a: int, b: int, min_nodes_on_path: int = 0):
        """
        Return a list of velocities in meter/sec that describes the average velocity all trajectories
        between tile a and b.
        :param sess:
        :param a:
        :param b:
        :return:
        """
        df = sess.get_df(self.cfg.location_settings.bodypart)
        condition.threshold = min_nodes_on_path
        a_to_b_indices = a_to_b(df, TILE_ID, a, b, condition)

        times_between_a_to_b = [(ind_b + 1 - ind_a) * (1 / sess.session_stream_info[FPS]) for (ind_a, ind_b) in
                                a_to_b_indices]

        path_traveled_between_a_to_b = []
        for ind_a, ind_b in a_to_b_indices:
            path_segments_length_pixels = ((df.iloc[ind_a:ind_b + 1].x.diff() ** 2) +
                                           (df.iloc[ind_a:ind_b + 1].y.diff() ** 2)) ** 0.5
            total_path_length_meters = path_segments_length_pixels.sum() / sess.map_labeler.pixel_to_meter
            path_traveled_between_a_to_b.append(total_path_length_meters)

        return list(np.array(path_traveled_between_a_to_b)/np.array(times_between_a_to_b))

    @analyze_session
    def num_nodes_in_path(self, sess: Session, a: int, b: int, min_nodes_on_path: int = 0,
                          mode=NODE_PATH_TOPOLOGICAL_MODE, min_frame_rep=1):
        """

        :param sess:
        :param a:
        :param b:
        :param min_nodes_on_path:
        :param mode:
        :return:
        """
        df = sess.get_df(self.cfg.location_settings.bodypart)
        condition.threshold = min_nodes_on_path
        a_to_b_indices = a_to_b(df, TILE_ID, a, b, condition)

        num_nodes_in_path = []
        for ind_a, ind_b in a_to_b_indices:
            if mode == NODE_PATH_TOPOLOGICAL_MODE:
                tree_frame_position = df[ind_a:ind_b + 1][TREE_POSITION]
                num_nodes_in_path.append(count_node_visits_eliminating_sequences(tree_frame_position))
            else:
                num_nodes_in_path.append(count_unique_type_specific_objects(
                    df[ind_a:ind_b + 1][TREE_POSITION], NODE_POSSIBLE_DTYPES) / sess.tree.tree.number_of_nodes())

        return num_nodes_in_path

    @analyze_session
    def exploration_percentage(self, sess: Session):
        """
        Return the percentage of the nodes covered per frame
        :param sess:
        :return:
        """
        def keep_node_positions(x):
            if type(x) == int:
                return x
            elif isinstance(x, frozenset):
                return [i for i in x if type(i) == int][0]
            else:
                return None

        df = sess.get_df(self.cfg.location_settings.bodypart)
        nodes_visited = df[TREE_POSITION].apply(keep_node_positions)
        na_indicator = nodes_visited.isna()
        flag_unique = ~nodes_visited.duplicated() & ~na_indicator
        return flag_unique.cumsum() / sess.tree.tree.number_of_nodes()

    @analyze_session
    def avg_node_time(self, sess: Session, mode=AVG_NODE_TIME_OVERALL, a: int = None, b: int = None,
                      min_nodes_on_path: int = 0):
        """
        Return a dictionary containing the overall node average time and for every node in the path between a and b
        :param sess:
        :param mode:
        :param a:
        :param b:
        :param min_nodes_on_path:
        :return:
        """
        def time_in_node_sec(nodes: pd.Series, time_in_sec=False):
            nodes = nodes.dropna()
            visits = {}
            for node in nodes:
                node_id = node if isinstance(node, int) else [item for item in node if type(item) == int][0]
                if node_id not in visits:
                    visits[node_id] = 0
                if time_in_sec:
                    visits[node_id] += sess.session_stream_info[FPS]
                else:
                    visits[node_id] += 1

            node_avg = np.mean(list(visits.values()))
            node_median = np.median(list(visits.values()))
            visits['node_avg'] = node_avg
            visits['node_median'] = node_median

            return visits

        df = sess.get_df(self.cfg.location_settings.bodypart)
        if mode == AVG_NODE_TIME_OVERALL:
            num_visits_per_node = time_in_node_sec(df[TREE_POSITION])
        elif mode == AVG_NODE_TIME_BY_PATH:
            condition.threshold = min_nodes_on_path
            a_to_b_indices = a_to_b(df, TILE_ID, a, b, condition)
            num_visits_per_node = []
            for ind_a, ind_b in a_to_b_indices:
                tree_frame_position = df[ind_a:ind_b + 1][TREE_POSITION]
                num_visits_per_node.append(time_in_node_sec(tree_frame_position))

        else:
            raise ValueError(f"mode {mode} is not supported")

        return num_visits_per_node

    @analyze_session
    def shortest_path_from_a_to_b(self, sess: Session, a: int, b: int, min_nodes_on_path: int = 0, levels = None,
                                  strikes=None):
        """
        Return the shortest path length from a to b that has been completed without straying from the path
        :param sess:
        :param a:
        :param b:
        :param min_nodes_on_path:
        levels: list of integers representing the tree levels to allow for mistakes
        strikes: number of strikes allowed on the levels
        """

        df = sess.get_df(self.cfg.location_settings.bodypart)
        condition.threshold = min_nodes_on_path
        a_to_b_indices = a_to_b(df, TILE_ID, a, b, condition)
        b_node = sess.tree.get_tree_location(b)
        reward_node = b_node if isinstance(b_node, int) else [item for item in b_node if type(item) == int][0]

        num_nodes_on_shortest_path = []
        for ind_a, ind_b in a_to_b_indices:
            tree_frame_position = df[ind_a:ind_b + 1][TREE_POSITION]
            ind_on_shortest_path = -1
            shortest_path_from_current_node_to_reward = None
            strike = 0
            previous_node = None

            for frame_num, cur_node in tree_frame_position.items():

                # Skip any type of nan value
                if cur_node is None or (isinstance(cur_node, float) and np.isnan(cur_node)):
                    continue

                # Skip edges
                if type(cur_node) == tuple:
                    continue

                # handle cases where the current node is both a node and an edge
                if isinstance(cur_node, frozenset):
                    cur_node = [item for item in cur_node if isinstance(item, int)][0]

                # On the first non-nan value update the shortest path
                if cur_node is not None and ind_on_shortest_path == -1:
                    shortest_path_from_current_node_to_reward = sess.tree.get_shortest_path(cur_node, reward_node)
                    ind_on_shortest_path = 0
                    continue

                # # Skip the current node if it is the same as the previous node
                # if cur_node == shortest_path_from_current_node_to_reward[ind_on_shortest_path]:
                #     continue

                # # Skip the current node if it is the same as the previous node
                if cur_node == previous_node:
                    continue
                # Update the previous node
                previous_node = cur_node

                # Allow for up to strikes mistakes on specific tree levels
                if levels is not None and strikes is not None:
                    if cur_node != shortest_path_from_current_node_to_reward[ind_on_shortest_path + 1]:
                        if int(str(cur_node)[0]) in levels:
                            if strike < strikes:
                                strike += 1
                                continue

                if cur_node == shortest_path_from_current_node_to_reward[ind_on_shortest_path + 1]:
                    ind_on_shortest_path += 1
                    strike = 0
                    #  Check if we reached the reward - if so, add the eureka moment and break
                    if ind_on_shortest_path == len(shortest_path_from_current_node_to_reward) - 1:
                        break
                    else:
                        continue
                else:
                    # Reset the shortest path
                    shortest_path_from_current_node_to_reward = sess.tree.get_shortest_path(cur_node, reward_node)
                    ind_on_shortest_path = 0

            # add the shortest path length to list
            num_nodes_on_shortest_path.append(len(shortest_path_from_current_node_to_reward))

        return num_nodes_on_shortest_path

    def run_analysis(self) -> pd.DataFrame:
        analysis_results = {}
        feature_dict = self.cfg.get(ANALYZE_CFG_KEY, {}).get(METRICS_KEY, {})
        for feature_name, func_dict in feature_dict.items():
            func_name = func_dict[FUNC_NAME]
            func_args = func_dict[FUNC_ARGS] if FUNC_ARGS in func_dict else {}
            analysis_results[feature_name] = getattr(self, func_name)(**func_args)

        df = pd.DataFrame(analysis_results)
        output_path = self.cfg.get(OUTPUT_PATH, None)

        # Save analysis
        if self.cfg.get(ANALYZE_CFG_KEY, {}).get(SAVE_CSV, False) and output_path is not None:
            df.to_csv(os.path.join(output_path, OUTPUT_FILE_NAME + '.csv'))

        if self.cfg.get(ANALYZE_CFG_KEY, {}).get(SAVE_PKL, False) and output_path is not None:
            df.to_pickle(os.path.join(output_path, OUTPUT_FILE_NAME + '.pkl'))

        if self.cfg.get(ANALYZE_CFG_KEY, {}).get(SAVE_RAW, False) and output_path is not None:
            for sess in self.session_list:
                raw_df = sess.get_df(self.cfg.location_settings.bodypart)
                raw_df.to_pickle(os.path.join(output_path, sess._session_name + '_raw.pkl'))

        return df


