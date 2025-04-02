"""
Session-Level Analysis Functions

Each function processes a single Session's data and returns an AnalysisResult.
All functions are registered using the session analysis registry.
"""

import operator
from functools import partial
import numpy as np
import pandas as pd

from session_analysis_registry import register_session_analysis
from analysis_result import AnalysisResult
from navigraph.session import Session, FPS, TILE_ID, TREE_POSITION

from analysis_utils import a_to_b, count_unique_type_specific_objects, count_node_visits_eliminating_sequences, Condition

# Constants used by analysis functions.
DEFAULT_MIN_NODES_ON_PATH = 0
NODE_PATH_TOPOLOGICAL_MODE = 'topological_distance'
NODE_PATH_EXPLORATION_MODE = 'exploration'
AVG_NODE_TIME_BY_PATH = 'by_path'
AVG_NODE_TIME_OVERALL = 'overall'
# (Other constants like TIME_FROM_SESSION_START etc. can be added if needed.)

NODE_POSSIBLE_DTYPES = (int, frozenset)
count_unique_nodes = partial(count_unique_type_specific_objects, dtype=NODE_POSSIBLE_DTYPES)
condition = Condition(column_name=TREE_POSITION,
                      func=count_unique_nodes,
                      threshold=DEFAULT_MIN_NODES_ON_PATH,
                      operator=operator.gt)


@register_session_analysis("time_a_to_b")
def time_a_to_b(session: Session, a: int, b: int, min_nodes_on_path: int = 0, insert_trial_number: bool = True) -> AnalysisResult:
    """
    Compute time intervals (in seconds) between event 'a' and event 'b' for a given session.
    If insert_trial_number is True, the session's DataFrame is augmented with a 'trial' column.
    """
    df = session.get_df(session.cfg.location_settings.bodypart)
    condition.threshold = min_nodes_on_path
    indices = a_to_b(df, TILE_ID, a, b, condition)

    # Optionally insert trial numbers.
    if insert_trial_number:
        col_name = 'trial'
        df[col_name] = np.nan
        for i, (start, end) in enumerate(indices, start=1):
            df.loc[start:end, col_name] = i
        session.insert_data(session.cfg.location_settings.bodypart, col_name, df[col_name])

    fps = session.session_stream_info[FPS]
    intervals = [(end + 1 - start) * (1 / fps) for start, end in indices]

    return AnalysisResult(
        metric_name="time_a_to_b",
        session_name=session.session_name,
        value=intervals,
        units="seconds"
    )


@register_session_analysis("velocity_a_to_b")
def velocity_a_to_b(session: Session, a: int, b: int, min_nodes_on_path: int = 0) -> AnalysisResult:
    """
    Compute average velocity (in m/s) for trajectories between event a and event b.
    """
    df = session.get_df(session.cfg.location_settings.bodypart)
    condition.threshold = min_nodes_on_path
    indices = a_to_b(df, TILE_ID, a, b, condition)
    fps = session.session_stream_info[FPS]
    times = [(end + 1 - start) * (1 / fps) for start, end in indices]

    velocities = []
    for start, end in indices:
        # Calculate the distance traveled between indices.
        segment = df.iloc[start:end+1]
        # Using Euclidean distance between successive (x, y) coordinates.
        path_length_pixels = np.sqrt((segment.x.diff() ** 2) + (segment.y.diff() ** 2)).sum()
        # Convert pixels to meters.
        distance_meters = path_length_pixels / session.map_labeler.pixel_to_meter
        if times and times[indices.index([start, end])] > 0:
            velocities.append(distance_meters / times[indices.index([start, end])])
        else:
            velocities.append(np.nan)
    return AnalysisResult(
        metric_name="velocity_a_to_b",
        session_name=session.session_name,
        value=velocities,
        units="m/s"
    )


@register_session_analysis("num_nodes_in_path")
def num_nodes_in_path(session: Session, a: int, b: int, min_nodes_on_path: int = 0,
                      mode: str = NODE_PATH_TOPOLOGICAL_MODE, min_frame_rep: int = 1) -> AnalysisResult:
    """
    Compute the number of nodes in the path between events a and b.
    If mode is 'topological_distance', uses a method that eliminates consecutive duplicates.
    Otherwise, computes a normalized count based on unique nodes.
    """
    df = session.get_df(session.cfg.location_settings.bodypart)
    condition.threshold = min_nodes_on_path
    indices = a_to_b(df, TILE_ID, a, b, condition)
    node_counts = []
    for start, end in indices:
        if mode == NODE_PATH_TOPOLOGICAL_MODE:
            tree_positions = df.iloc[start:end+1][TREE_POSITION]
            node_counts.append(count_node_visits_eliminating_sequences(tree_positions))
        else:
            unique_count = count_unique_type_specific_objects(df.iloc[start:end+1][TREE_POSITION], NODE_POSSIBLE_DTYPES)
            total_nodes = session.tree.tree.number_of_nodes()
            node_counts.append(unique_count / total_nodes)
    return AnalysisResult(
        metric_name="num_nodes_in_path",
        session_name=session.session_name,
        value=node_counts
    )


@register_session_analysis("exploration_percentage")
def exploration_percentage(session: Session) -> AnalysisResult:
    """
    Calculate the cumulative percentage of unique nodes visited over time.
    """
    df = session.get_df(session.cfg.location_settings.bodypart)

    def keep_node(x):
        if isinstance(x, int):
            return x
        elif isinstance(x, frozenset):
            return [i for i in x if isinstance(i, int)][0]
        else:
            return None

    nodes_visited = df[TREE_POSITION].apply(keep_node)
    na_indicator = nodes_visited.isna()
    unique_flags = (~nodes_visited.duplicated()) & (~na_indicator)
    cumulative = unique_flags.cumsum()
    total_nodes = session.tree.tree.number_of_nodes()
    exploration_pct = cumulative / total_nodes
    return AnalysisResult(
        metric_name="exploration_percentage",
        session_name=session.session_name,
        value=exploration_pct.tolist() if hasattr(exploration_pct, "tolist") else exploration_pct,
        units="percentage"
    )


@register_session_analysis("avg_node_time")
def avg_node_time(session: Session, mode: str = AVG_NODE_TIME_OVERALL, a: int = None, b: int = None,
                  min_nodes_on_path: int = 0) -> AnalysisResult:
    """
    Compute the average time (or frame count) spent per node.
    In mode 'overall' this is computed for the entire session;
    in mode 'by_path', it is computed for each path between events a and b.
    """
    def time_in_node_sec(nodes: pd.Series, time_in_sec: bool = False) -> dict:
        nodes = nodes.dropna()
        visits = {}
        for node in nodes:
            # Extract a single integer from node (or frozenset)
            if isinstance(node, int):
                node_id = node
            elif isinstance(node, frozenset):
                node_id = [item for item in node if isinstance(item, int)][0]
            else:
                continue
            visits[node_id] = visits.get(node_id, 0) + (session.session_stream_info[FPS] if time_in_sec else 1)
        if visits:
            node_avg = np.mean(list(visits.values()))
            node_median = np.median(list(visits.values()))
            visits['node_avg'] = node_avg
            visits['node_median'] = node_median
        return visits

    df = session.get_df(session.cfg.location_settings.bodypart)
    if mode == AVG_NODE_TIME_OVERALL:
        result = time_in_node_sec(df[TREE_POSITION])
    elif mode == "by_path":
        condition.threshold = min_nodes_on_path
        indices = a_to_b(df, TILE_ID, a, b, condition)
        results = []
        for start, end in indices:
            segment = df.iloc[start:end+1][TREE_POSITION]
            results.append(time_in_node_sec(segment))
        result = results
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return AnalysisResult(
        metric_name="avg_node_time",
        session_name=session.session_name,
        value=result
    )


@register_session_analysis("shortest_path_from_a_to_b")
def shortest_path_from_a_to_b(session: Session, a: int, b: int, min_nodes_on_path: int = 0,
                              levels: list = None, strikes: int = None) -> AnalysisResult:
    """
    Compute the length of the shortest valid path from event a to event b that meets certain criteria.
    This function attempts to filter out deviations by resetting the path if too many errors occur.
    """
    df = session.get_df(session.cfg.location_settings.bodypart)
    condition.threshold = min_nodes_on_path
    indices = a_to_b(df, TILE_ID, a, b, condition)
    b_node = session.tree.get_tree_location(b)
    # Determine the reward node id from b_node.
    if isinstance(b_node, int):
        reward_node = b_node
    else:
        reward_node = [item for item in b_node if isinstance(item, int)][0]

    path_lengths = []
    for start, end in indices:
        tree_positions = df.iloc[start:end+1][TREE_POSITION]
        ind_on_path = -1
        shortest_path = None
        strike = 0
        previous_node = None

        for idx, cur_node in tree_positions.items():
            if cur_node is None or (isinstance(cur_node, float) and np.isnan(cur_node)):
                continue
            if isinstance(cur_node, tuple):
                continue
            if isinstance(cur_node, frozenset):
                cur_node = [item for item in cur_node if isinstance(item, int)][0]
            if cur_node is not None and ind_on_path == -1:
                shortest_path = session.tree.get_shortest_path(cur_node, reward_node)
                ind_on_path = 0
                continue
            if cur_node == previous_node:
                continue
            previous_node = cur_node
            if levels is not None and strikes is not None:
                # Allow a limited number of errors for specified levels.
                if cur_node != shortest_path[ind_on_path + 1]:
                    if int(str(cur_node)[0]) in levels:
                        if strike < strikes:
                            strike += 1
                            continue
            if cur_node == shortest_path[ind_on_path + 1]:
                ind_on_path += 1
                strike = 0
                if ind_on_path == len(shortest_path) - 1:
                    break
                else:
                    continue
            else:
                shortest_path = session.tree.get_shortest_path(cur_node, reward_node)
                ind_on_path = 0
        if shortest_path is not None:
            path_lengths.append(len(shortest_path))
        else:
            path_lengths.append(None)
    return AnalysisResult(
        metric_name="shortest_path_from_a_to_b",
        session_name=session.session_name,
        value=path_lengths
    )
