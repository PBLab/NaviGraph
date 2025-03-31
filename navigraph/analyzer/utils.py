from dataclasses import dataclass
from typing import Union, Any, Callable
from numpy.typing import NDArray
import pandas as pd
import numpy as np

CONDITIONS_DICT_COL_NAME_KEY = 'column_name'
CONDITIONS_DICT_FUNC_KEY = 'func'
CONDITIONS_DICT_THRESHOLD_KEY = 'threshold'
CONDITIONS_DICT_OPERATOR_KEY = 'operator'


@dataclass
class Condition:
    column_name: str
    func: Callable
    threshold: Union[str, float, int]
    operator: Callable


def count_node_visits_eliminating_sequences(ser: Union[pd.Series, NDArray]):
    def int_nodes_only(x):
        if isinstance(x, frozenset):
            return [v for v in x if isinstance(v, int)][0]
        elif isinstance(x, int):
            return x
        else:
            return None

    ser = ser.apply(lambda x: int_nodes_only(x)).dropna()
    ser_shifted = ser.shift(1)
    df_temp = pd.DataFrame(ser)
    df_temp['seq'] = ser != ser_shifted

    visits_per_node = df_temp.groupby(ser.name).sum()

    return np.sum(visits_per_node.values)


def count_type_specific_objects(ser: Union[pd.Series, NDArray], dtype: Any):
    return np.sum([True if isinstance(obj, dtype) else False for obj in ser])


def count_unique_type_specific_objects(ser: pd.Series, dtype: Any):
    ser_unique = ser.unique()
    return count_type_specific_objects(ser_unique, dtype)


def a_to_b(df: pd.DataFrame,
           column_name: str,
           val_a: Union[int, float, str],
           val_b: Union[int, float, str],
           condition: Condition):
    # TODO: support both a before b and b before a
    """
    Find all appearances of certain values in a pandas Dataframe such that meet specific conditions described in
    conditions. conditions dictionary structure must be as follows:
    conditions = {column_name: str, func: callable, threshold: Union[str, int, float], operator: callable}
    """
    a_b_pairs = []
    while len(df) > 0 and val_a in list(df[column_name]) and val_b in list(df[column_name]):
        # First arrival to a and b
        first_a = df[column_name].eq(val_a).idxmax()
        first_b = df[column_name].eq(val_b).idxmax()

        if first_a < first_b:
            if condition is None:
                a_b_pairs.append([first_a, first_b])
            else:
                # Filter cases that do not meet required conditions
                value_to_threshold = condition.func(df.loc[first_a:first_b][condition.column_name])

                if condition.operator(value_to_threshold, condition.threshold):
                    a_b_pairs.append([first_a, first_b])

        # cut dataframe to find next interval
        df = df[df.index > first_b]

    return a_b_pairs
