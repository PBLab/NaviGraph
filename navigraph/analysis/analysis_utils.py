"""
Utilities for common analysis calculations.
These functions and helper classes are shared among different analysis metrics.
"""

from dataclasses import dataclass
from typing import Union, Any, Callable
from numpy.typing import NDArray
import pandas as pd
import numpy as np

# A simple helper class for specifying conditions.
@dataclass
class Condition:
    column_name: str
    func: Callable[[pd.Series], Union[int, float]]
    threshold: Union[str, float, int]
    operator: Callable[[Union[int, float], Union[int, float]], bool]

def count_node_visits_eliminating_sequences(ser: Union[pd.Series, NDArray]) -> int:
    """
    Count node visits in a Series by eliminating consecutive duplicates.
    Returns the total number of (distinct) visits.
    """
    def int_nodes_only(x):
        if isinstance(x, frozenset):
            # Pick the first integer from the frozenset.
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
    return int(np.sum(visits_per_node.values))

def count_type_specific_objects(ser: Union[pd.Series, NDArray], dtype: Any) -> int:
    """Return the number of elements of the given type in the series/array."""
    return int(np.sum([True if isinstance(obj, dtype) else False for obj in ser]))

def count_unique_type_specific_objects(ser: pd.Series, dtype: Any) -> int:
    """Return the number of unique objects of the specified type."""
    ser_unique = ser.unique()
    return count_type_specific_objects(ser_unique, dtype)

def a_to_b(df: pd.DataFrame,
           column_name: str,
           val_a: Union[int, float, str],
           val_b: Union[int, float, str],
           condition: Condition) -> list:
    """
    Find all occurrences in a DataFrame where value val_a is followed by val_b (with val_a preceding val_b).
    The 'condition' parameter provides additional filtering:
      - condition.func is applied to the DataFrame slice between the two events;
      - condition.operator compares the computed value with condition.threshold.
    Returns a list of index pairs [start, end].
    """
    a_b_pairs = []
    while len(df) > 0 and val_a in list(df[column_name]) and val_b in list(df[column_name]):
        first_a = df[column_name].eq(val_a).idxmax()
        first_b = df[column_name].eq(val_b).idxmax()
        if first_a < first_b:
            if condition is None:
                a_b_pairs.append([first_a, first_b])
            else:
                value_to_threshold = condition.func(df.loc[first_a:first_b][condition.column_name])
                if condition.operator(value_to_threshold, condition.threshold):
                    a_b_pairs.append([first_a, first_b])
        df = df[df.index > first_b]
    return a_b_pairs
