"""
Cross-Session (Global) Analysis Functions

Each function processes an aggregated DataFrame of session-level metrics and returns a single AnalysisResult.
"""

import pandas as pd
from analysis_result import AnalysisResult
from cross_session_analysis_registry import register_cross_session_analysis

@register_cross_session_analysis("average_metric")
def average_metric(metrics_df: pd.DataFrame, **kwargs) -> AnalysisResult:
    """
    Compute the mean of each metric across sessions.

    Args:
        metrics_df (pd.DataFrame): DataFrame with sessions as rows and metric names as columns.

    Returns:
        AnalysisResult: Contains the average for each metric.
    """
    averages = metrics_df.mean()
    return AnalysisResult(
        metric_name="cross_session_average",
        session_name=None,
        value=averages,
        units="various"
    )

@register_cross_session_analysis("median_metric")
def median_metric(metrics_df: pd.DataFrame, **kwargs) -> AnalysisResult:
    """
    Compute the median of each metric across sessions.
    """
    medians = metrics_df.median()
    return AnalysisResult(
        metric_name="cross_session_median",
        session_name=None,
        value=medians,
        units="various"
    )
