import pandas as pd
from omegaconf import DictConfig
from typing import List
import logging as lg
from session_analysis_registry import get_session_analysis
from cross_session_analysis_registry import get_cross_session_analysis
from analysis_result import AnalysisResult

logger = lg.getLogger(__name__)


class SessionAnalyzer:
    def __init__(self, cfg: DictConfig, session_list: List[Any]) -> None:
        self.cfg: DictConfig = cfg
        self.session_list: List[Any] = session_list

    def run_session_level_analysis(self) -> pd.DataFrame:
        """
        Run each registered session-level analysis function for every session.

        Configuration should have a dictionary under analyze.metrics where each key is a metric name,
        and each value is a dict with keys "func_name" and optionally "args".

        Returns:
            A DataFrame with sessions as rows and metric names as columns.
        """
        results = {}
        metrics_cfg = self.cfg.get("analyze", {}).get("metrics", {})
        for metric_name, metric_def in metrics_cfg.items():
            func_name = metric_def["func_name"]
            args = metric_def.get("args", {})
            analysis_func = get_session_analysis(func_name)
            if analysis_func is None:
                logger.error("Session analysis function '%s' not found", func_name)
                continue
            for session in self.session_list:
                result: AnalysisResult = analysis_func(session, **args)
                results.setdefault(session.session_name, {})[metric_name] = result.value
        return pd.DataFrame(results).transpose()

    def run_cross_session_analysis(self, metrics_df: pd.DataFrame) -> AnalysisResult:
        """
        Run a cross-session analysis function on the aggregated metrics DataFrame.

        The configuration may specify a global (now cross-session) function under analyze.global_func.
        If not provided, this method returns an AnalysisResult that wraps the input DataFrame.

        Returns:
            An AnalysisResult representing the cross-session analysis.
        """
        global_cfg = self.cfg.get("analyze", {}).get("global_func", None)
        if global_cfg is None:
            return AnalysisResult(metric_name="cross_session", session_name=None, value=metrics_df)
        func_name = global_cfg["func_name"]
        args = global_cfg.get("args", {})
        analysis_func = get_cross_session_analysis(func_name)
        if analysis_func is None:
            logger.error("Cross-session analysis function '%s' not found", func_name)
            return AnalysisResult(metric_name="cross_session", session_name=None, value=metrics_df)
        return analysis_func(metrics_df, **args)
