from typing import Callable, Dict, Any
from analysis_result import AnalysisResult
import pandas as pd

CROSS_SESSION_ANALYSIS_REGISTRY: Dict[str, Callable[..., AnalysisResult]] = {}

def register_cross_session_analysis(name: str) -> Callable:
    """
    Decorator to register a cross-session analysis function under a given name.
    """
    def decorator(func: Callable[..., AnalysisResult]) -> Callable[..., AnalysisResult]:
        CROSS_SESSION_ANALYSIS_REGISTRY[name] = func
        return func
    return decorator

def get_cross_session_analysis(name: str) -> Callable[..., AnalysisResult]:
    """
    Retrieve a registered cross-session analysis function by name.
    """
    return CROSS_SESSION_ANALYSIS_REGISTRY.get(name)
