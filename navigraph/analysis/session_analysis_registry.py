from typing import Callable, Dict, Any
from analysis_result import AnalysisResult

# Global registry dictionary for session-level analysis functions.
SESSION_ANALYSIS_REGISTRY: Dict[str, Callable[..., AnalysisResult]] = {}

def register_session_analysis(name: str) -> Callable:
    """
    Decorator to register a session-level analysis function under a given name.
    """
    def decorator(func: Callable[..., AnalysisResult]) -> Callable[..., AnalysisResult]:
        SESSION_ANALYSIS_REGISTRY[name] = func
        return func
    return decorator

def get_session_analysis(name: str) -> Callable[..., AnalysisResult]:
    """
    Retrieve a registered session-level analysis function by name.
    """
    return SESSION_ANALYSIS_REGISTRY.get(name)
