from dataclasses import dataclass
from typing import Any, Optional, Dict


@dataclass
class AnalysisResult:
    """
    A standardized container for analysis results.

    Attributes:
        metric_name: The name of the computed metric.
        session_name: The name of the session (None for cross-session analysis).
        value: The computed metric value (scalar, list, dict, pd.Series, np.array, etc.).
        units: Optional string describing the units.
        metadata: Optional dictionary with extra information.
    """
    metric_name: str
    session_name: Optional[str]
    value: Any
    units: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
