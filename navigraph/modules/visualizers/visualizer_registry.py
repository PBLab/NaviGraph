from typing import Dict
from base_frame_visualizer import BaseFrameVisualizer


class VisualizerRegistry:
    """
    Registry for frame visualizers.

    This allows new visualizers to be registered and retrieved by name.
    """
    _registry: Dict[str, BaseFrameVisualizer] = {}

    @classmethod
    def register_visualizer(cls, name: str, visualizer: BaseFrameVisualizer) -> None:
        cls._registry[name] = visualizer

    @classmethod
    def get_visualizer(cls, name: str) -> BaseFrameVisualizer:
        return cls._registry.get(name)

    @classmethod
    def list_visualizers(cls) -> Dict[str, BaseFrameVisualizer]:
        return cls._registry
