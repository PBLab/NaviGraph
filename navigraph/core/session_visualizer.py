"""Session Visualizer orchestrator for NaviGraph.

Manages visualization pipeline for a session.
Orchestrates the looping over video frames, passes each frame through
registered visualizer functions in sequence, and manages output creation.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
import signal
import sys

from .registry import registry
from .exceptions import NavigraphError


class SessionVisualizer:
    """Orchestrates visualization pipeline for a session."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with visualization configuration.
        
        Args:
            config: Dict with structure:
                {
                    'visualizations': [
                        {
                            'name': 'bodypart_display',
                            'type': 'bodyparts',
                            'config': {'bodyparts': ['Nose'], 'radius': 5}
                        },
                        {
                            'name': 'trajectory_trail',
                            'type': 'trajectory',
                            'config': {'trail_length': 50}
                        }
                    ],
                    'output': {
                        'enabled': True,
                        'path': './output/videos',
                        'format': 'mp4',
                        'fps': 30,
                        'codec': 'mp4v'
                    }
                }
        """
        viz_config = config.get('visualizations', {})
        
        # Handle both list format (pipeline) and dict format with output + pipeline
        if isinstance(viz_config, list):
            self.visualizer_specs = viz_config
            self.output_config = {}
        else:
            self.visualizer_specs = viz_config.get('pipeline', [])
            self.output_config = viz_config.get('output', {})
            
        self.logger = logger

        # Initialize video writer references for signal handling
        self._video_writer = None
        self._video_cap = None
        self._video_writer_params = None
        self._setup_signal_handlers()
    
    def process_video(
        self, 
        video_path: str,
        dataframe: pd.DataFrame, 
        shared_resources: Dict[str, Any],
        output_name: str = "output",
        output_dir: Optional[str] = None,
        show_realtime: bool = False
    ) -> Optional[str]:
        """Process video through visualization pipeline.
        
        Args:
            video_path: Path to input video
            dataframe: Session dataframe with frame-aligned data
            shared_resources: Shared resources (graph, mapping, etc.)
            output_name: Name for output file (without extension)
            output_dir: Directory for output video
            show_realtime: Whether to display frames in real-time during processing
            
        Returns:
            Path to output video if created, None otherwise
        """
        # Check if output is enabled
        if not self.output_config.get('enabled', True):
            self.logger.info("Video output disabled in config")
            return None
        
        # Check if we have visualizer specs
        if not self.visualizer_specs:
            self.logger.warning("No visualizers configured")
            return None
        
        # Validate and load visualizer functions
        visualizers = []
        for spec in self.visualizer_specs:
            viz_name = spec.get('name', 'unnamed')
            viz_type = spec.get('type')
            viz_config = spec.get('config', {})
            
            if not viz_type:
                self.logger.error(f"Visualizer '{viz_name}' missing type")
                continue
                
            try:
                viz_func = registry.get_visualizer(viz_type)
                visualizers.append((viz_name, viz_func, viz_config))
                self.logger.info(f"Loaded visualizer: {viz_name} ({viz_type})")
            except NavigraphError as e:
                self.logger.error(f"Visualizer type '{viz_type}' not found: {e}")
                if spec.get('required', False):
                    raise
        
        if not visualizers:
            self.logger.error("No valid visualizers loaded")
            return None
        
        # Open video
        self._video_cap = cv2.VideoCapture(video_path)
        if not self._video_cap.isOpened():
            raise NavigraphError(f"Failed to open video: {video_path}")

        try:
            # Get video properties
            fps = self.output_config.get('fps', self._video_cap.get(cv2.CAP_PROP_FPS))
            original_width = int(self._video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self._video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self._video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Setup output path (lazy video writer initialization)
            if output_dir:
                output_path = Path(output_dir)
            else:
                output_path = Path(self.output_config.get('path', './output'))
            output_path.mkdir(parents=True, exist_ok=True)

            output_format = self.output_config.get('format', 'mp4')
            output_file = output_path / f"{output_name}.{output_format}"

            # Store parameters for lazy video writer initialization
            codec_str = self.output_config.get('codec', 'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            self._video_writer_params = {
                'output_file': str(output_file),
                'fourcc': fourcc,
                'fps': fps,
                'codec_str': codec_str
            }

            self.logger.info(f"Processing {total_frames} frames through {len(visualizers)} visualizers")
            self.logger.info(f"Output will be saved to: {output_file}")
            self.logger.info(f"Video writer will be lazily initialized after first frame")
            
            if show_realtime:
                self.logger.info("🎬 Real-time visualization enabled - press 'q' to quit, space or 'p' to pause/resume")
                window_name = f"NaviGraph - {output_name}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, original_width//2, original_height//2)  # Display at half size for performance
                paused = False
            
            # Process frames
            frame_idx = 0
            last_progress = 0
            
            while self._video_cap.isOpened():
                ret, frame = self._video_cap.read()
                if not ret or frame is None:
                    break
                
                # Get frame data from dataframe
                if frame_idx < len(dataframe):
                    frame_data = dataframe.iloc[frame_idx]
                else:
                    # No data for this frame
                    frame_data = pd.Series()
                
                # Apply visualizer pipeline (each modifies the frame)
                for viz_name, viz_func, viz_config in visualizers:
                    try:
                        frame = viz_func(
                            frame=frame,
                            frame_data=frame_data,
                            shared_resources=shared_resources,
                            **viz_config
                        )
                        
                        # Validate frame is still valid
                        if frame is None or not isinstance(frame, np.ndarray):
                            raise TypeError(f"Visualizer '{viz_name}' returned invalid frame")
                            
                    except Exception as e:
                        self.logger.error(f"Visualizer '{viz_name}' failed on frame {frame_idx}: {e}")
                        if viz_config.get('required', False):
                            raise
                
                # Show real-time display if enabled
                if show_realtime:
                    # Handle pause/resume and quit controls
                    while paused:
                        key = cv2.waitKey(30) & 0xFF
                        if key == ord(' ') or key == ord('p') or key == ord('P'):  # Space or P to resume
                            paused = False
                            self.logger.info("Resumed")
                        elif key == ord('q') or key == ord('Q'):  # Quit
                            self.logger.info("User requested quit")
                            break
                    
                    if not paused:
                        # Display the frame
                        cv2.imshow(window_name, frame)
                        
                        # Handle key presses (non-blocking)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q'):  # Quit
                            self.logger.info("User requested quit - finalizing video")
                            break
                        elif key == ord(' ') or key == ord('p') or key == ord('P'):  # Pause
                            paused = True
                            self.logger.info("Paused - press space or 'p' to resume")
                
                # Check if user quit during pause
                if show_realtime and paused and 'key' in locals() and (key == ord('q') or key == ord('Q')):
                    break
                
                # Lazy video writer initialization on first frame
                if self._video_writer is None:
                    frame_height, frame_width = frame.shape[:2]
                    self._video_writer = cv2.VideoWriter(
                        self._video_writer_params['output_file'],
                        self._video_writer_params['fourcc'],
                        self._video_writer_params['fps'],
                        (frame_width, frame_height)
                    )
                    if not self._video_writer.isOpened():
                        raise NavigraphError(f"Failed to create lazy video writer for: {self._video_writer_params['output_file']}")

                    self.logger.info(f"✓ Video writer initialized: {frame_width}x{frame_height} @ {self._video_writer_params['fps']}fps")

                # Write processed frame
                self._video_writer.write(frame)
                
                # Progress logging (every 10%)
                progress = int((frame_idx / total_frames) * 100)
                if progress >= last_progress + 10:
                    self.logger.debug(f"Progress: {progress}%")
                    last_progress = progress
                
                frame_idx += 1
            
            self.logger.info(f"✓ Processed {frame_idx} frames")
            
        finally:
            # Cleanup video resources
            self._cleanup_video_resources()
            cv2.destroyAllWindows()
        
        self.logger.info(f"✓ Video saved to: {output_file}")
        return str(output_file)
    
    def find_video(self, session_path: Path) -> Optional[Path]:
        """Find video file in session directory.
        
        Args:
            session_path: Path to session directory
            
        Returns:
            Path to video file or None if not found
        """
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        
        for ext in video_extensions:
            videos = list(session_path.glob(f'*{ext}'))
            if videos:
                # Return first video found
                return videos[0]
        
        # Check subdirectories
        for ext in video_extensions:
            videos = list(session_path.rglob(f'*{ext}'))
            if videos:
                return videos[0]
        
        return None
    
    def validate_pipeline(self) -> List[str]:
        """Validate that all visualizers in specs are registered.
        
        Returns:
            List of missing visualizer type names
        """
        missing = []
        for spec in self.visualizer_specs:
            viz_type = spec.get('type')
            if viz_type:
                try:
                    registry.get_visualizer(viz_type)
                except NavigraphError:
                    missing.append(viz_type)
        
        return missing


    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful video saving on interruption."""
        def signal_handler(signum, frame):
            if self._video_writer is not None:
                self.logger.warning(f"Received signal {signum}, saving video before exit...")
                self._cleanup_video_resources()
            else:
                self.logger.warning(f"Received signal {signum}, no video to save (writer not initialized)")
            cv2.destroyAllWindows()
            sys.exit(0)

        # Handle Ctrl+C (SIGINT) and termination (SIGTERM)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _cleanup_video_resources(self):
        """Clean up video capture and writer resources."""
        try:
            if self._video_cap is not None:
                self._video_cap.release()
                self._video_cap = None
                self.logger.debug("Video capture released")
        except Exception as e:
            self.logger.warning(f"Error releasing video capture: {e}")

        try:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None
                self.logger.info("✓ Video writer saved and released")
        except Exception as e:
            self.logger.warning(f"Error releasing video writer: {e}")