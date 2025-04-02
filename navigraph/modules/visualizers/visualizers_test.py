import cv2
import numpy as np
import pandas as pd
import networkx as nx
from pose_visualizer import PoseVisualizer
from map_visualizer import MapVisualizer
from graph_visualizer import GraphVisualizer
from composite_frame_visualizer import CompositeFrameVisualizer
from visualizer_registry import VisualizerRegistry

# Register individual visualizers.
VisualizerRegistry.register_visualizer("pose", PoseVisualizer(keypoint_color=(0,255,0), keypoint_radius=5))
VisualizerRegistry.register_visualizer("map", MapVisualizer(tile_color=(255,0,0)))
VisualizerRegistry.register_visualizer("graph", GraphVisualizer(node_color="blue", edge_color="gray", node_size=300))

# Define a composite visualizer that will use the "custom" mode.
composite_vis = CompositeFrameVisualizer(
    order=["pose", "map", "graph"],
    mode="custom",
    opacity=0.6,
    custom_params={"resize_factor": 0.3, "opacity": 0.6, "frame_location": "bottom_right", "method": "on_top"}
)

# Simulate a base video frame.
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Simulate a DataFrame row with overlay data.
data_row = pd.Series({
    "keypoints": [(100,150), (200,250), (300,350)],
    "tile_bbox": [50,50,100,100],
    "tile_id": 5,
    "graph": nx.complete_graph(5)
})

# Use the composite visualizer.
final_frame = composite_vis.visualize(frame, data_row)

cv2.imshow("Final Visualization", final_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
