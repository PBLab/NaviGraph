<p align="left">
  <img src="/docs/images/NaviGraph_logo_white_noback.png" alt="NaviGraph Logo" width="400"/>
</p>

## A framework for analyzing spatial decision-making by integrating multimodal data into a unified topological representation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add additional badges: PyPI, build status, coverage -->

---

<p align="left">
  <img src="/docs/images/software_pipeline.png" alt="NaviGraph pipeline"/> 
</p>

---

## 🚀 Installation
**Clone the repository**

`git clone https://github.com/PBLab/NaviGraph.git`

**Open a terminal and navigate to the project directory**

`cd /path/to/your/project`

**Install dependencies**

`pip install -r requirements.txt`

---

## ⚡ Quick Start

1. **Prepare your data:** Pose tracking (`.h5`) via [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut), calcium imaging (`zarr`, optional) via [MiniAn](https://github.com/denisecailab/minian).
2. **Configure the pipeline:** Customize example (`.yaml`) config file (details in next section).
3. **Run the pipeline**  Execute the main script:
   `python run.py`

---

## 🔧 Configuration Overview

Pipeline behavior is controlled via a YAML config file located at `configs/navigraph_config.yaml`.  

### Key Sections

**General Paths**
- `project_local_data_path`: Root directory for all input data (pose, video, etc.)
- `keypoint_detection_file_path`: Directory containing DeepLabCut `.h5` files
- `map_path`: Maze image for spatial alignment
- `output_directory_path`: Directory where outputs (calibration, analysis, plots, etc.) are saved

**Running Mode**
- `system_running_mode`: Defines the pipeline mode; options include `calibrate`, `test`, `visualize`, or `analyze`

**Calibration**
- `calibrator_parameters`: Settings for point capture and spatial alignment 

**Map Settings**  
- `map_settings`: Setup parameters for maze segmentation and scaling (segment length, origin, grid size, pixel-to-meter ratio)  
  *This section must be calculated for every new experiment.*

**Location Settings**  
- `location_settings`: Specifies body part for tracking (e.g., `Nose`) and likelihood threshold for filtering  

**Analysis Metrics**
- `analyze.metrics`: Specifies behavioral and topological metrics to compute 

**Graph Display**  
- `graph`: Visualization options for the maze graph, including node colors, sizes, and layout  

**Visualization Settings**  
- `visualization`: Controls frame overlays, map drawing options, video export, and visualization behavior  

---

To get started, copy `configs/navigraph_config.yaml`, update relevant paths, select the desired `system_running_mode`, and run the pipeline with:

```bash
python run.py
```

---

## 📄 Citation

If you use NaviGraph, please cite:

> Koren Iton, A., Iton, E., Michaelson, D. M., & Blinder, P. (2025). NaviGraph: A graph-based framework for multimodal analysis of spatial decision-making. bioRxiv

---

## 🙌 Contributing

Contributions, issue reports, and feature requests are welcome! Please open issues or submit pull requests.

---

## 📜 License

Licensed under the MIT License. See [LICENSE](LICENSE) for more details.
