# NaviGraph 🗺️: Navigation on the Graph

**A graph-based Python pipeline for analyzing spatial decision-making in maze-like environments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges here later: PyPI version, build status, code coverage -->

---

NaviGraph addresses the challenge of analyzing complex, multi-modal datasets from neuroscience experiments, particularly those involving spatial navigation in mazes. Classic methods often analyze behavioral tracking and neuronal activity streams independently, potentially missing the intricate dynamics of decision-making, especially subtle changes in disease models.

**NaviGraph integrates diverse data streams (behavior, neural activity, physiology) into a unified graph-based framework, offering a holistic and sensitive perspective on cognitive processes.**

![NaviGraph Pipeline](https://raw.githubusercontent.com/your-username/navigraph/main/docs/images/pipeline_figure.png)  
*Fig. 1: NaviGraph pipeline workflow (Conceptual — **Replace with your actual Figure 1 image path**). Shows stages from raw data input, through preprocessing (pose estimation, calcium analysis), registration (calibration, validation), analysis (classic & topological metrics), to aggregation.*

---

## ✨ Key Features

- **Graph-Based Framework:** Models maze decision points as **nodes** and paths as **edges**, providing a powerful topological representation.
- **Multi-Modal Data Integration:** Seamlessly combines behavioral data (pose estimation via DeepLabCut), neuronal activity (e.g., calcium imaging via Minian), and physiological data (e.g., head orientation) onto the graph structure.
- **Synergistic Analysis:** Computes both **classic metrics** (time to reward, velocity) and novel **topological metrics** (cumulative path length on graph, wandering index, direct path length) for deeper insights.
- **Modular & Flexible:** Easily adaptable pipeline structure allows integration of various data streams and customization for different maze designs and experimental setups.
- **Configuration Driven:** Uses YAML files (managed via Hydra) for easy setup of paths, parameters, and system modes (calibrate, analyze, visualize, test).
- **Extensible:** Built with registries for analysis functions and visualizers, allowing users to easily add custom metrics and plots.
- **Visualization:** Offers powerful visualization capabilities, including mapping behavior, neural activity, and physiological data directly onto the maze graph structure.

---

## 🤔 Why NaviGraph?

- **Uncover Subtle Phenotypes:** Topological metrics can reveal behavioral and cognitive differences missed by classic measures alone.
- **Holistic Understanding:** Analyze interactions between behavior, neural activity, and physiology within the same spatial context.
- **Standardized & Reproducible:** Provides a structured framework for processing and analyzing complex multi-stream data.
- **Intuitive Representation:** Visualizing data directly on the graph provides clear insights into navigation strategies and decision-making processes.

---

## 💡 Core Concepts

At its heart, NaviGraph treats the maze not just as physical space, but as a **topological graph**:

- **Nodes:** Represent key locations, typically decision points in the maze.
- **Edges:** Represent the paths or transitions between these nodes.
- **Data Layers:** Behavioral metrics, neural activity traces, head direction angles, etc., are associated with specific nodes or edges.

---

## ⚙️ Pipeline Workflow

NaviGraph processes data through several key stages, orchestrated by the `SessionManager`:

1. **Input Data:** Raw behavioral video, neuronal recordings (e.g., miniscope data), optional physiological data.
2. **Preprocessing:** Independent processing of raw streams (e.g., pose estimation with DeepLabCut, calcium trace extraction with Minian).
3. **Registration (Calibration & Validation):**
   - Aligns behavioral data (pixel coordinates) to the maze map.
   - Validates the registration accuracy.
   - Maps pixel coordinates to graph positions (nodes/edges).
4. **Data Coupling:** Synchronizes all data streams on a frame-by-frame basis and stores the integrated data per session (e.g., as a `.pkl`).
5. **Analysis:** Applies registered analysis functions to compute classic and topological metrics.
6. **Visualization & Aggregation:** Generates visualizations and aggregates results across sessions for comparison.

---

## 🚀 Installation

NaviGraph uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/your-username/navigraph.git
cd navigraph

# Install dependencies
poetry install
```

---

## ⚡ Quick Start

1. **Prepare your data:** Pose tracking `.h5` and optional calcium imaging `zarr`.
2. **Configure your run:** Use an example YAML config and modify as needed.
3. **Run the pipeline:**

```bash
poetry shell
python navigraph/run.py --config-path path/to/configs --config-name your_config
```

---

## 🔧 Configuration Overview

YAML configuration via Hydra includes:

- `system_running_mode`: `calibrate`, `test`, `analyze`, `visualize`
- Input paths: `stream_path`, `keypoint_detection_file_path`, etc.
- `map_path`, `experiment_output_path`
- `map_settings`, `calibrator_parameters`, `datasources`, `analyze`

See `examples/configs/base_config.yaml`.

---

## 📁 Project Structure

```
.
├── examples/              
├── LICENSE
├── navigraph/             
│   ├── analysis/          
│   ├── configs/           
│   ├── modules/           
│   ├── session/           
│   ├── utils/             
│   ├── run.py             
│   └── session_manager.py 
├── poetry.lock
├── pyproject.toml         
├── README.md              
├── scripts/               
└── tests/                 
```

---

## 📄 Citation

> Iton, A. K., Iton, E., Michaelson, D. M., & Blinder, P. (Year). NaviGraph - a graph-based approach for analyzing spatial decision making in maze-like environments. *Journal Name*, Volume(Issue), Pages. [DOI]

*(Update when publication is available)*

---

## 🙌 Contributing

Pull requests and issue reports are welcome!

---

## 📜 License

Licensed under the MIT License. See [LICENSE](LICENSE).