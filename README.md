<p align="center">
  <img src="/docs/images/navigraph_logo.png" alt="NaviGraph Logo" width="400"/>
</p>

<br>

# NaviGraph 🗺️: Navigation on the Graph

**A graph-based Python pipeline for comprehensive analysis of spatial decision-making in maze-like environments.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add additional badges: PyPI, build status, coverage -->

---

**NaviGraph** is designed to address challenges in analyzing complex, multimodal datasets from neuroscience experiments involving spatial navigation. Unlike classical approaches, NaviGraph integrates diverse data streams—behavior, neural activity, and physiological measurements—within a unified graph-based framework, revealing intricate cognitive dynamics that traditional methods may overlook.

![NaviGraph Pipeline](https://raw.githubusercontent.com/your-username/navigraph/main/docs/images/pipeline_figure.png)  
*Fig. 1: NaviGraph pipeline overview. (Replace with actual image path)*

---

## ✨ Key Features

- **Graph-Based Modeling:** Represents maze decision points as nodes and navigational paths as edges, enabling robust topological analysis.
- **Multi-Modal Integration:** Combines behavioral (DeepLabCut), neuronal (Minian calcium imaging), and physiological (head orientation) data seamlessly.
- **Comprehensive Metrics:** Includes both classic (e.g., reward latency, velocity) and advanced topological metrics (path efficiency, wandering indices).
- **Modular & Flexible:** Easily adaptable pipeline supports diverse maze configurations and experimental setups.
- **Configuration-Driven:** Uses YAML configuration files via Hydra for effortless management of pipeline parameters and system modes.
- **Extensible Framework:** Built-in registries facilitate quick addition of custom analysis functions and visualization modules.
- **Advanced Visualization:** Powerful tools to visualize and interpret behavioral, neuronal, and physiological data directly on graph structures.

---

## 🤔 Why Use NaviGraph?

- **Enhanced Sensitivity:** Identify subtle behavioral and cognitive phenotypes undetected by standard metrics.
- **Unified Analysis:** Analyze behavior, neural activity, and physiology within a coherent spatial context.
- **Reproducible Research:** Standardized processing and clear documentation ensure reproducibility across studies.
- **Intuitive Insights:** Graph-based visualizations simplify interpretation of complex navigational strategies.

---

## 💡 Core Concepts

NaviGraph transforms maze data into a **topological graph**:

- **Nodes:** Key decision points in the maze.
- **Edges:** Navigational paths between nodes.
- **Data Layers:** Behavioral metrics, neuronal activity, physiological measures mapped directly onto graph elements.

---

## ⚙️ Workflow

The pipeline is orchestrated by `SessionManager`:

1. **Data Input:** Raw behavioral videos, neuronal recordings, physiological measurements.
2. **Preprocessing:** Pose estimation (DeepLabCut), calcium trace extraction (Minian).
3. **Registration:** Align and validate behavioral data with maze topology.
4. **Data Integration:** Synchronize data streams, stored per session (`.pkl` format).
5. **Analysis:** Compute classic and topological metrics.
6. **Visualization & Aggregation:** Generate visual insights and aggregate results across experimental conditions.

---

## 🚀 Installation

NaviGraph dependencies are managed via [Poetry](https://python-poetry.org/).

```bash
# Clone the repository
git clone https://github.com/your-username/navigraph.git
cd navigraph

# Install dependencies
poetry install
```

---

## ⚡ Quick Start

1. **Prepare your data:** Pose tracking (`.h5`), calcium imaging (`zarr`, optional).
2. **Configure the pipeline:** Customize example YAML config files.
3. **Execute pipeline:**

```bash
poetry shell
python navigraph/run.py --config-path path/to/configs --config-name your_config
```

---

## 🔧 Configuration Overview

Configurations in YAML (via Hydra) include:

- `system_running_mode`: `calibrate`, `test`, `analyze`, `visualize`
- Input paths: Data streams and keypoint detections
- Maze map and output paths
- Calibration and analysis parameters

Refer to [`examples/configs/base_config.yaml`](examples/configs/base_config.yaml).

---

## 📁 Project Structure

```bash
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

If you use NaviGraph, please cite:

> Iton, A. K., Iton, E., Michaelson, D. M., & Blinder, P. (Year). NaviGraph: A graph-based framework for spatial decision-making analysis in maze-like environments. *Journal Name*, Volume(Issue), Pages. [DOI]

*(Update citation upon publication.)*

---

## 🙌 Contributing

Contributions, issue reports, and feature requests are welcome! Please open issues or submit pull requests.

---

## 📜 License

Licensed under the MIT License. See [LICENSE](LICENSE) for more details.