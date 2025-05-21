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

Refer to [`examples/configs/base_config.yaml`](configs/base_config.yaml).

---

## 📁 Project Structure

```bash
.
├── configs/
├── examples/
├── docs/
├── navigraph/
│   ├── analysis/
│   ├── modules/
│   ├── session/
│   ├── utils/
│   └── session_manager.py
├── scripts/
├── tests/
├── poetry.lock
├── pyproject.toml
├── README.md
├── LICENSE
└──  run.py
```

---

## 📄 Citation

If you use NaviGraph, please cite:

> Iton, A. K., Iton, E., Michaelson, D. M., & Blinder, P. (2025). NaviGraph: A graph-based framework for multimodal analysis of spatial decision-making. bioRxiv

---

## 🙌 Contributing

Contributions, issue reports, and feature requests are welcome! Please open issues or submit pull requests.

---

## 📜 License

Licensed under the MIT License. See [LICENSE](LICENSE) for more details.
