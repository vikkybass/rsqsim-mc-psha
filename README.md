# RSQSim Monte Carlo Probabilistic Seismic Hazard Assessment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A physics-based probabilistic seismic hazard assessment framework that integrates RSQSim earthquake catalogs with ground motion prediction equations (GMPEs) using Monte Carlo sampling methods.

## 🎯 Overview

This project implements a Monte Carlo-based PSHA framework that:

- Uses physics-based RSQSim synthetic earthquake catalogs
- Calculates ground motions using NSHM 2023 compliant GMPE ensembles
- Performs regional seismic hazard analysis with spatial optimization
- Provides comprehensive visualization and statistical analysis tools

**Key Features:**

- ✅ NSHM 2023 standard GMPE ensemble (ASK14, BSSA14, CB14, CY14)
- ✅ Finite-rupture distance calculations (Rrup, Rjb, Rx, Ry0)
- ✅ Optimized spatial indexing for large catalogs (100k+ events)
- ✅ HPC-ready parallel processing with memory management
- ✅ Multi-window temporal analysis for hazard validation

## 📋 Requirements

### Core Dependencies

```text
python >= 3.8
numpy >= 1.20
pandas >= 1.3
scipy >= 1.7
```

### Ground Motion Libraries

```text
pygmm >= 2.0  # Ground Motion Prediction Equations
```

### Spatial & Scientific Computing

```text
pyproj >= 3.0
shapely >= 1.8
```

### HPC & Performance

```text
psutil >= 5.8  # Memory management
joblib >= 1.0  # Parallel processing
```

### Visualization (Optional)

```text
matplotlib >= 3.3
seaborn >= 0.11
cartopy >= 0.20  # For map projections
```

## 🚀 Installation

### Option 1: Clone and Install (Recommended)

```bash
# Clone repository
git clone https://github.com/vikkybass/rsqsim_mc_psha.git
cd rsqsim_mc_psha

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Option 2: Direct Installation

```bash
pip install git+https://github.com/vikkybass/rsqsim_mc_psha.git
```

### Option 3: HPC Cluster Setup

```bash
# Load modules on your HPC system
module load python/3.8
module load gcc/9.3.0

# Create environment in scratch space
cd /scratch/$USER
python -m venv rsqsim_env
source rsqsim_env/bin/activate

# Install with --break-system-packages flag if needed
pip install --break-system-packages -r requirements.txt
```

## 📁 Project Structure

```text
rsqsim_mc/
│
├── src/                              # Core source code
│   ├── gm_simulator_main.py          # Main simulation orchestrator
│   ├── gmpe_adapter.py               # PyGMM interface wrapper
│   ├── gmpe_calculator.py            # GMPE calculation engine
│   ├── unified_gmpe.py               # NSHM 2023 ensemble implementation
│   ├── rupture_geometry.py           # Finite-rupture distance calculations
│   ├── scenario_builder.py           # Earthquake scenario construction
│   ├── new_Ran.py                    # RSQSim catalog reader
│   ├── memory_manager.py             # Memory monitoring & optimization
│   └── spatial_indexing.py           # KD-tree spatial queries
│
├── scripts/                          # Execution scripts
│   ├── run_mc.py                     # Monte Carlo orchestrator
│   ├── run_on_cluster.sh             # SLURM batch job template
│   └── submit_parallel_windows.sh    # Parallel job submission
│
├── configs/                          # Configuration files
│   ├── config.py                     # Base configuration
│   └── los_angeles_config.py         # LA-specific regional config
│
├── notebooks/                        # Analysis notebooks
│   ├── Regional_Time_window.ipynb    # Catalog filtering
│   └── time_window_selector.py       # Window generation utilities
│
├── data/                             # Data directory (not in git)
│   ├── windows/                      # Time-windowed subcatalogs
│   └── output/                       # Ground motion results
│
├── tests/                            # Unit tests (to be added)
│
├── docs/                             # Documentation
│
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🔧 Configuration

### Basic Regional Configuration

Create a region-specific config file (e.g., `configs/my_region_config.py`):

```python
from configs.config import load_gmpe_config

def load_config(mode="sequential"):
    base_config = load_gmpe_config()
    
    region_config = {
        "region": {
            "name": "my_region",
            "polygon": [...],  # Regional boundary
            "bounds": {...}
        },
        "gmpe": {
            "use_ensemble": True,
            "models": ["ASK14", "BSSA14", "CB14", "CY14"],
            "ensemble_weights": {
                "ASK14": 0.25, "BSSA14": 0.25,
                "CB14": 0.25, "CY14": 0.25
            }
        }
    }
    
    return {**base_config, **region_config}
```

## 📖 Usage

### 1. Prepare Time Windows

Filter RSQSim catalog into time windows for Monte Carlo sampling:

```bash
# Using the notebook
jupyter notebook notebooks/Regional_Time_window.ipynb
```

### 2. Single Window Analysis

Run analysis on a single time window:

```bash
python scripts/run_mc.py los_angeles sequential \
    --pattern "window_0.csv" \
    --cores 16
```

### 3. Parallel Multi-Window Analysis

Submit multiple windows in parallel on HPC:

```bash
# Submit all windows with default settings
./scripts/submit_parallel_windows.sh los_angeles sequential

# Custom parallel configuration
./scripts/submit_parallel_windows.sh los_angeles sequential \
    --parallel 10 \
    --cores 32 \
    --memory 64GB \
    --time 24:00:00
```

### 4. Monitor Progress

```bash
# Check SLURM queue
squeue -u $USER

# Check individual job output
tail -f /scratch/$USER/rsqsim_jobs/los_angeles/sequential/*.out
```

## 🔬 Scientific Background

### Monte Carlo PSHA Framework

This implementation follows the methodology of:

- **Shaw et al. (2025)**: RSQSim-based Monte Carlo hazard assessment
- **Ebel & Kafka (1999)**: Monte Carlo seismic hazard framework
- **USGS NSHM 2023**: Ground motion model ensemble weighting

### Ground Motion Prediction Equations

Four NGA-West2 models are used in ensemble:

- **ASK14** (Abrahamson, Silva & Kamai, 2014)
- **BSSA14** (Boore, Stewart, Seyhan & Atkinson, 2014)
- **CB14** (Campbell & Bozorgnia, 2014)
- **CY14** (Chiou & Youngs, 2014)

Equal weights (0.25 each) follow NSHM 2023 standard practice.

### Distance Metrics

Finite-rupture distance calculations include:

- **Rrup**: Closest distance to rupture plane
- **Rjb**: Joyner-Boore distance (surface projection)
- **Rx**: Site coordinate along strike
- **Ry0**: Site coordinate perpendicular to strike

## ⚡ Performance Optimization

### Spatial Indexing

- KD-tree implementation for O(log N) event queries
- ~100× speedup vs. naive distance calculations
- Handles catalogs with 100k+ events efficiently

### Memory Management

- Automatic memory monitoring with psutil
- Adaptive batch processing for large site grids
- Memory cleanup between processing batches
- Configurable memory limits per job

### Parallel Processing

- Window-level parallelization for independent analysis
- Site-level parallelization within windows
- Scales to 100+ concurrent jobs on HPC clusters

## 📊 Output Files

### Ground Motion Results

```text
output/
└── {region}/
    └── {mode}/
        ├── window_0_results.csv       # Per-event ground motions
        ├── window_0_summary.json      # Statistical summary
        └── visualizations/
            ├── maps/                   # Spatial hazard maps
            ├── plots/                  # Hazard curves, spectra
            └── gis/                    # GIS-compatible outputs
```

### File Formats

**results.csv**: Per-event ground motions

```csv
event_id,site_lat,site_lon,magnitude,distance_km,pga,sa_0.2s,sa_1.0s
```

**summary.json**: Statistical aggregations

```json
{
  "total_events": 50000,
  "magnitude_range": [5.0, 7.8],
  "mean_pga": 0.15,
  "percentiles": {...}
}
```

## 🧪 Testing

```bash
# Run unit tests (when implemented)
pytest tests/

# Test memory management
python src/memory_manager.py

# Test spatial indexing
python src/spatial_indexing.py
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{rsqsim_mc_psha,
  author = {Victor Olawoyin},
  title = {RSQSim Monte Carlo Probabilistic Seismic Hazard Assessment},
  year = {2025},
  url = {https://github.com/vikkybass/rsqsim_mc_psha}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Boston College Weston Observatory**: Research support
- **SCEC RSQSim**: Earthquake simulation framework
- **USGS**: National Seismic Hazard Model standards
- **PyGMM**: Ground motion prediction equation library

## 📧 Contact

Victor - Boston College, Earth and Environmental Sciences

Project Link: [https://github.com/vikkybass/rsqsim_mc_psha](https://github.com/vikkybass/rsqsim_mc_psha)

---

**Note**: This is research software under active development. For production use, please validate results against established seismic hazard models.
