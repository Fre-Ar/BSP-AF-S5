# Implicit Global Frontiers 🌍

**Benchmarking Neural Implicit Representations on Political Spherical Geometry**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation for the Bachelor Semester Project: **"Implicit Global Frontiers: Benchmarking Neural Implicit Representations on Political Spherical Geometry."** It provides a complete pipeline for training and evaluating Neural Implicit Representations (NIRs) to approximate the Earth's political distance-to-border field and country classification labels.

## 🛠 Features

* **Global Spherical Sampling**: Hybrid geodesic normal displacement strategy for generating training points biased toward complex borders.
* **Diverse Architectures**: Support for SIREN, MFN (Gabor/Fourier), INCODE, HOSC, SINC, WIRE, FINER, and FR-MLPs.
* **Joint Regression/Classification**: Multi-head architectures predicting continuous distance fields and categorical country IDs (via Softmax or ECOC).
* **Automated Sweeps**: Optuna integration for hyperparameter optimization.

## 🚀 Setup

### Prerequisites

* Python 3.10+
* CUDA-capable GPU (recommended for training, MPS is acceptable)

We recommend using Conda to create a virtual environment:

```bash
# Create and activate environment
conda create -n bsp-nir
conda activate bsp-nir
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Fre-Ar/BSP-AF-S5
cd BSP-AF-S5

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```

> **Note for Windows Users:** If using CUDA, ensure you install the CUDA-enabled version of PyTorch manually via [pytorch.org](https://www.google.com/search?q=https://pytorch.org/get-started/locally/) before installing `requirements.txt`.


## 📂 Codebase Structure

* `src/main.py`: Entry point for training, visualization, and rasterization.
* `src/config.py`: Global hyperparameters, architecture selection, and training constants.
* `src/build_geodata.py`: Pipeline for creating adjacency graphs, ECOC codes, and Parquet training datasets.
* `src/nirs/`:
  * `nns/`: Implementation of individual NIR layers (SIREN, MFN, HOSC, etc.).
  * `training.py`: Training loops with pruning logic for Optuna.
  * `viz/`: Tools for global rasterization and prediction comparison.
* `src/sweep.py`: Optimization script for architecture search under the 8MB limit.

## 📖 Usage

### 1. Data Preparation

1. Download the data files found in the [public OneDrive folder](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/023122492c_uni_lu/IgA96X0uOwBATK9mQAaGxRg1AeyhPTf10V0_mdw4NmC655U?e=KmfGcL).
2. Copy the `data` and `parquet` folders and place them into the `src/geodata` directory.

> **Note:** The `data/world_bank_geodata.gpkg` file is a modified version of the [ADM0 World Bank Official Boundaries (GeoPackage)](https://datacatalog.worldbank.org/search/dataset/0038272/world-bank-official-boundaries).

Alternatively, the same files (except `data/world_bank_geodata.gpkg`) can be generated using the codebase by running all commented functions in `src/build_geodata.py`:

```python
if __name__ == "__main__":
    create_adjacency_graph() # creates adjacency graph for ECOC classificatiom
    create_ecoc() # creates ECOC mapping
    preprocess_borders() # proprocesses borders from GPKG to FGB
    create_training_data() # samples training points into parquet files
```

### 2. Configuration

Modify `src/config.py` to select your model and parameters:

* `MODEL`: Choice of `"relu"`, `"siren"`, `"mfn_gabor"`, `"hosc"`, etc.
* `INIT_REGIME`: Choice of initialization regime (SIREN-style, default, etc.)
* `ENCODING`: Choice of Coordinate encoding, `None` by default.
* `TOTAL_LAYERS` & `WIDTH`: Defines the MLP trunk.
* `LR`: the learning rate.

### 3. Training & Inference

Run tasks by uncommenting the relevant lines in `src/main.py`:

```python
if __name__ == "__main__":
    train() # Start training
    viz()   # Compare predictions vs Ground Truth
    img()   # Rasterize a global map

```

### 4. Hyperparameter Sweep

To find the optimal hyperparameters for a specific architectures, a skeleton module that may be modified is provided:

```bash
python src/sweep.py

```

## 📊 Benchmarking Metrics

The models are evaluated on:

* **L1/MSE**: Log-Distance field regression accuracy.
* **Classification Accuracy**: Country identification (Top-1).

## 🔗 Supplementary Material

* **Pretrained Model checkpoints:** Available in the [OneDrive folder](https://uniluxembourg-my.sharepoint.com/:f:/g/personal/023122492c_uni_lu/IgA96X0uOwBATK9mQAaGxRg1AeyhPTf10V0_mdw4NmC655U?e=KmfGcL) inside `checkpoints`. It is recommended to place them inside the `src` folder to run the `viz()` and `img()` functions.

## ⚖️ License

This project was produced at the **University of Luxembourg** (Academic Year 2025/26).