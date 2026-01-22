# Plan: MNIST with Ray Train, MLflow, and DVC

This project aims to demonstrate a machine learning workflow using the MNIST dataset.

**Key Technologies:**
- **Ray Train**: For distributed model training.
- **MLflow**: For experiment tracking (metrics, parameters, artifacts).
- **DVC (Data Version Control)**: For data management and versioning.
- **PyTorch**: Deep learning framework.

## Roadmap

### 1. Environment Setup
- [x] Define dependencies in `pyproject.toml`.
    - `ray[train]`, `ray[default]`
    - `mlflow`
    - `dvc`
    - `torch`, `torchvision`
    - `numpy`
    - `polars`
- [x] Install dependencies.

### 2. Data Versioning (DVC)
- [x] Initialize DVC: `dvc init`
- [x] Configure local storage for DVC (simulation of remote).
- [x] Create a script/notebook to download MNIST data raw files.
- [x] Add data to DVC: `dvc add data/raw/mnist.parquet`
- [x] Commit DVC files to git.

### 3. Model & Training Implementation (Ray + PyTorch)
- [x] Create `src/model.py`: Define a simple CNN for MNIST.
- [x] Create `src/train.py`:
    - Implement a Ray Train `train_loop_per_worker`.
    - Handle data loading (integration with DVC-managed paths).
    - Initialize the model, loss, and optimizer.
    - Implement the training loop.

### 4. MLflow Integration
- [ ] Integrate MLflow within `src/train.py`.
    - Use `MLflowLoggerCallback` provided by Ray Train or manual logging inside the loop.
    - Log hyperparameters (config).
    - Log metrics (loss, accuracy) per epoch.
    - Log the final model artifact.

### 5. Execution & Verification
- [ ] Run the training job.
- [ ] Verify logs in MLflow UI: `mlflow ui`.
- [ ] Verify data versioning with DVC.

## File Structure Draft
```text
.
├── dvc.yaml            # DVC pipeline (optional, or just .dvc files)
├── data/               # Git-ignored data folder
│   └── mnist/
├── src/
│   ├── __init__.py
│   ├── model.py        # PyTorch Model definition
│   └── train.py        # Ray Train entry point
├── pyproject.toml
└── README.md
```