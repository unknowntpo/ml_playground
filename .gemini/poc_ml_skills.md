# Agent Skill: POC Machine Learning Skills

This document defines the core competencies and workflows established in this repository for Proof-of-Concept Machine Learning projects.

## Overview
This skill encompasses the ability to scaffold, execute, and manage end-to-end ML workflows using Ray, DVC, and MLflow.

## Core Capabilities

### 1. Distributed Training & Tuning (Ray)
- **Frameworks:** Ray Train, Ray Tune.
- **Workflow:** Implementing `Trainable` functions or using `TorchTrainer` for scalable model training.
- **Artifacts:** Results are typically stored in `ray_results/`.

### 2. Data Version Control (DVC)
- **Workflow:** Using DVC to track large datasets (e.g., Parquet files) without committing them to Git.
- **Commands:** `dvc add`, `dvc push`, `dvc pull`.
- **Integration:** Data is versioned in `data/` and metadata is kept in `.dvc` files.

### 3. Experiment Tracking (MLflow)
- **Workflow:** Logging metrics, parameters, and models during training runs.
- **Backend:** Local `mlflow.db` or a remote tracking server.
- **UI:** View results via `mlflow ui`.

### 4. Project Orchestration
- **Dependency Management:** `uv` and `pyproject.toml`.
- **Task Runner:** `just` (via `justfile`) for standardizing ML pipelines (e.g., `just train`, `just download-data`).

## Project Structure Conventions
- `src/`: Source code for models and training scripts.
- `data/raw/`: Raw data tracked by DVC.
- `plans/`: Markdown documents outlining project roadmaps.
- `mlruns/`: MLflow experiment data.
- `ray_results/`: Ray training logs and checkpoints.
