import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import polars as pl
import numpy as np
import ray
import mlflow
import mlflow.pytorch
from ray import train
from ray.train import ScalingConfig, RunConfig
from ray.train.torch import TorchTrainer

# Handle imports whether running as module or script
try:
    from src.model import MNISTModel
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from src.model import MNISTModel


class MNISTParquetDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading data from: {parquet_path}")
        self.df = pl.read_parquet(parquet_path)
        self.labels = self.df["label"].to_numpy()
        drop_cols = ["label", "split"]
        self.features = self.df.drop(drop_cols).to_numpy().reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_func(config):
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 0.001)
    epochs = config.get("epochs", 3)
    data_path = config.get("data_path")
    mlflow_tracking_uri = config.get("mlflow_tracking_uri")
    mlflow_experiment_name = config.get("mlflow_experiment_name")

    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Worker cannot find data at: {data_path}")

    # Only rank 0 logs to MLflow to avoid duplicate logging
    is_rank_zero = train.get_context().get_world_rank() == 0

    if is_rank_zero and mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)
        # Enable system metrics (CPU, RAM, GPU usage)
        mlflow.enable_system_metrics_logging()
        # Start MLflow run
        mlflow.start_run()
        # Log hyperparameters
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "num_workers": train.get_context().get_world_size(),
        })

    # Prepare Data
    full_dataset = MNISTParquetDataset(data_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loader = train.torch.prepare_data_loader(train_loader)
    val_loader = train.torch.prepare_data_loader(val_loader)

    model = MNISTModel()
    model = train.torch.prepare_model(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        metrics = {
            "train_loss": avg_train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch,
        }

        # Log metrics to MLflow (rank 0 only)
        if is_rank_zero and mlflow_tracking_uri:
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
            }, step=epoch)

        # Report to Ray Train
        train.report(metrics)

    # Log model artifact at the end (rank 0 only)
    if is_rank_zero and mlflow_tracking_uri:
        # Get unwrapped model for saving
        unwrapped_model = model.module if hasattr(model, "module") else model
        mlflow.pytorch.log_model(unwrapped_model, "model")
        mlflow.end_run()


if __name__ == "__main__":
    print("Starting Ray Train with PyTorch + MLflow...")

    # Config from env vars (for Docker) or defaults (local)
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    ray_address = os.environ.get("RAY_ADDRESS")  # None = start local Ray
    mlflow_experiment_name = "mnist_ray_train"

    project_root = os.getcwd()
    
    # Determine data path based on environment
    if ray_address:
        # In Docker, the project root is mounted to /app
        # We use absolute path to avoid CWD issues on workers
        data_path = "/app/data/raw/mnist.parquet"
    else:
        # Local run - use absolute path from current CWD
        data_path = os.path.abspath("data/raw/mnist.parquet")
        if not os.path.exists(data_path):
             # Try one level up if running from src
             if os.path.exists("../data/raw/mnist.parquet"):
                 data_path = os.path.abspath("../data/raw/mnist.parquet")

    if not ray_address and not os.path.exists(data_path):
         raise FileNotFoundError(f"Could not find data at {data_path} (cwd: {os.getcwd()})")

    print(f"Using data path: {data_path}")
    print(f"MLflow tracking URI: {mlflow_tracking_uri}")
    print(f"MLflow experiment: {mlflow_experiment_name}")
    print(f"Ray address: {ray_address or 'local'}")

    # Determine storage path for Ray results
    if ray_address:
        # If running on cluster (Docker), use the path mapped in docker-compose
        storage_path = "/tmp/ray"
    else:
        # If running locally, use a local directory
        storage_path = os.path.abspath("./ray_results")
    
    print(f"Ray storage path: {storage_path}")

    # Connect to Ray cluster if specified
    if ray_address:
        # Set PYTHONPATH to /app so workers can import src.model
        # This assumes the code is mounted to /app in the Docker container
        ray.init(address=ray_address, runtime_env={"env_vars": {"PYTHONPATH": "/app"}})
    else:
        ray.init()

    use_gpu = torch.cuda.is_available()
    print(f"GPU Available: {use_gpu}")

    scaling_config = ScalingConfig(num_workers=2, use_gpu=use_gpu)
    run_config = RunConfig(storage_path=storage_path, name=mlflow_experiment_name)

    # If running in Docker (Ray Client), workers need to access MLflow via service name
    worker_mlflow_uri = mlflow_tracking_uri
    if ray_address and ("localhost" in worker_mlflow_uri or "127.0.0.1" in worker_mlflow_uri):
        worker_mlflow_uri = worker_mlflow_uri.replace("localhost", "mlflow").replace("127.0.0.1", "mlflow")
        print(f"Adjusted MLflow URI for workers: {worker_mlflow_uri}")

    config = {
        "batch_size": 64,
        "epochs": 5,
        "lr": 0.001,
        "data_path": data_path,
        "mlflow_tracking_uri": worker_mlflow_uri,
        "mlflow_experiment_name": mlflow_experiment_name,
    }

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        train_loop_config=config,
        run_config=run_config,
    )

    print("Starting training...")
    result = trainer.fit()
    print(f"Training finished. Metrics: {result.metrics}")
    print(f"\nView results: mlflow ui --backend-store-uri {mlflow_tracking_uri}")
