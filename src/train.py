import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import polars as pl
import numpy as np
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

# Handle imports whether running as module or script
try:
    from src.model import MNISTModel
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from src.model import MNISTModel

# --- Custom Dataset for Polars Parquet ---
class MNISTParquetDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading data from: {parquet_path}")
        self.df = pl.read_parquet(parquet_path)
        
        self.labels = self.df["label"].to_numpy()
        # reshape(-1, ...) allows it to infer the batch size (N)
        # We must drop 'label' AND 'split' (added in download_data.py)
        drop_cols = ["label", "split"]
        self.features = self.df.drop(drop_cols).to_numpy().reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_func(config):
    # Hyperparameters
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 0.001)
    epochs = config.get("epochs", 3)
    data_path = config.get("data_path") # Get absolute path from config

    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Worker cannot find data at: {data_path}")

    # Prepare Data
    full_dataset = MNISTParquetDataset(data_path)
    
    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Ray Train's prepare_data_loader handles distributed sampling
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
        # Iterate over batches
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.long() 
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation (Once per epoch)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.long()
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = running_loss / len(train_loader)
        
        # Report metrics to Ray Train
        train.report({"loss": avg_loss, "accuracy": accuracy, "epoch": epoch})

if __name__ == "__main__":
    print("Starting Ray Train with PyTorch...")
    
    # Calculate absolute path to ensure workers find it
    # We assume this script is run from the project root or we can find data relative to it
    project_root = os.getcwd()
    data_path = os.path.join(project_root, "data/raw/mnist.parquet")
    
    if not os.path.exists(data_path):
        # Fallback check if user is in src/
        if os.path.exists("../data/raw/mnist.parquet"):
             data_path = os.path.abspath("../data/raw/mnist.parquet")
        else:
            raise FileNotFoundError(f"Could not find data at {data_path}")

    print(f"Using data path: {data_path}")

    # Use 2 workers (processes) to simulate distributed training
    scaling_config = ScalingConfig(num_workers=2, use_gpu=False)

    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        train_loop_config={
            "batch_size": 64, 
            "epochs": 5, 
            "lr": 0.001,
            "data_path": data_path # Pass the absolute path
        }
    )

    result = trainer.fit()
    print(f"Training finished. Result: {result.metrics}")