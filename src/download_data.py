import shutil
from pathlib import Path
import numpy as np
import polars as pl
from torchvision.datasets import MNIST

def main():
    # Configuration
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    temp_download_dir = data_dir / "temp_download"
    parquet_path = raw_dir / "mnist.parquet"

    # Ensure directories exist
    raw_dir.mkdir(parents=True, exist_ok=True)
    temp_download_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading MNIST dataset...")
    # Download to a temp folder so we don't clutter our versioned folder with raw binaries
    train_ds = MNIST(root=str(temp_download_dir), train=True, download=True)
    test_ds = MNIST(root=str(temp_download_dir), train=False, download=True)

    print("Converting to Polars DataFrame...")

    def to_polars(ds, split):
        # Data is (N, 28, 28) -> Flatten to (N, 784)
        X = ds.data.numpy().reshape(-1, 28 * 28)
        y = ds.targets.numpy()
        
        # efficient column naming
        schema = [f"pixel_{i}" for i in range(784)]
        
        # Create DataFrame from numpy array
        # orient='row' is implicit for (N, M) array but good to be explicit or let it infer
        df = pl.DataFrame(X, schema=schema)
        
        # Add metadata
        df = df.with_columns([
            pl.Series("label", y),
            pl.lit(split).alias("split")
        ])
        return df

    df_train = to_polars(train_ds, "train")
    df_test = to_polars(test_ds, "test")

    print("Concatenating datasets...")
    df_full = pl.concat([df_train, df_test])

    print(f"Saving to {parquet_path}...")
    df_full.write_parquet(parquet_path)
    
    # Cleanup temp downloads
    print("Cleaning up temp files...")
    shutil.rmtree(temp_download_dir)

    print("Success! Data saved.")

if __name__ == "__main__":
    main()
