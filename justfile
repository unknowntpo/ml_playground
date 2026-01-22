# Show help message
help:
    just --list

# Start Docker services
up:
    docker compose up -d
    @echo "Waiting for services..."
    @sleep 5
    @echo "MLflow UI: http://localhost:5000"
    @echo "Ray Dashboard: http://localhost:8265"

# Stop Docker services
down:
    docker compose down

# Train with Docker services
train:
    MLFLOW_TRACKING_URI=http://localhost:5000 RAY_ADDRESS=ray://localhost:10001 uv run src/train.py

# Train locally (no Docker)
train-local:
    uv run src/train.py

# Open MLflow UI (Docker)
mlflow-ui:
    open http://localhost:5000

# Open MLflow UI (local db)
mlflow-ui-local:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

# View Docker logs
logs:
    docker compose logs -f

# Clean up
clean:
    docker compose down -v
    rm -rf mlflow_data ray_results
