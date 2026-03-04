"""
FastAPI application for inventory text classification.

Endpoints:
- POST /train: Start training with CSV dataset (async, returns 202)
- GET /metrics: Get test metrics of the newest model
- POST /predict: Run inference on CSV, return CSV with predictions
"""

import io
import json
import logging
import subprocess
import uuid
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from config import Config
from inference import InferencePipeline

# Project root (api.py lives in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load config and resolve paths
config = Config.get_default_config()
config.training.model_save_dir = str(PROJECT_ROOT / "models" / "best_model")
config.training.results_dir = str(PROJECT_ROOT / "results")
config.training.output_dir = str(PROJECT_ROOT / "models" / "checkpoints")
config.training.logging_dir = str(PROJECT_ROOT / "logs")

# Upload directory for training datasets
UPLOADS_DIR = PROJECT_ROOT / "data" / "api_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Inventory Text Classification API",
    description="Train, evaluate, and run inference on inventory text classification",
)

# Lazy-loaded inference pipeline
_inference_pipeline: InferencePipeline | None = None


def _get_inference_pipeline() -> InferencePipeline:
    """Lazy-load the inference pipeline."""
    global _inference_pipeline
    if _inference_pipeline is None:
        model_path = config.training.model_save_dir
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Train a model first via POST /train",
            )
        _inference_pipeline = InferencePipeline(
            model_path=model_path,
            config=config,
        )
        _inference_pipeline.load_model()
    return _inference_pipeline


def _parse_csv_content(content: bytes) -> pd.DataFrame:
    """Parse CSV with auto-detected delimiter (comma or tab)."""
    try:
        df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}") from e
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    return df


def _validate_training_columns(df: pd.DataFrame) -> None:
    """Validate required columns for training."""
    required = [config.data.text_column, config.data.label_column]
    if config.data.use_additional_features:
        required.extend([
            config.data.quantity_column,
            config.data.unit_column,
            config.data.price_column,
        ])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}. Expected: {required}. Found: {list(df.columns)}",
        )


def _validate_predict_columns(df: pd.DataFrame) -> None:
    """Validate required columns for inference (text column required)."""
    if config.data.text_column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Missing text column: {config.data.text_column}. Found: {list(df.columns)}",
        )


@app.post("/train")
async def train(file: Annotated[UploadFile, File()]):
    """
    Start training with a CSV dataset.
    Returns 202 immediately; training runs in background.
    CSV must include: text, label, and optionally quantity, unit, price.
    """
    content = await file.read()
    df = _parse_csv_content(content)
    _validate_training_columns(df)

    # Save to uploads dir with unique name
    train_id = str(uuid.uuid4())[:8]
    dataset_path = UPLOADS_DIR / f"train_{train_id}.csv"
    df.to_csv(dataset_path, index=False, encoding="utf-8")

    # Spawn training subprocess (fire-and-forget)
    abs_path = str(dataset_path.resolve())
    src_dir = str(PROJECT_ROOT / "src")
    cmd = [
        "uv", "run", "--directory", src_dir,
        "train_pipeline.py",
        "--dataset_path", abs_path,
    ]
    try:
        subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=None,
            stderr=None,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="uv not found. Ensure uv is installed and in PATH.",
        ) from None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return JSONResponse(
        status_code=202,
        content={
            "message": "Training started",
            "dataset_path": str(dataset_path),
            "train_id": train_id,
        },
    )


@app.get("/metrics")
async def get_metrics():
    """
    Get test metrics of the most recently trained model.
    Returns accuracy, precision, recall, and f1 score.
    """
    metrics_path = Path(config.training.results_dir) / "test_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No metrics found. Train a model first via POST /train.",
        )
    try:
        with open(metrics_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"Cannot read metrics: {e}") from e

    return {
        "accuracy": data.get("test_accuracy"),
        "precision": data.get("test_precision"),
        "recall": data.get("test_recall"),
        "f1": data.get("test_f1"),
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]):
    """
    Run inference on a CSV file.
    Returns a CSV with a predicted_label column appended.
    Input must include a text column; optionally quantity, unit, price.
    """
    content = await file.read()
    df = _parse_csv_content(content)
    _validate_predict_columns(df)

    pipeline = _get_inference_pipeline()
    df_processed = pipeline.preprocess_data(
        df=df,
        text_column=config.data.text_column,
        quantity_column=config.data.quantity_column if config.data.use_additional_features else None,
        unit_column=config.data.unit_column if config.data.use_additional_features else None,
        price_column=config.data.price_column if config.data.use_additional_features else None,
    )
    predictions = pipeline.predict(
        df=df_processed,
        text_column=config.data.text_column,
        quantity_column=config.data.quantity_column if config.data.use_additional_features else None,
        unit_column=config.data.unit_column if config.data.use_additional_features else None,
        price_column=config.data.price_column if config.data.use_additional_features else None,
    )
    df["predicted_label"] = predictions

    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )
