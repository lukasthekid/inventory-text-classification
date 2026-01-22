"""
Configuration module for training pipeline.
Contains all hyperparameters, paths, and settings.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

@dataclass
class LoggingConfig:
    """Logging-related configuration."""
    log_level: str = "INFO"

@dataclass
class DataConfig:
    """Data-related configuration."""
    dataset_path: str = "../data/dataset.xlsx"
    text_column: str = "text"
    label_column: str = "label"
    quantity_column: str = "quantity"  # Column name for quantity feature
    unit_column: str = "unit"  # Column name for unit feature
    price_column: str = "price"  # Column name for price feature
    use_additional_features: bool = True  # Whether to use quantity, unit, price features
    test_size: float = 0.2
    val_size: float = 0.1  # Validation split from training data
    random_state: int = 42
    max_length: int = 128  # Maximum sequence length for tokenization
    

@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_name: str = "distilbert/distilbert-base-german-cased"
    num_labels: int = 3
    dropout: float = 0.1
    

@dataclass
class TrainingConfig:
    """Training-related configuration."""
    output_dir: str = "../models/checkpoints"
    logging_dir: str = "../logs"
    model_save_dir: str = "../models"
    results_dir: str = "../results"
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.001
    warmup_steps: int = 500
    weight_decay: float = 0.01
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 50
    seed: int = 42
    fp16: bool = False  # Mixed precision training (set True if GPU available)
    gradient_accumulation_steps: int = 1
    use_class_weights: bool = True  # Use class weights to handle imbalance
    

@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    experiment_name: str = "The Tean Experiment"
    tracking_uri: Optional[str] = "http://127.0.0.1:5000/"
    run_name: Optional[str] = None
    

@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    logging: LoggingConfig
    
    @classmethod
    def get_default_config(cls) -> "Config":
        """Get default configuration."""
        return cls(
            logging=LoggingConfig(),
            data=DataConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            mlflow=MLflowConfig()
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging and MLflow."""
        return asdict(self)


# Label mapping
LABEL_MAP = {
    "labor": 0,
    "material": 1,
    "other": 2
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


