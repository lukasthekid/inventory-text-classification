"""
Utility functions for the training pipeline.
Includes logging setup, metric visualization, and helper functions.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
from transformers import set_seed as hf_set_seed

from config import ID_TO_LABEL


def set_seed_everywhere(seed: int) -> None:
    """
    Set random seeds for full reproducibility across libraries.
    
    Args:
        seed: Seed value to use.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting global random seed to {seed}")

    # Python built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNG (CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Hugging Face / Transformers utilities
    hf_set_seed(seed)

    # Ensure deterministic behavior in cuDNN (at potential cost of speed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def save_metrics_to_json(metrics: Dict, output_path: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = float(value)
        else:
            serializable_metrics[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
    
    logging.info(f"Metrics saved to {output_path}")


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    output_path: str,
    title: str = "Confusion Matrix"
):
    """
    Plot and save confusion matrix.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        output_path: Path to save plot
        title: Plot title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))],
        yticklabels=[ID_TO_LABEL[i] for i in range(len(ID_TO_LABEL))]
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Confusion matrix saved to {output_path}")


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str,
    title: str = "Training History"
):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary containing metric histories
        output_path: Path to save plot
        title: Plot title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title)
    
    metrics_to_plot = ['loss', 'accuracy', 'f1', 'learning_rate']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        # Plot train metric
        if f'train_{metric}' in history:
            ax.plot(history[f'train_{metric}'], label=f'Train {metric}')
        
        # Plot validation metric
        if f'val_{metric}' in history:
            ax.plot(history[f'val_{metric}'], label=f'Val {metric}')
        
        ax.set_xlabel('Step')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over time')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training history plot saved to {output_path}")


def print_dataset_statistics(train_df, val_df, test_df):
    """
    Print dataset statistics.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
    """
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("DATASET STATISTICS")
    logger.info("="*60)
    
    total_samples = len(train_df) + len(val_df) + len(test_df)
    
    logger.info(f"\nTotal samples: {total_samples}")
    logger.info(f"Train samples: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
    logger.info(f"Validation samples: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
    logger.info(f"Test samples: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")
    
    logger.info("\nClass distribution:")
    for split_name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        logger.info(f"\n{split_name} set:")
        counts = df['label'].value_counts().sort_index()
        for label, count in counts.items():
            percentage = count / len(df) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
    
    logger.info("\n" + "="*60 + "\n")



