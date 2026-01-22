"""
Training Pipeline for Inventory Text Classification

This pipeline trains a text classification model to categorize inventory items into:
- labor
- material  
- other

The pipeline:
1. Loads and preprocesses the dataset
2. Trains a DistilBERT model for German text
3. Evaluates on test data
4. Logs all metrics and models to MLflow

Configuration is loaded from config.py.

Usage:
    python train_pipeline.py

Author: ML Engineering Team
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import mlflow

# Try to import mlflow.transformers for better model logging
try:
    import mlflow.transformers
    MLFLOW_TRANSFORMERS_AVAILABLE = True
except ImportError:
    MLFLOW_TRANSFORMERS_AVAILABLE = False

from config import Config
from data_loader import DataLoader
from model_trainer import ModelTrainer
from utils import (
    setup_logging,
    save_metrics_to_json,
    plot_confusion_matrix,
    print_dataset_statistics,
    set_seed_everywhere,
)

# Setup logger
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Training Pipeline for Inventory Text Classification"
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=None,
        help="Directory to save the trained model (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (overrides config)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the dataset file (overrides config)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info("\n1. Loading configuration...")
        config = Config.get_default_config()
        
        # Override config values with command-line arguments if provided
        if args.model_save_dir is not None:
            config.training.model_save_dir = args.model_save_dir
            logger.info(f"Overriding model_save_dir: {args.model_save_dir}")
        
        if args.output_dir is not None:
            config.training.output_dir = args.output_dir
            logger.info(f"Overriding output_dir: {args.output_dir}")
        
        if args.dataset_path is not None:
            config.data.dataset_path = args.dataset_path
            logger.info(f"Overriding dataset_path: {args.dataset_path}")

        # Set all relevant random seeds for reproducibility
        set_seed_everywhere(config.training.seed)

        # Setup logging (paths from config)
        log_dir = Path(config.training.logging_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"training_{timestamp}.log"
        
        setup_logging(log_level=config.logging.log_level, log_file=str(log_file))
        
        logger.info("="*80)
        logger.info("INVENTORY TEXT CLASSIFICATION - TRAINING PIPELINE")
        logger.info("="*80)
    
        # Set run name with timestamp
        config.mlflow.run_name = f"run_{timestamp}"
        
        logger.info("Configuration loaded successfully")
        logger.info(f"Config: {config.to_dict()}")
        
        # Initialize MLflow
        logger.info("\n2. Initializing MLflow...")
        #mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        
        mlflow.set_experiment(config.mlflow.experiment_name)
        # Start MLflow run
        with mlflow.start_run(run_name=config.mlflow.run_name) as run:
            logger.info(f"MLflow run ID: {run.info.run_id}")
            logger.info(f"MLflow experiment: {config.mlflow.experiment_name}")
            
            # Log all hyperparameters and configuration
            logger.info("Logging hyperparameters and configuration to MLflow...")
            config_dict = config.to_dict()
            
            # Log data configuration as parameters
            for key, value in config_dict['data'].items():
                mlflow.log_param(f"data.{key}", value)
            
            # Log model configuration as parameters
            for key, value in config_dict['model'].items():
                mlflow.log_param(f"model.{key}", value)
            
            # Log training configuration as parameters
            for key, value in config_dict['training'].items():
                mlflow.log_param(f"training.{key}", value)
            
            # Log MLflow configuration
            for key, value in config_dict['mlflow'].items():
                if value is not None:  # Skip None values
                    mlflow.log_param(f"mlflow.{key}", value)
            
            # Initialize data loader
            logger.info("\n3. Loading and preprocessing data...")
            data_loader = DataLoader(config.data)
            
            # Initialize model trainer
            logger.info("\n4. Initializing model...")
            trainer = ModelTrainer(config)
            model, tokenizer = trainer.load_model_and_tokenizer()
            
            # Prepare datasets
            logger.info("\n5. Preparing datasets...")
            (train_dataset, val_dataset, test_dataset) = data_loader.prepare_data(tokenizer)
            
            # Train model
            logger.info("\n6. Training model...")
            logger.info("-" * 80)
            train_metrics = trainer.train(train_dataset, val_dataset)
            logger.info("-" * 80)
            logger.info("Training completed!")
            
            # Evaluate on validation set
            logger.info("\n7. Evaluating on validation set...")
            val_metrics, val_preds, val_labels = trainer.evaluate(
                val_dataset, "validation"
            )
            
            # Evaluate on test set
            logger.info("\n8. Evaluating on test set...")
            test_metrics, test_preds, test_labels = trainer.evaluate(
                test_dataset, "test"
            )
            
            # Save model
            logger.info("\n9. Saving model...")
            model_save_path = Path(config.training.model_save_dir)
            trainer.save_model(str(model_save_path))
            
            # Save metrics
            logger.info("\n10. Saving results...")
            results_dir = Path(config.training.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save all metrics to JSON files
            save_metrics_to_json(
                train_metrics,
                results_dir / "train_metrics.json"
            )
            save_metrics_to_json(
                val_metrics,
                results_dir / "val_metrics.json"
            )
            
            # Save test metrics to JSON
            save_metrics_to_json(
                test_metrics,
                results_dir / "test_metrics.json"
            )
            
            # Plot confusion matrix for test set
            plot_confusion_matrix(
                test_labels,
                test_preds,
                results_dir / "confusion_matrix_test.png",
                title="Test Set Confusion Matrix"
            )
            
            # Log artifacts to MLflow
            logger.info("\n11. Logging to MLflow...")
            
            # Log test metrics
            mlflow.log_metrics(test_metrics)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"val_{key}", value)
            
            # Log training metrics
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"train_{key}", value)
            
            # Log artifacts (results directory)
            mlflow.log_artifacts(results_dir, artifact_path="results")
            mlflow.log_artifacts(model_save_path, artifact_path="best_model")
            
            logger.info("MLflow logging completed successfully!")
            
            # Print final summary
            logger.info("\n" + "="*80)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            logger.info(f"\nFinal Test Metrics:")
            logger.info(f"  Accuracy:  {test_metrics.get('test_accuracy', 0):.4f}")
            logger.info(f"  Precision: {test_metrics.get('test_precision', 0):.4f}")
            logger.info(f"  Recall:    {test_metrics.get('test_recall', 0):.4f}")
            logger.info(f"  F1 Score:  {test_metrics.get('test_f1', 0):.4f}")
            logger.info(f"\nModel saved to: {model_save_path}")
            logger.info(f"Results saved to: {results_dir}")
            logger.info(f"MLflow run ID: {run.info.run_id}")
            logger.info("="*80 + "\n")
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training pipeline: {e}", exc_info=True)
        raise
    finally:
        logger.info("Training pipeline execution finished.")


if __name__ == "__main__":
    main()
