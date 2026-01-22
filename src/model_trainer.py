"""
Model training and evaluation module.
Handles model initialization, training loop, and evaluation.
"""

import logging
from typing import Dict, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import mlflow
import mlflow.pytorch

from config import Config, ID_TO_LABEL

# Configure logging
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer class for text classification model."""
    
    def __init__(self, config: Config):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")
    
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Load pretrained model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.config.model.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.model_name,
                num_labels=self.config.model.num_labels
            )
            
            self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        return self.model, self.tokenizer
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Predictions from trainer
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Weighted metrics (good for imbalanced data)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted', zero_division=0
        )
        
        # Macro metrics (treats all classes equally, important for imbalanced data)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = (
            precision_recall_fscore_support(
                labels, predictions, average=None, zero_division=0
            )
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision_weighted,  # Weighted by default
            'recall': recall_weighted,
            'f1': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,  # Important for imbalanced classes
        }
        
        # Add per-class metrics
        for idx, label_name in ID_TO_LABEL.items():
            if idx < len(precision_per_class):
                metrics[f'precision_{label_name}'] = precision_per_class[idx]
                metrics[f'recall_{label_name}'] = recall_per_class[idx]
                metrics[f'f1_{label_name}'] = f1_per_class[idx]
        
        return metrics
    
    def _create_trainer(
        self,
        train_dataset,
        val_dataset
    ) -> Trainer:
        """
        Create Hugging Face Trainer.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Configured Trainer object
        """
        logger.info("Creating trainer...")
        
        # Create output directories
        output_dir = Path(self.config.training.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging_dir = Path(self.config.training.logging_dir)
        logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            logging_dir=str(logging_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            per_device_eval_batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            warmup_steps=self.config.training.warmup_steps,
            weight_decay=self.config.training.weight_decay,
            logging_steps=self.config.training.logging_steps,
            #eval_steps=self.config.training.eval_steps,
            #save_steps=self.config.training.save_steps,
            eval_strategy ="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",  # Use macro F1 for imbalanced classes
            greater_is_better=True,
            fp16=self.config.training.fp16,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            seed=self.config.training.seed,
            report_to=["none"],  # Disable default logging, use MLflow instead
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info("Trainer created successfully")
        
        return self.trainer
    
    def train(self, train_dataset, val_dataset) -> Dict:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Training metrics
        """
        logger.info("Starting training...")
        
        # Create trainer if not exists
        self._create_trainer(train_dataset, val_dataset)
        
        try:
            # Train model
            train_result = self.trainer.train()
            
            logger.info("Training completed successfully")
            logger.info(f"Training metrics: {train_result.metrics}")
            
            return train_result.metrics
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def evaluate(
        self, 
        dataset, 
        dataset_name: str = "test"
    ) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            dataset_name: Name of dataset for logging
            
        Returns:
            Tuple of (metrics, predictions, labels)
        """
        logger.info(f"Evaluating on {dataset_name} set...")
        
        try:
            # Get predictions
            predictions_output = self.trainer.predict(dataset)
            
            predictions = np.argmax(predictions_output.predictions, axis=1)
            labels = predictions_output.label_ids
            metrics = predictions_output.metrics
            
            # Generate classification report
            report = classification_report(
                labels,
                predictions,
                target_names=[ID_TO_LABEL[i] for i in range(self.config.model.num_labels)],
                digits=4
            )
            
            logger.info(f"\n{dataset_name.upper()} Classification Report:\n{report}")
            
            # Generate confusion matrix
            cm = confusion_matrix(labels, predictions)
            logger.info(f"\n{dataset_name.upper()} Confusion Matrix:\n{cm}")
            
            logger.info(f"{dataset_name.upper()} metrics: {metrics}")
            
            return metrics, predictions, labels
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def save_model(self, save_path: str):
        """
        Save model and tokenizer.
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        
        try:
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def log_to_mlflow(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        test_metrics: Dict,
        model_path: str
    ):
        """
        Log metrics and model to MLflow.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            test_metrics: Test metrics
            model_path: Path to saved model
        """
        try:
            # Log hyperparameters
            mlflow.log_params(self.config.to_dict()['data'])
            mlflow.log_params(self.config.to_dict()['model'])
            mlflow.log_params(self.config.to_dict()['training'])
            
            # Log training metrics
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train_{key}", value)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                if key.startswith('test_'):
                    key = key.replace('test_', 'val_')
                mlflow.log_metric(key, value)
            
            # Log test metrics
            for key, value in test_metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name="inventory-classifier"
            )
            
            # Log model artifacts
            mlflow.log_artifacts(model_path, artifact_path="model_files")
            
            logger.info("Successfully logged to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")
            # Don't raise - logging failure shouldn't stop the pipeline


