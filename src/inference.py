"""
Inference Pipeline for Inventory Text Classification

This script loads a trained model and performs predictions on new text data.
It accepts a CSV file, performs preprocessing, makes predictions, and saves
the results back to a new CSV file with predicted labels appended.

Usage:
    python inference.py --input_csv path/to/data.csv --model_path path/to/model
    python inference.py --input_csv path/to/data.csv  # Uses default model path from config

Author: ML Engineering Team
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config, ID_TO_LABEL
from utils import setup_logging

# Setup logger
logger = logging.getLogger(__name__)


class InferenceDataset(Dataset):
    """PyTorch Dataset for inference (no labels needed)."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: AutoTokenizer, 
        max_length: int,
        quantities: Optional[List] = None,
        units: Optional[List] = None,
        prices: Optional[List] = None,
        use_additional_features: bool = True
    ):
        """
        Initialize inference dataset.
        
        Args:
            texts: List of text samples
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            quantities: List of quantity values (optional)
            units: List of unit strings (optional)
            prices: List of price values (optional)
            use_additional_features: Whether to concatenate additional features to text
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quantities = quantities
        self.units = units
        self.prices = prices
        self.use_additional_features = use_additional_features
    
    def _format_features(
        self, 
        text: str, 
        quantity: Optional[float] = None, 
        unit: Optional[str] = None, 
        price: Optional[float] = None
    ) -> str:
        """
        Format text with additional features (reused from InventoryDataset).
        
        Args:
            text: Original text
            quantity: Quantity value
            unit: Unit string
            price: Price value
            
        Returns:
            Formatted text string with features concatenated
        """
        if not self.use_additional_features:
            return text
        
        feature_parts = []
        
        # Add quantity if available
        if quantity is not None:
            try:
                # Check if it's a valid numeric value (not NaN)
                if pd.notna(quantity) and not (isinstance(quantity, float) and np.isnan(quantity)):
                    feature_parts.append(f"Menge: {quantity}")
            except (TypeError, ValueError):
                pass
        
        # Add unit if available
        if unit is not None:
            unit_str = str(unit).strip()
            if unit_str and unit_str.lower() not in ['nan', 'none', '']:
                feature_parts.append(f"Einheit: {unit_str}")
        
        # Add price if available
        if price is not None:
            try:
                # Check if it's a valid numeric value (not NaN)
                if pd.notna(price) and not (isinstance(price, float) and np.isnan(price)):
                    feature_parts.append(f"Preis: {price}")
            except (TypeError, ValueError):
                pass
        
        # Concatenate features to text
        if feature_parts:
            features_text = " | ".join(feature_parts)
            return f"{text} | {features_text}"
        
        return text
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single item from dataset."""
        text = str(self.texts[idx])
        
        # Get additional features if available
        quantity = self.quantities[idx] if self.quantities is not None else None
        unit = self.units[idx] if self.units is not None else None
        price = self.prices[idx] if self.prices is not None else None
        
        # Format text with additional features
        formatted_text = self._format_features(text, quantity, unit, price)
        
        # Tokenize text
        encoding = self.tokenizer(
            formatted_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


class InferencePipeline:
    """Pipeline for performing inference on new data."""
    
    def __init__(
        self, 
        model_path: str,
        config: Optional[Config] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to saved model directory
            config: Configuration object (optional, uses default if not provided)
            device: PyTorch device (auto-detected if not provided)
        """
        self.model_path = Path(model_path)
        self.config = config or Config.get_default_config()
        self.device = device or torch.device("cpu")
        
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Using device: {self.device} (CPU-only)")
    
    def load_model(self):
        """Load model and tokenizer from saved directory."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found at {self.model_path}. "
                "Please ensure the model was saved correctly."
            )
        
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def preprocess_data(
        self, 
        df: pd.DataFrame,
        text_column: str,
        quantity_column: Optional[str] = None,
        unit_column: Optional[str] = None,
        price_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Preprocess input DataFrame (similar to DataLoader._preprocess_dataset).
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            quantity_column: Name of quantity column (optional)
            unit_column: Name of unit column (optional)
            price_column: Name of price column (optional)
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Convert text to string and strip whitespace
        df[text_column] = df[text_column].astype(str).str.strip()
        
        # Handle additional features if columns exist
        if quantity_column and quantity_column in df.columns:
            df[quantity_column] = pd.to_numeric(
                df[quantity_column], errors='coerce'
            )
        
        if unit_column and unit_column in df.columns:
            df[unit_column] = df[unit_column].astype(str).str.strip()
        
        if price_column and price_column in df.columns:
            df[price_column] = pd.to_numeric(
                df[price_column], errors='coerce'
            )
        
        logger.info(f"Preprocessed {len(df)} samples")
        
        return df
    
    def predict(
        self, 
        df: pd.DataFrame,
        text_column: str = "text",
        batch_size: int = 32,
        quantity_column: Optional[str] = None,
        unit_column: Optional[str] = None,
        price_column: Optional[str] = None
    ) -> List[str]:
        """
        Perform predictions on DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column
            batch_size: Batch size for inference
            quantity_column: Name of quantity column (optional)
            unit_column: Name of unit column (optional)
            price_column: Name of price column (optional)
            
        Returns:
            List of predicted labels
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first. Call load_model()")
        
        logger.info(f"Making predictions on {len(df)} samples...")
        
        # Prepare texts and features
        texts = df[text_column].tolist()
        quantities = df[quantity_column].tolist() if quantity_column and quantity_column in df.columns else None
        units = df[unit_column].tolist() if unit_column and unit_column in df.columns else None
        prices = df[price_column].tolist() if price_column and price_column in df.columns else None
        
        # Create dataset
        dataset = InferenceDataset(
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_length,
            quantities=quantities,
            units=units,
            prices=prices,
            use_additional_features=self.config.data.use_additional_features
        )
        
        # Create data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        # Make predictions
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get predicted class indices
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        # Convert predictions to labels
        predicted_labels = [ID_TO_LABEL[pred] for pred in predictions]
        
        logger.info(f"Predictions completed. Predicted {len(predicted_labels)} labels")
        
        return predicted_labels


def main():
    """Main inference pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Inference Pipeline for Inventory Text Classification"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model directory (overrides config if provided)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_csv with '_predictions' suffix)"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="Name of text column (overrides config if provided)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        setup_logging(log_level="INFO")
        
        logger.info("="*80)
        logger.info("INVENTORY TEXT CLASSIFICATION - INFERENCE PIPELINE")
        logger.info("="*80)
        
        # Load configuration
        config = Config.get_default_config()
        
        # Override model path if provided
        model_path = args.model_path or config.training.model_save_dir
        if args.model_path:
            logger.info(f"Overriding model_path: {args.model_path}")
        
        # Override text column if provided
        text_column = args.text_column or config.data.text_column
        if args.text_column:
            logger.info(f"Overriding text_column: {args.text_column}")
        
        # Load input CSV
        input_csv_path = Path(args.input_csv)
        if not input_csv_path.exists():
            raise FileNotFoundError(f"Input CSV file not found: {input_csv_path}")
        
        logger.info(f"\n1. Loading input CSV from {input_csv_path}")
        df = pd.read_csv(input_csv_path, sep="\t")
        logger.info(f"Loaded {len(df)} rows")
        
        # Validate text column exists
        if text_column not in df.columns:
            raise ValueError(
                f"Text column '{text_column}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Initialize inference pipeline
        logger.info(f"\n2. Initializing inference pipeline...")
        pipeline = InferencePipeline(model_path=model_path, config=config)
        
        # Load model
        logger.info(f"\n3. Loading model from {model_path}...")
        pipeline.load_model()
        
        # Preprocess data
        logger.info(f"\n4. Preprocessing data...")
        df_processed = pipeline.preprocess_data(
            df=df,
            text_column=text_column,
            quantity_column=config.data.quantity_column if config.data.use_additional_features else None,
            unit_column=config.data.unit_column if config.data.use_additional_features else None,
            price_column=config.data.price_column if config.data.use_additional_features else None
        )
        
        # Make predictions
        logger.info(f"\n5. Making predictions...")
        predicted_labels = pipeline.predict(
            df=df_processed,
            text_column=text_column,
            batch_size=args.batch_size,
            quantity_column=config.data.quantity_column if config.data.use_additional_features else None,
            unit_column=config.data.unit_column if config.data.use_additional_features else None,
            price_column=config.data.price_column if config.data.use_additional_features else None
        )
        
        # Append predictions to DataFrame
        logger.info(f"\n6. Appending predictions to DataFrame...")
        df['predicted_label'] = predicted_labels
        
        # Determine output path
        if args.output_csv:
            output_csv_path = Path(args.output_csv)
        else:
            # Create output filename with '_predictions' suffix
            stem = input_csv_path.stem
            suffix = input_csv_path.suffix
            output_csv_path = input_csv_path.parent / f"{stem}_predictions{suffix}"
        
        # Save results
        logger.info(f"\n7. Saving results to {output_csv_path}...")
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False, sep="\t")
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("INFERENCE PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"\nPredictions summary:")
        label_counts = pd.Series(predicted_labels).value_counts()
        for label, count in label_counts.items():
            percentage = count / len(predicted_labels) * 100
            logger.info(f"  {label}: {count} ({percentage:.1f}%)")
        logger.info(f"\nResults saved to: {output_csv_path}")
        logger.info("="*80 + "\n")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in inference pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

