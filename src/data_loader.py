"""
Data loading and preprocessing module.
Handles dataset loading, cleaning, and preparation for training.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

from config import DataConfig, LABEL_MAP
from utils import parse_german_float

# Configure logging
logger = logging.getLogger(__name__)


class InventoryDataset(Dataset):
    """PyTorch Dataset for inventory text classification."""
    
    def __init__(
        self, 
        texts: list, 
        labels: list, 
        tokenizer: AutoTokenizer, 
        max_length: int,
        quantities: list = None,
        units: list = None,
        prices: list = None,
        use_additional_features: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            texts: List of text samples
            labels: List of label indices
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            quantities: List of quantity values (optional)
            units: List of unit strings (optional)
            prices: List of price values (optional)
            use_additional_features: Whether to concatenate additional features to text
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.quantities = quantities
        self.units = units
        self.prices = prices
        self.use_additional_features = use_additional_features
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def _format_features(self, text: str, quantity: float = None, unit: str = None, price: float = None) -> str:
        """
        Format text with additional features.
        
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
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from dataset."""
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DataLoader:
    """Data loader for inventory text classification."""
    
    def __init__(self, config: DataConfig):
        """
        Initialize data loader.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.df = None
        self.tokenizer = None
        
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that expected columns exist in the dataset.
        
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = [self.config.text_column, self.config.label_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Expected columns: {required_cols}. "
                f"Available columns: {list(df.columns)}"
            )
        
        if self.config.use_additional_features:
            optional_cols = [
                self.config.quantity_column,
                self.config.unit_column,
                self.config.price_column,
            ]
            missing_optional = [col for col in optional_cols if col not in df.columns]
            if missing_optional:
                raise ValueError(
                    f"Additional feature columns not found: {missing_optional}. "
                    f"Expected columns (when use_additional_features=True): {required_cols + optional_cols}. "
                    f"Available columns: {list(df.columns)}"
                )
        
        logger.info(f"Column validation passed. Columns: {list(df.columns)}")

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If dataset file not found
            ValueError: If required columns missing
        """
        dataset_path = Path(self.config.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. "
                "Please ensure the file exists."
            )
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        try:
            df = pd.read_csv(dataset_path, encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        
        self._validate_columns(df)
        
        logger.info(f"Dataset loaded successfully with {len(df)} samples")
        self.df = df
        
        return df
    
    def _preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess dataset: clean text, handle missing values, etc.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing dataset...")
        
        df = df.copy()
        
        # Remove rows with missing text or labels
        initial_len = len(df)
        df = df.dropna(subset=[self.config.text_column, self.config.label_column])
        
        if len(df) < initial_len:
            logger.warning(
                f"Removed {initial_len - len(df)} rows with missing values"
            )
        
        # Convert text to string and strip whitespace
        df[self.config.text_column] = (
            df[self.config.text_column]
            .astype(str)
            .str.strip()
        )
        
        # Preprocess additional features if enabled
        if self.config.use_additional_features:
            # Handle quantity column: convert German decimal format (comma) to float
            if self.config.quantity_column in df.columns:
                df[self.config.quantity_column] = df[self.config.quantity_column].apply(
                    parse_german_float
                )
            else:
                df[self.config.quantity_column] = np.nan
            
            # Handle unit column
            if self.config.unit_column in df.columns:
                df[self.config.unit_column] = (
                    df[self.config.unit_column]
                    .astype(str)
                    .str.strip()
                )
            else:
                df[self.config.unit_column] = None
            
            # Handle price column: convert German decimal format (comma) to float
            if self.config.price_column in df.columns:
                df[self.config.price_column] = df[self.config.price_column].apply(
                    parse_german_float
                )
            else:
                df[self.config.price_column] = np.nan
        
        # Normalize labels to lowercase
        df[self.config.label_column] = (
            df[self.config.label_column]
            .astype(str)
            .str.lower()
            .str.strip()
        )
        # Map labels to integers
        df['label_id'] = df[self.config.label_column].map(LABEL_MAP)
        
        # Log class distribution
        label_counts = df[self.config.label_column].value_counts()
        logger.info(f"Class distribution:\n{label_counts}")
        
        logger.info(f"Preprocessing complete. Final dataset size: {len(df)}")
        
        return df
    
    def _split_dataset(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting dataset...")
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=df['label_id']
        )
        
        # Second split: train vs val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=train_val_df['label_id']
        )
        
        logger.info(
            f"Dataset split: "
            f"Train={len(train_df)}, "
            f"Val={len(val_df)}, "
            f"Test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def _create_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: AutoTokenizer
    ) -> Tuple[InventoryDataset, InventoryDataset, InventoryDataset]:
        """
        Create PyTorch datasets from DataFrames.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            tokenizer: Tokenizer to use
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("Creating PyTorch datasets...")
        
        self.tokenizer = tokenizer
        
        # Prepare additional features if enabled
        if self.config.use_additional_features:
            train_quantities = train_df[self.config.quantity_column].tolist() if self.config.quantity_column in train_df.columns else None
            train_units = train_df[self.config.unit_column].tolist() if self.config.unit_column in train_df.columns else None
            train_prices = train_df[self.config.price_column].tolist() if self.config.price_column in train_df.columns else None
            
            val_quantities = val_df[self.config.quantity_column].tolist() if self.config.quantity_column in val_df.columns else None
            val_units = val_df[self.config.unit_column].tolist() if self.config.unit_column in val_df.columns else None
            val_prices = val_df[self.config.price_column].tolist() if self.config.price_column in val_df.columns else None
            
            test_quantities = test_df[self.config.quantity_column].tolist() if self.config.quantity_column in test_df.columns else None
            test_units = test_df[self.config.unit_column].tolist() if self.config.unit_column in test_df.columns else None
            test_prices = test_df[self.config.price_column].tolist() if self.config.price_column in test_df.columns else None
            
            logger.info("Using additional features: quantity, unit, price")
        else:
            train_quantities = train_units = train_prices = None
            val_quantities = val_units = val_prices = None
            test_quantities = test_units = test_prices = None
        
        train_dataset = InventoryDataset(
            texts=train_df[self.config.text_column].tolist(),
            labels=train_df['label_id'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            quantities=train_quantities,
            units=train_units,
            prices=train_prices,
            use_additional_features=self.config.use_additional_features
        )
        
        val_dataset = InventoryDataset(
            texts=val_df[self.config.text_column].tolist(),
            labels=val_df['label_id'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            quantities=val_quantities,
            units=val_units,
            prices=val_prices,
            use_additional_features=self.config.use_additional_features
        )
        
        test_dataset = InventoryDataset(
            texts=test_df[self.config.text_column].tolist(),
            labels=test_df['label_id'].tolist(),
            tokenizer=tokenizer,
            max_length=self.config.max_length,
            quantities=test_quantities,
            units=test_units,
            prices=test_prices,
            use_additional_features=self.config.use_additional_features
        )
        
        logger.info("PyTorch datasets created successfully")
        
        return train_dataset, val_dataset, test_dataset
    
    def prepare_data(
        self, 
        tokenizer: AutoTokenizer
    ) -> Tuple[InventoryDataset, InventoryDataset, InventoryDataset]:
        """
        Complete data preparation pipeline.
        
        Args:
            tokenizer: Tokenizer to use
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset, 
                     train_df, val_df, test_df)
        """
        # Load data
        df = self._load_dataset()
        
        # Preprocess
        df = self._preprocess_dataset(df)
        
        # Split
        train_df, val_df, test_df = self._split_dataset(df)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self._create_datasets(
            train_df, val_df, test_df, tokenizer
        )
        
        return (
            train_dataset, val_dataset, test_dataset
        )


