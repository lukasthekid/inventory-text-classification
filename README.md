# Inventory Text Classification

A production-ready machine learning pipeline for classifying German inventory text into three categories: **labor**, **material**, and **other**.

## 🎯 Project Overview

This project implements a complete text classification system using state-of-the-art transformer models (DistilBERT) fine-tuned on German inventory data. The pipeline includes data preprocessing, model training, evaluation, and inference capabilities with full MLflow experiment tracking.

### Key Features

- ✅ **Modular Architecture**: Clean separation of concerns across multiple modules
- ✅ **Production-Ready**: Comprehensive error handling and logging
- ✅ **Experiment Tracking**: Full MLflow integration for reproducibility
- ✅ **Best Practices**: Type hints, docstrings, parameterized configs
- ✅ **German Language Support**: Optimized for German text with DistilBERT
- ✅ **Easy Inference**: Simple API for predictions on new data

## 📊 Dataset

The dataset contains inventory item descriptions with the following structure:

| Column | Description |
|--------|-------------|
| id | Unique identifier |
| text | Item description (German) |
| quantity | Item quantity |
| unit | Unit of measurement |
| price | Item price |
| label | Category (labor/material/other) |

**Class Distribution:**
- Labor: 80 samples (~35%)
- Material: 137 samples (~60%)
- Other: 12 samples (~5%)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd inventory-text-classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd scripts
python train_pipeline.py
```

The pipeline will:
1. Load and preprocess the dataset
2. Split into train/validation/test sets
3. Fine-tune DistilBERT model
4. Evaluate on test data
5. Save the best model
6. Log everything to MLflow

### 3. View Results

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

### 4. Run Inference

```bash
# Single prediction
python inference.py \
    --model ../models/best_model_TIMESTAMP \
    --text "Deckenfläche streichen" \
    --probabilities

# Batch predictions
python inference.py \
    --model ../models/best_model_TIMESTAMP \
    --input ../data/new_items.csv \
    --output ../predictions.csv
```

## 📁 Project Structure

```
inventory-text-classification/
├── data/
│   └── dataset.xlsx              # Training dataset
├── scripts/
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── model_trainer.py          # Training & evaluation
│   ├── utils.py                  # Utility functions
│   ├── train_pipeline.py         # Main training script
│   ├── inference.py              # Inference script
│   └── README.md                 # Detailed scripts documentation
├── notebooks/
│   └── EDA.ipynb                 # Exploratory data analysis
├── models/                       # Saved models (generated)
├── results/                      # Training results (generated)
├── logs/                         # Training logs (generated)
├── mlruns/                       # MLflow tracking (generated)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

All hyperparameters are centralized in `scripts/config.py`:

```python
# Data Configuration
dataset_path = "data/dataset.xlsx"
max_length = 128
test_size = 0.2
val_size = 0.1

# Model Configuration
model_name = "distilbert/distilbert-base-german-cased"
num_labels = 3

# Training Configuration
num_epochs = 5
batch_size = 16
learning_rate = 2e-5
```

Modify these values to experiment with different settings!

## 📈 Model Performance

Expected performance metrics after training:

| Metric | Score |
|--------|-------|
| Accuracy | ~0.85+ |
| Precision (weighted) | ~0.85+ |
| Recall (weighted) | ~0.85+ |
| F1 Score (weighted) | ~0.85+ |

*Note: Actual performance depends on data split and training conditions*

## 🛠️ Advanced Usage

### Custom Configuration

```bash
python train_pipeline.py \
    --log-level DEBUG \
    --experiment-name custom-experiment \
    --run-name trial-001
```

### GPU Training

The pipeline automatically detects and uses GPU if available. To force CPU:

```python
# In config.py
fp16 = False  # Disable mixed precision training
```

### Batch Inference with Probabilities

```bash
python inference.py \
    --model models/best_model_20240120 \
    --input data/inventory_items.csv \
    --output predictions_with_probs.csv \
    --probabilities \
    --text-column description
```

## 📊 Monitoring Training

Training progress is logged to three places:

1. **Console Output**: Real-time progress bars and metrics
2. **Log Files**: Detailed logs in `logs/training_TIMESTAMP.log`
3. **MLflow UI**: Interactive dashboard with metrics and artifacts

### What's Tracked in MLflow

- **Parameters**: All hyperparameters and config values
- **Metrics**: 
  - Accuracy, Precision, Recall, F1 (overall)
  - Per-class metrics for each category
  - Training/validation/test splits
- **Artifacts**:
  - Trained model files
  - Confusion matrices
  - Metrics JSON files
  - Training logs

## 🎓 Model Architecture

**Base Model**: DistilBERT (German)
- Smaller, faster version of BERT
- Pre-trained on German text corpus
- 6 transformer layers
- 66M parameters

**Fine-tuning**:
- Classification head added on top
- Trained end-to-end on inventory data
- Early stopping to prevent overfitting

## 🔍 Data Preprocessing

The pipeline performs:

1. **Data Validation**: Check for required columns
2. **Missing Value Handling**: Remove incomplete samples
3. **Text Normalization**: Strip whitespace, lowercase labels
4. **Label Mapping**: Convert text labels to integers
5. **Stratified Splitting**: Maintain class distribution across splits
6. **Tokenization**: Convert text to BERT input format

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory Error**
```python
# Reduce batch size in config.py
batch_size = 8
gradient_accumulation_steps = 2
```

**2. Model Download Issues**
```bash
# Set cache directory
export TRANSFORMERS_CACHE=./cache
```

**3. Missing Dependencies**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

**4. Poor Performance on "Other" Class**
- Expected due to class imbalance (only 12 samples)
- Consider data augmentation or oversampling
- Collect more samples for minority class

## 🧪 Testing

### Validate Installation

```python
# Test data loading
from scripts.data_loader import DataLoader
from scripts.config import DataConfig

loader = DataLoader(DataConfig())
df = loader.load_dataset()
print(f"Loaded {len(df)} samples")
```

### Quick Training Test

```bash
# Train for 1 epoch to verify setup
python train_pipeline.py --log-level DEBUG
# Then check logs/training_*.log
```

## 📚 Best Practices Implemented

1. **Code Organization**: Modular design with single responsibility principle
2. **Configuration Management**: Centralized, parameterized settings
3. **Error Handling**: Comprehensive try-except blocks with informative messages
4. **Logging**: Multi-level logging (console + file + MLflow)
5. **Type Safety**: Full type hints for better IDE support
6. **Documentation**: Docstrings for all public functions/classes
7. **Reproducibility**: Fixed random seeds, version pinning
8. **Experiment Tracking**: MLflow for full reproducibility
9. **Code Quality**: No hardcoded values, clean code standards
10. **Version Control**: Proper .gitignore for ML projects

## 🔄 Development Workflow

```bash
# 1. Explore data
jupyter notebook notebooks/EDA.ipynb

# 2. Modify configuration if needed
vim scripts/config.py

# 3. Train model
cd scripts
python train_pipeline.py

# 4. Review results
mlflow ui

# 5. Test inference
python inference.py --model ../models/best_model_xxx --text "Test item"

# 6. Deploy or iterate
```

## 📖 Documentation

Detailed documentation available in:
- `scripts/README.md`: Comprehensive guide to all modules
- Inline docstrings: Every function and class documented
- Type hints: Clear parameter and return types

## 🤝 Contributing

To extend the pipeline:

1. **Add new features**: Extend relevant modules in `scripts/`
2. **Add new models**: Update `model_name` in config
3. **Add preprocessing**: Modify `data_loader.py`
4. **Add metrics**: Extend `compute_metrics` in `model_trainer.py`

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face Transformers for model architecture
- MLflow for experiment tracking
- scikit-learn for evaluation metrics

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the ML Engineering team.

---

**Built with ❤️ using PyTorch and Transformers**
