# Phishing URL Detection Project

A transformer-based machine learning project for detecting phishing URLs using DistilBERT.

## Project Overview

This project implements a phishing URL detection system using a fine-tuned DistilBERT model. The system can classify URLs as either phishing (malicious) or legitimate with high accuracy.

## Features

- **Efficient Architecture**: Uses DistilBERT for 40% smaller model size and 60% faster inference
- **Robust Training**: Includes early stopping, learning rate scheduling, and stratified data splitting
- **Comprehensive Evaluation**: Reports accuracy, F1-score, and AUC metrics
- **Production Ready**: Modular code structure with clear separation of concerns
- **Auto-Generated Outputs**: Output folder is automatically created during training

## Project Structure

```
phishing_project/
├── src/
│   ├── data/
│   │   ├── Train.csv          # Your training data goes here
│   │   └── Test.csv           # Your test data goes here
│   ├── data_utils.py          # Data loading and preprocessing
│   ├── model_utils.py         # Model architecture and setup
│   ├── train.py               # Training script
│   ├── predict.py             # Prediction script
│   └── outputs/               # Auto-created during training
│       ├── model.pt           # Trained model (Auto-created after training)
│       ├── tokenizer/         # Saved tokenizer (Auto-created after training)
│       ├── training_history.csv      # Training metrics (Auto-created after training)
│       ├── test_predictions.csv      # Basic predictions (Auto-created after predictions)
│       └── test_predictions_detailed.csv  # Detailed predictions (Auto-created after predictions)
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place your training data as `src/data/Train.csv`
   - Place your test data as `src/data/Test.csv`

### Data Format Requirements

**Training Data (`src/data/Train.csv`)**:
```csv
url,label
https://www.google.com,0
http://phishing-site.com,1
https://www.github.com,0
```
- Column 1: `url` (the URL string)
- Column 2: `label` (0 for legitimate, 1 for phishing)

**Test Data (`src/data/Test.csv`)**:
```csv
url
https://test-url1.com
https://test-url2.com
```
- Column 1: `url` (the URL string to classify)
- Optional: `label` or `true_label` column for evaluation

## Usage Instructions

### Step 1: Navigate to Source Directory

**IMPORTANT**: Always run commands from the `src` directory:

```bash
cd src
```

### Step 2: Train the Model

```bash
python train.py
```

**What happens during training:**
- Loads and preprocesses training data from `data/Train.csv`
- Splits into train/validation sets (80/20 split)
- Trains the DistilBERT-based classifier
- **Automatically creates** `outputs/` folder if it doesn't exist
- Saves the best model to `outputs/model.pt`
- Saves tokenizer to `outputs/tokenizer/`
- Logs training metrics to `outputs/training_history.csv`

**Expected output:**
```
Using device: cpu
Loading and preprocessing data...
Training samples: 800
Validation samples: 200
Setting up model and tokenizer...
Starting training...

Epoch 1/5
Training: 100%|██████████| 50/50 [02:30<00:00]
Evaluating: 100%|██████████| 13/13 [00:15<00:00]
Train Loss: 0.6543
Val Loss: 0.4321
Accuracy: 0.8950
F1 Score: 0.8876
AUC: 0.9234
New best model saved! AUC: 0.9234
...
```

### Step 3: Generate Predictions

```bash
python predict.py
```

**What happens during prediction:**
- Loads the trained model from `outputs/model.pt`
- Loads test data from `data/Test.csv`
- Generates predictions for all test URLs
- Saves results to `outputs/`

**Expected output:**
```
Using device: cpu
Loading trained model...
Model loaded successfully!
Test samples: 100
Generating predictions: 100%|██████████| 7/7 [00:05<00:00]
Basic predictions saved to: outputs/test_predictions.csv
Detailed predictions saved to: outputs/test_predictions_detailed.csv

Prediction Summary:
Total URLs: 100
Predicted Phishing: 23 (23.00%)
Predicted Legitimate: 77 (77.00%)
```

## Output Files (Auto-created in `outputs/`)

The `outputs/` folder is automatically created when you run `python train.py` for the first time.

### 1. **`test_predictions.csv`** (Basic format)
```csv
url,prediction
https://www.google.com,0
http://phishing-site.com,1
https://www.github.com,0
```

### 2. **`test_predictions_detailed.csv`** (Detailed format)
```csv
url,predicted_probability,predicted_label,true_label
https://www.google.com,0.05,0,0
http://phishing-site.com,0.95,1,1
https://www.github.com,0.02,0,0
```

### 3. **`training_history.csv`** (Training metrics)
```csv
epoch,train_loss,val_loss,accuracy,f1_score,auc
1,0.6543,0.4321,0.8950,0.8876,0.9234
2,0.4123,0.3876,0.9100,0.9045,0.9456
```

### 4. **`model.pt`** (Trained model checkpoint)
### 5. **`tokenizer/`** (Saved tokenizer directory)

## Path Configuration (Optional)

The code is designed to work automatically with relative paths. However, if you need to modify paths:

### Option 1: Default Setup (Recommended)
Just ensure you're running from the correct directory:
```bash
cd src
python train.py
python predict.py
```

### Option 2: Custom Paths (Optional)
If you need to specify custom paths, edit these files:

**In `train.py`**, modify the CONFIG section:
```python
CONFIG = {
    'train_path': 'path/to/your/Train.csv',
    'test_path': 'path/to/your/Test.csv',
    # ... other config options
}
```

**In `predict.py`**, modify the paths at the top of main():
```python
def main():
    model_path = 'path/to/your/model.pt'
    test_data_path = 'path/to/your/Test.csv'
    output_path = 'path/to/your/output.csv'
```

## Model Configuration

Key parameters (configurable in `train.py`):

- **Model**: DistilBERT-base-uncased
- **Max Sequence Length**: 128 tokens
- **Batch Size**: 16
- **Learning Rate**: 2e-5
- **Epochs**: 5 (with early stopping)
- **Optimizer**: AdamW with linear warmup

## Performance

The model typically achieves:
- **Accuracy**: >95% on validation data
- **F1-Score**: >95% on validation data
- **AUC**: >98% on validation data
- **Training Time**: ~10-15 minutes on CPU (depends on data size)

## Troubleshooting

### Common Issues

1. **"No such file or directory" errors**:
   - **Solution**: Make sure you're running commands from the `src/` directory
   - Check: `pwd` should show `.../phishing_project/src`

2. **Data files not found**:
   - **Solution**: Ensure your CSV files are in `src/data/` folder
   - Check: `ls data/` should show `Train.csv` and `Test.csv`

3. **CUDA out of memory**: 
   - **Solution**: Reduce `batch_size` in CONFIG (try 8 or 4)

4. **Slow training**: 
   - **Solution**: Consider using GPU if available, or reduce `max_length`

5. **Import errors**: 
   - **Solution**: Install dependencies: `pip install -r requirements.txt`

6. **Model not found when predicting**:
   - **Solution**: Run `python train.py` first to create the model
   - Check: `ls outputs/` should show `model.pt`

### Directory Structure Check

Your working directory should look like this:
```bash
# When you run 'pwd' from src folder, you should see:
/path/to/your/phishing_project/src

# When you run 'ls' from src folder, you should see:
data/  data_utils.py  model_utils.py  train.py  predict.py
outputs/  # (created after running train.py)
```

## Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Place `Train.csv` in `src/data/` folder
- [ ] Place `Test.csv` in `src/data/` folder
- [ ] Navigate to source directory: `cd src`
- [ ] Run training: `python train.py` (creates `outputs/` folder automatically)
- [ ] Run predictions: `python predict.py`
- [ ] Check results in `outputs/` folder

## Complete Workflow Example

```bash
# 1. Install dependencies (run from project root)
pip install -r requirements.txt

# 2. Navigate to source directory
cd src

# 3. Verify your data files are in place
ls data/
# Should show: Train.csv  Test.csv

# 4. Run training (creates outputs folder automatically)
python train.py

# 5. Verify training completed successfully
ls outputs/
# Should show: model.pt  tokenizer/  training_history.csv

# 6. Run predictions
python predict.py

# 7. Check your results
ls outputs/
# Should now also show: test_predictions.csv  test_predictions_detailed.csv
```

## Important Notes

- **Directory Requirement**: Always run `python train.py` and `python predict.py` from the `src/` directory
- **Auto-Creation**: The `outputs/` folder is automatically created during training - no manual setup needed
- **Data Location**: Your CSV files should be in `src/data/` folder
- **No Path Editing**: The code is pre-configured to work with the correct relative paths
- **Sequential Execution**: Always run training before predictions

---

**Remember**: The key to success is running the commands from the correct directory (`src/`). Everything else is handled automatically!