# DNA-Promoter-Classifier

A PyTorch-based deep learning classifier for DNA promoter sequence detection using the UCI Promoter Gene Sequences dataset. Implemented using a 1D Convolutional Neural Network (CNN).

## Project Overview

### Features

- Loads and parses the UCI Promoter dataset  
- One-hot encodes DNA sequences (A, T, G, C)  
- Configurable 1D CNN for binary classification 

#### Dataset
- **Source:** [UCI Promoter Gene Sequences Dataset](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences)
- **Size:** 106 sequences (57 nucleotides each)
- **Labels:** '+' for promoter, '-' for non-promoter

## Getting Started

### Installation
```bash
git clone https://github.com/Thanos-png/DNA-Promoter-Classifier.git
cd DNA-Promoter-Classifier
```

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

### Ensure GPU is Available
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `True` your GPU is ready.If `False` check your CUDA installation or the CPU will be used automatically.

## Running the Project
### Train the PromoterCNN Model
```bash
cd src/
python train.py
```

#### This will:
- You can choose if you want the trainning to be done with Cross-Validation `python train.py --method cv` or using the default train/val/test split method `python train.py`.
- First try to load preprocessed data from `../data/processed/`.
- If not found, automatically:
    - Load the raw data from the specified path
    - Preprocess it (one-hot encoding, train/test split)
    - Save the processed data for future use
    - Return the `DataLoaders` for training
- Loads processed data (`DataLoaders`).
- Instantiates `PromoterCNN` and `Adam` optimizer.
- Runs epochs of forward and backward passes with gradient updates.
- Logs training & validation loss/accuracy.
- Save the model and the preprocessed data.

#### Hyperparameters used:
| Parameter    | Value   | Description                              |
| ------------ | ------- | ---------------------------------------- |
| `epochs`     | `20`    | Number of epochs                         |
| `batch_size` | `16`    | Number of samples per batch              |
| `lr`         | `1e-3`  | Learning rate                            |
| `val_size`   | `0.25`  | Validation size for train/val/test split |
| `test_size`  | `0.2`   | Test size for train/val/test split       |
| `method`     | `split` | Training method (split/cv)               |
| `k_folds`    | `5`     | Number of folds for cross-validation     |


#### Expected output:
```
Training with Train/Val/Test Split
Using device: cuda
Training samples: 63
Validation samples: 21
Test samples: 22
Total samples: 106

Starting training for 20 epochs...
Epoch 1/20 - Train Loss: 0.6921 - Val Loss: 0.6780 - Val Acc: 0.5238
New best model saved! Accuracy: 0.5238
Epoch 2/20 - Train Loss: 0.6528 - Val Loss: 0.6701 - Val Acc: 0.5714
New best model saved! Accuracy: 0.5714
Epoch 3/20 - Train Loss: 0.6030 - Val Loss: 0.6455 - Val Acc: 0.7619
New best model saved! Accuracy: 0.7619
Epoch 4/20 - Train Loss: 0.5755 - Val Loss: 0.6222 - Val Acc: 0.7619
Epoch 5/20 - Train Loss: 0.5476 - Val Loss: 0.6024 - Val Acc: 0.7143
Epoch 6/20 - Train Loss: 0.5238 - Val Loss: 0.5835 - Val Acc: 0.8095
New best model saved! Accuracy: 0.8095
Epoch 7/20 - Train Loss: 0.4392 - Val Loss: 0.5630 - Val Acc: 0.7619
Epoch 8/20 - Train Loss: 0.4716 - Val Loss: 0.5417 - Val Acc: 0.8571
New best model saved! Accuracy: 0.8571
Epoch 9/20 - Train Loss: 0.3845 - Val Loss: 0.5203 - Val Acc: 0.6667
Epoch 10/20 - Train Loss: 0.3189 - Val Loss: 0.4981 - Val Acc: 0.7619
Epoch 11/20 - Train Loss: 0.2973 - Val Loss: 0.4771 - Val Acc: 0.7619
Epoch 12/20 - Train Loss: 0.2997 - Val Loss: 0.4538 - Val Acc: 0.8571
Epoch 13/20 - Train Loss: 0.2287 - Val Loss: 0.4377 - Val Acc: 0.8571
Epoch 14/20 - Train Loss: 0.2143 - Val Loss: 0.4283 - Val Acc: 0.7619
Epoch 15/20 - Train Loss: 0.1571 - Val Loss: 0.4202 - Val Acc: 0.7143
Epoch 16/20 - Train Loss: 0.1462 - Val Loss: 0.4052 - Val Acc: 0.7143
Epoch 17/20 - Train Loss: 0.1459 - Val Loss: 0.3834 - Val Acc: 0.8095
Epoch 18/20 - Train Loss: 0.1119 - Val Loss: 0.3832 - Val Acc: 0.7619
Epoch 19/20 - Train Loss: 0.1301 - Val Loss: 0.3732 - Val Acc: 0.8095
Epoch 20/20 - Train Loss: 0.0833 - Val Loss: 0.3718 - Val Acc: 0.7619

Training completed! Best Accuracy: 0.8571
Best model saved as: ../results/models/promoter_cnn.pth
```

### Test the PromoterCNN Model
```bash
cd src/
python test.py
```

#### This will:
- You can choose to test either the train/val/test split model `python test.py` or the cross-validation model `python test.py --method cv`.
- Load the **saved CNN model** from `../results/models/`.
- Load the **processed test data** or create it from raw data if not found.
- Evaluate the model on the test set using **BCEWithLogitsLoss**.
- Generate and save **comprehensive visualization plots**:
  - Confusion matrix heatmap with percentages
  - ROC curve with AUC score
  - Precision-Recall curve with AUC score
  - Prediction score distribution by class
  - Metrics summary bar chart
- Display detailed **classification metrics**:
  - Test loss and accuracy
  - Precision, recall, F1-score for each class
  - Confusion matrix breakdown
  - ROC-AUC and PR-AUC scores
- Save all plots to `../results/plots/` directory.

#### Expected output:
```
Testing Train/Val/Test split model
Using device: cuda
Test samples: 22
Loaded model from: ../results/models/promoter_cnn.pth

==================
 GENERATING PLOTS
==================
Confusion matrix saved: ../results/plots/confusion_matrix_train-val-test_split.png
ROC curve saved: ../results/plots/roc_curve_train-val-test_split.png
Precision-Recall curve saved: ../results/plots/precision_recall_train-val-test_split.png
Prediction distribution saved: ../results/plots/prediction_distribution_train-val-test_split.png
Metrics summary saved: ../results/plots/metrics_summary_train-val-test_split.png

===============================
 FINAL TEST EVALUATION RESULTS
===============================
Test Loss: 0.4982
Test Accuracy: 0.9545

Classification Report:
              precision    recall  f1-score   support

Non-Promoter       1.00      0.91      0.95        11
    Promoter       0.92      1.00      0.96        11

    accuracy                           0.95        22
   macro avg       0.96      0.95      0.95        22
weighted avg       0.96      0.95      0.95        22


Confusion Matrix:
True Negatives: 10, False Positives: 1
False Negatives: 0, True Positives: 11

Additional Metrics:
Accuracy: 0.9545
Precision: 0.9167
Recall: 1.0000
F1-Score: 0.9565
ROC-AUC: 0.9835
PR-AUC: 0.9834

Final Test Accuracy: 0.9545
```

## Key Implementations
### Data Preprocessing
- Read `promoters.data`, parse labels (`+` → 1, `-` → 0) and sequences.
- One-hot encode each nucleotide into a 4-dim vector.
- Split into train/val/test sets (60/20/20 stratified).
- Wrap data in a PyTorch `Dataset` and create `DataLoaders`.

### CNN Model
A 1D Convolutional Neural Network (CNN) implemented with PyTorch:

**Architecture:**
- **Input:** One-hot encoded DNA sequences (57×4)
- **Conv1D Layer:** 4→32 channels, kernel_size=5, ReLU activation
- **MaxPool1D:** kernel_size=2 for downsampling
- **Fully Connected:** 832→64 neurons, ReLU activation
- **Dropout:** 50% regularization
- **Output Layer:** 64→1 neuron (raw logits)
- **Loss Function:** BCEWithLogitsLoss (combines sigmoid + BCE)

**Parameters:** ~27K trainable parameters

## Results & Analysis

### Test Performance
- **Test Loss:** 0.4982
- **Test Accuracy:** 0.9545

### Classification Report
```
              precision    recall  f1-score   support

Non-Promoter       1.00      0.91      0.95        11
    Promoter       0.92      1.00      0.96        11

    accuracy                           0.95        22
   macro avg       0.96      0.95      0.95        22
weighted avg       0.96      0.95      0.95        22
```

### Confusion Matrix
```
True Negatives: 10
False Positives: 1
False Negatives: 0
True Positives: 11
```

### Additional Metrics
- **Accuracy:** 0.9545
- **Precision:** 0.9167
- **Recall:** 1.0000
- **F1-Score:** 0.9565
- **ROC-AUC:** 0.9835
- **PR-AUC:** 0.9834

### Model Variance
Due to the small dataset size (106 samples), model performance can vary between training runs:
- **Typical Range:** 70-95% test accuracy
- **Small Test Set:** Only 22 samples, so 1-2 misclassifications significantly impact accuracy

## Contact
For questions or feedback, feel free to reach me out:
* **Email:** thanos.panagiotidis@protonmail.com
