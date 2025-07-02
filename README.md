# DNA-Promoter-Classifier

A deep learning-based classifier for DNA promoter sequence detection using the UCI Promoter Gene Sequences dataset. Implemented in PyTorch with one-hot encoding and a simple 1D CNN.

## Project Overview

### Features

- Loads and parses the UCI Promoter dataset  
- One-hot encodes DNA sequences (A, T, G, C)  
- Configurable 1D CNN for binary classification 

#### This project uses the **UCI Promoter Gene Sequences Dataset**, available at:
[UCI Promoter Gene Sequences dataset](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences)

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
| Parameter    | Value  | Description                              |
| ------------ | ------ | ---------------------------------------- |
| `epochs`     | `20`   | Number of epochs                         |
| `batch_size` | `16`   | Number of samples per batch              |
| `lr`         | `1e-3` | Learning rate                            |
| `val_size`   | `0.25` | Validation size for train/val/test split |
| `test_size`  | `0.2`  | Test size for train/val/test split       |
| `k_folds`    | `5`    | Number of folds for cross-validation     |


#### Expected output:
```
Training with Train/Val/Test Split
Using device: cuda
Processed data not found in ../data/processed
Running preprocessing automatically...
Loaded 106 sequences
All sequences have the same length.
Processed data saved to ../data/processed
Files created:
  - train_dataset.pt
  - val_dataset.pt
  - test_dataset.pt
  - nucleotide_mapping.pkl
  - metadata.pkl
Preprocessing completed successfully!
Training samples: 63
Validation samples: 21
Test samples: 22
Total samples: 106

Starting training for 20 epochs...
Epoch 1/20 - Train Loss: 0.6877 - Val Loss: 0.6760 - Val Acc: 0.6667
New best model saved! Accuracy: 0.6667
Epoch 2/20 - Train Loss: 0.6711 - Val Loss: 0.6626 - Val Acc: 0.6190
Epoch 3/20 - Train Loss: 0.6485 - Val Loss: 0.6421 - Val Acc: 0.9048
New best model saved! Accuracy: 0.9048
Epoch 4/20 - Train Loss: 0.6245 - Val Loss: 0.6222 - Val Acc: 0.9048
Epoch 5/20 - Train Loss: 0.5792 - Val Loss: 0.5979 - Val Acc: 0.9048
Epoch 6/20 - Train Loss: 0.5601 - Val Loss: 0.5761 - Val Acc: 0.9048
Epoch 7/20 - Train Loss: 0.4843 - Val Loss: 0.5515 - Val Acc: 0.8571
Epoch 8/20 - Train Loss: 0.4565 - Val Loss: 0.5224 - Val Acc: 0.9048
Epoch 9/20 - Train Loss: 0.4489 - Val Loss: 0.4916 - Val Acc: 0.9048
Epoch 10/20 - Train Loss: 0.3775 - Val Loss: 0.4622 - Val Acc: 0.9524
New best model saved! Accuracy: 0.9524
Epoch 11/20 - Train Loss: 0.3350 - Val Loss: 0.4339 - Val Acc: 0.9524
Epoch 12/20 - Train Loss: 0.3042 - Val Loss: 0.4089 - Val Acc: 0.9048
Epoch 13/20 - Train Loss: 0.2371 - Val Loss: 0.3815 - Val Acc: 0.9524
Epoch 14/20 - Train Loss: 0.2194 - Val Loss: 0.3600 - Val Acc: 0.9524
Epoch 15/20 - Train Loss: 0.2177 - Val Loss: 0.3472 - Val Acc: 0.9048
Epoch 16/20 - Train Loss: 0.1779 - Val Loss: 0.3146 - Val Acc: 0.9524
Epoch 17/20 - Train Loss: 0.1377 - Val Loss: 0.2972 - Val Acc: 0.9524
Epoch 18/20 - Train Loss: 0.1283 - Val Loss: 0.3041 - Val Acc: 0.9048
Epoch 19/20 - Train Loss: 0.1012 - Val Loss: 0.2857 - Val Acc: 0.9048
Epoch 20/20 - Train Loss: 0.1236 - Val Loss: 0.2599 - Val Acc: 0.9524

Training completed! Best Accuracy: 0.9524
Best model saved as: ../results/models/promoter_cnn.pth
```

### Test the PromoterCNN Model
```bash
cd src/
python test.py
```

#### This will:
- Load the **saved CNN model** and **processed data**.

#### Expected output:
```

```

## Key Implementations
### Data Preprocessing
- Read `promoters.data`, parse labels (`+` → 1, `-` → 0) and sequences.
- One-hot encode each nucleotide into a 4-dim vector.
- Split into train/test sets (70/30 stratified).
- Wrap data in a PyTorch `Dataset` and create `DataLoaders`.

## Results & Analysis
### PromoterCNN metrics

#### The trained models are stored in the `results/models` directory.
#### The metrics visualizations are stored in the `results/plots` directory.

## Contact
For questions or feedback, feel free to reach me out:
* **Email:** thanos.panagiotidis@protonmail.com
