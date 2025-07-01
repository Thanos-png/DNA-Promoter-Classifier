# DNA-Promoter-Classification

A deep learning–based classifier for DNA promoter sequence detection using the UCI Promoter Gene Sequences dataset. Implemented in PyTorch with one-hot encoding and a simple 1D CNN.

## Project Overview

### Features

- Loads and parses the UCI Promoter dataset  
- One-hot encodes DNA sequences (A, T, G, C)  
- Configurable 1D CNN for binary classification 

#### This project uses the **UCI Promoter Gene Sequences Dataset**, available at:
[UCI Promoter Gene Sequences dataset](https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences)

## Getting Started

### Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/Thanos-png/DNA-Promoter-Classifier.git
cd DNA-Promoter-Classifier
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
- You can choose if you want the trainning to be done with Cross-Validation `python train.py --method cv` or using the default train/test split method `python train.py`.
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
| Parameter    | Value  | Description                          |
| `epochs`     | `20`   | Number of epochs                     |
| `batch_size` | `16`   | Number of samples per batch          |
| `lr`         | `1e-3` | Learning rate                        |
| `test_size`  | `0.3`  | Test size for train/test split       |
| `k_folds`    | `5`    | Number of folds for cross-validation |


#### Expected output:
```
Training with Train/Test Split
Using device: cuda
Training samples: 84
Test samples: 22
Total samples: 106

Starting training for 20 epochs...
Epoch 1/20 - Train Loss: 0.6885 - Test Loss: 0.6765 - Test Acc: 0.7273
New best model saved! Accuracy: 0.7273
Epoch 2/20 - Train Loss: 0.6594 - Test Loss: 0.6548 - Test Acc: 0.6364
Epoch 3/20 - Train Loss: 0.6213 - Test Loss: 0.6289 - Test Acc: 0.8636
New best model saved! Accuracy: 0.8636
Epoch 4/20 - Train Loss: 0.5706 - Test Loss: 0.5997 - Test Acc: 0.9091
New best model saved! Accuracy: 0.9091
Epoch 5/20 - Train Loss: 0.5269 - Test Loss: 0.5620 - Test Acc: 0.8636
Epoch 6/20 - Train Loss: 0.4689 - Test Loss: 0.5216 - Test Acc: 0.8636
Epoch 7/20 - Train Loss: 0.4365 - Test Loss: 0.4816 - Test Acc: 0.9545
New best model saved! Accuracy: 0.9545
Epoch 8/20 - Train Loss: 0.3658 - Test Loss: 0.4555 - Test Acc: 0.8182
Epoch 9/20 - Train Loss: 0.3221 - Test Loss: 0.4026 - Test Acc: 0.9545
Epoch 10/20 - Train Loss: 0.2610 - Test Loss: 0.3664 - Test Acc: 0.9545
Epoch 11/20 - Train Loss: 0.2199 - Test Loss: 0.3322 - Test Acc: 0.9545
Epoch 12/20 - Train Loss: 0.1848 - Test Loss: 0.3098 - Test Acc: 0.9091
Epoch 13/20 - Train Loss: 0.1805 - Test Loss: 0.2943 - Test Acc: 0.9545
Epoch 14/20 - Train Loss: 0.1445 - Test Loss: 0.2943 - Test Acc: 0.9545
Epoch 15/20 - Train Loss: 0.1053 - Test Loss: 0.2540 - Test Acc: 0.9545
Epoch 16/20 - Train Loss: 0.1000 - Test Loss: 0.2430 - Test Acc: 0.9091
Epoch 17/20 - Train Loss: 0.0716 - Test Loss: 0.2630 - Test Acc: 0.9545
Epoch 18/20 - Train Loss: 0.0717 - Test Loss: 0.2196 - Test Acc: 0.9091
Epoch 19/20 - Train Loss: 0.0675 - Test Loss: 0.2101 - Test Acc: 0.9091
Epoch 20/20 - Train Loss: 0.0522 - Test Loss: 0.2060 - Test Acc: 0.9091

Training completed! Best Test Accuracy: 0.9545
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


#### The visualizations are stored in the `results/plots` directory.
