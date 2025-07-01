import argparse
import os
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_promoter_data, one_hot_encode, DNADataset, save_processed_data, mapping
from model import get_model


def get_data_loaders_from_processed(
    processed_dir: str = '../data/processed',
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Load preprocessed datasets and return DataLoaders.
    """
    if not os.path.exists(os.path.join(processed_dir, 'train_dataset.pt')):
        raise FileNotFoundError(f"Processed data not found in {processed_dir}. Run preprocess.py first.")

    train_dataset = torch.load(os.path.join(processed_dir, 'train_dataset.pt'), weights_only=False)
    test_dataset = torch.load(os.path.join(processed_dir, 'test_dataset.pt'), weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def get_raw_data(data_path: str):
    """
    Load and encode raw data without train/test split.
    Returns sequences and labels for cross-validation.
    """
    sequences, labels = load_promoter_data(data_path)
    encoded_seqs = one_hot_encode(sequences)
    return encoded_seqs, labels


def get_data_loaders(
    file_path: str,
    batch_size: int,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Load data, preprocess, and return train/test DataLoaders.
    Args:
        file_path: Path to the promoters.data file.
        batch_size: Batch size for DataLoaders.
        test_size: Fraction of data for testing.
        random_state: Seed for reproducibility.
    Returns:
        train_loader, test_loader
    """
    sequences, labels = load_promoter_data(file_path)
    encoded = one_hot_encode(sequences)
    X_train, X_test, y_train, y_test = train_test_split(
        encoded,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    train_dataset = DNADataset(X_train, y_train)
    test_dataset = DNADataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def auto_preprocess_and_get_loaders(
    data_path: str,
    processed_dir: str = '../data/processed',
    batch_size: int = 16,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Automatically preprocess data if not found, then return DataLoaders.
    """
    try:
        # Try to load preprocessed data first
        return get_data_loaders_from_processed(processed_dir, batch_size)
    except FileNotFoundError:
        print(f"Processed data not found in {processed_dir}")
        print("Running preprocessing automatically...")

        # Load and preprocess data
        sequences, labels = load_promoter_data(data_path)
        encoded_seqs = one_hot_encode(sequences)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_seqs,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        # Create datasets
        train_dataset = DNADataset(X_train, y_train)
        test_dataset = DNADataset(X_test, y_test)

        # Save processed data for future use
        save_processed_data(train_dataset, test_dataset, mapping, processed_dir)

        # Create and return DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("Preprocessing completed successfully!")
        return train_loader, test_loader


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer
) -> float:
    """
    Train the model for one epoch.
    Returns:
        Average training loss.
    """
    model.train()
    running_loss = 0.0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # output: (batch_size, 1), target: (batch_size,)
        loss = criterion(output.squeeze(), target.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    return running_loss / len(train_loader.dataset)


def evaluate(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module
) -> Tuple[float, float]:
    """
    Evaluate the model on the test set.
    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            running_loss += loss.item() * data.size(0)
            preds = (output.squeeze() >= 0.5).long()
            correct += preds.eq(target).sum().item()
    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy


def train_single_split(
    data_path: str,
    processed_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    model_path: str
):
    """
    Train model using single train/test split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Automatically handle preprocessing
    train_loader, test_loader = auto_preprocess_and_get_loaders(
        data_path, processed_dir, batch_size
    )

    # Print dataset sizes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Total samples: {len(train_loader.dataset) + len(test_loader.dataset)}")

    model = get_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Create models directory
    model_dir = "../results/models"
    os.makedirs(model_dir, exist_ok=True)
    full_model_path = os.path.join(model_dir, model_path)

    print(f"\nStarting training for {epochs} epochs...")
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)
        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Test Loss: {test_loss:.4f} - "
            f"Test Acc: {test_acc:.4f}"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), full_model_path)
            print(f"New best model saved! Accuracy: {best_acc:.4f}")

    print(f"\nTraining completed! Best Test Accuracy: {best_acc:.4f}")
    print(f"Best model saved as: {full_model_path}")


def train_with_cross_validation(
    data_path: str,
    k_folds: int = 5,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 16,
    model_path: str = "promoter_cnn_cv.pth"
):
    """
    Perform k-fold cross-validation for more robust evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load raw data for cross-validation
    sequences, labels = get_raw_data(data_path)
    print(f"Total samples for CV: {len(sequences)}")

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_accuracies = []

    # Initialize variables to track best model across all folds
    best_overall_acc = 0.0
    best_model_state = None

    # Create models directory
    model_dir = "../results/models"
    os.makedirs(model_dir, exist_ok=True)
    full_model_path = os.path.join(model_dir, model_path)

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

        # Split data
        X_train = sequences[train_idx]
        X_val = sequences[val_idx]
        y_train = [labels[i] for i in train_idx]
        y_val = [labels[i] for i in val_idx]

        # Create datasets and loaders
        train_dataset = DNADataset(X_train, y_train)
        val_dataset = DNADataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Train model
        model = get_model().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        best_val_acc = 0.0
        for epoch in range(epochs):
            train_loss = train(model, device, train_loader, criterion, optimizer)
            val_loss, val_acc = evaluate(model, device, val_loader, criterion)

            if epoch % 5 == 0 or epoch == epochs - 1:  # Print every 5 epochs
                print(f"  Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        fold_accuracies.append(best_val_acc)
        print(f"Fold {fold + 1} Best Accuracy: {best_val_acc:.4f}")

        # Save the best model across all folds
        if best_val_acc > best_overall_acc:
            best_overall_acc = best_val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best model found in fold {fold + 1}! Accuracy: {best_overall_acc:.4f}")

    # Save the best model
    if best_model_state is not None:
        torch.save(best_model_state, full_model_path)
        print(f"\nBest model saved as: {full_model_path}")
        print(f"Best model accuracy: {best_overall_acc:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n--- Cross-Validation Results ---")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")

    return fold_accuracies, mean_acc, std_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PromoterCNN model")
    parser.add_argument("--data_path", type=str, default="../data/molecular+biology+promoter+gene+sequences/promoters.data")
    parser.add_argument("--processed_dir", type=str, default="../data/processed")
    parser.add_argument("--method", type=str, choices=['split', 'cv'], default='split',
                        help="Training method: 'split' for train/test split, 'cv' for cross-validation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model_path", type=str, default="promoter_cnn.pth")
    parser.add_argument("--test_size", type=float, default=0.3, help="Test size for train/test split")
    parser.add_argument("--k_folds", type=int, default=5, help="Number of folds for cross-validation")
    args = parser.parse_args()

    if args.method == 'cv':
        print("Training with Cross-Validation")
        cv_model_path = args.model_path.replace('.pth', '_cv.pth')
        fold_accuracies, mean_acc, std_acc = train_with_cross_validation(
            args.data_path,
            k_folds=args.k_folds,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            model_path=cv_model_path
        )
    else:
        print("Training with Train/Test Split")
        train_single_split(
            args.data_path,
            args.processed_dir,
            args.batch_size,
            args.epochs,
            args.lr,
            args.model_path
        )


if __name__ == "__main__":
    main()
