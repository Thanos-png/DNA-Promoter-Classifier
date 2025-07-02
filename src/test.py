import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_promoter_data, one_hot_encode, DNADataset
from model import get_model
from train import get_data_loaders_from_processed, auto_preprocess_and_get_loaders, evaluate


def create_plots_directory():
    """Create plots directory if it doesn't exist."""
    plots_dir = "../results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_confusion_matrix(y_true, y_pred, model_name, plots_dir):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Promoter', 'Promoter'],
                yticklabels=['Non-Promoter', 'Promoter'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Add percentage annotations
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm[i,j]/total:.1%})', 
                    ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    save_path = os.path.join(plots_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")

    return save_path


def plot_roc_curve(y_true, y_scores, model_name, plots_dir):
    """Create and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {save_path}")

    return save_path, roc_auc


def plot_precision_recall_curve(y_true, y_scores, model_name, plots_dir):
    """Create and save Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AUC = {pr_auc:.3f})')

    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--', 
                label=f'Random Classifier (AP = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f'precision_recall_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision-Recall curve saved: {save_path}")

    return save_path, pr_auc


def plot_prediction_distribution(y_true, y_scores, model_name, plots_dir):
    """Create and save prediction score distribution plot."""
    plt.figure(figsize=(10, 6))

    # Convert to numpy arrays first
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Separate scores by true class
    non_promoter_scores = y_scores[np.array(y_true) == 0]
    promoter_scores = y_scores[np.array(y_true) == 1]

    plt.hist(non_promoter_scores, bins=30, alpha=0.7, label='Non-Promoter', 
            color='red', density=True)
    plt.hist(promoter_scores, bins=30, alpha=0.7, label='Promoter', 
            color='blue', density=True)

    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, 
                label='Decision Threshold (0.5)')

    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title(f'Prediction Score Distribution - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f'prediction_distribution_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Prediction distribution saved: {save_path}")

    return save_path


def plot_metrics_summary(metrics_dict, model_name, plots_dir):
    """Create and save a metrics summary plot."""
    plt.figure(figsize=(10, 6))

    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']

    bars = plt.bar(metrics, values, color=colors[:len(metrics)])

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    plt.title(f'Model Performance Metrics - {model_name}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path = os.path.join(plots_dir, f'metrics_summary_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Metrics summary saved: {save_path}")

    return save_path


def detailed_evaluation_with_plots(model, device, test_loader, criterion, model_name, plots_dir):
    """
    Perform detailed evaluation with metrics and plots.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_scores = []  # For ROC and PR curves

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            running_loss += loss.item() * data.size(0)

            # Get predictions and scores
            scores = torch.sigmoid(output.squeeze())  # Convert to probabilities
            preds = (scores >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())

    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    # Generate all plots
    print("\n" + "="*18)
    print(" GENERATING PLOTS")
    print("="*18)

    # 1. Confusion Matrix
    plot_confusion_matrix(all_targets, all_preds, model_name, plots_dir)

    # 2. ROC Curve
    _, roc_auc = plot_roc_curve(all_targets, all_scores, model_name, plots_dir)

    # 3. Precision-Recall Curve
    _, pr_auc = plot_precision_recall_curve(all_targets, all_scores, model_name, plots_dir)

    # 4. Prediction Distribution
    plot_prediction_distribution(all_targets, all_scores, model_name, plots_dir)

    # 5. Calculate additional metrics for summary
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    # 6. Metrics Summary Plot
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc
    }
    plot_metrics_summary(metrics_dict, model_name, plots_dir)

    return avg_loss, accuracy, all_preds, all_targets, metrics_dict


def load_test_data(processed_dir: str, batch_size: int = 16):
    """
    Load test data from processed files.
    """
    try:
        _, _, test_loader = get_data_loaders_from_processed(processed_dir, batch_size)
        return test_loader
    except FileNotFoundError:
        print(f"Processed data not found in {processed_dir}")
        print("Please run training first to generate test data.")
        return None


def detailed_evaluation(model, device, test_loader, criterion):
    """
    Perform detailed evaluation with metrics.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.squeeze(), target.float())
            running_loss += loss.item() * data.size(0)

            # Get predictions
            preds = (output.squeeze() >= 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = running_loss / len(test_loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))

    return avg_loss, accuracy, all_preds, all_targets


def test_model(
    model_path: str,
    processed_dir: str = '../data/processed',
    batch_size: int = 16,
    data_path: str = None,
    val_size: float = 0.25,
    test_size: float = 0.2
):
    """
    Test a trained model on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create plots directory
    plots_dir = create_plots_directory()

    # Load test data
    if data_path and not os.path.exists(os.path.join(processed_dir, 'test_dataset.pt')):
        print("Processed test data not found. Creating from raw data...")
        _, _, test_loader = auto_preprocess_and_get_loaders(
            data_path, processed_dir, batch_size, val_size, test_size
        )
    else:
        test_loader = load_test_data(processed_dir, batch_size)
        if test_loader is None:
            return None

    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    full_model_path = os.path.join("../results/models", model_path)
    if not os.path.exists(full_model_path):
        print(f"Model not found: {full_model_path}")
        return None

    model = get_model().to(device)
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    print(f"Loaded model from: {full_model_path}")

    # Evaluate with plots
    criterion = nn.BCEWithLogitsLoss()
    model_name = "Train-Val-Test Split"
    test_loss, test_acc, all_preds, all_targets, metrics_dict = detailed_evaluation_with_plots(
        model, device, test_loader, criterion, model_name, plots_dir
    )

    # Print results
    print("\n" + "="*31)
    print(" FINAL TEST EVALUATION RESULTS")
    print("="*31)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Non-Promoter', 'Promoter']))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(f"True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

    # Print additional metrics
    print(f"\nAdditional Metrics:")
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

    return test_acc


def test_cross_validation_model(
    data_path: str,
    model_path: str = "promoter_cnn_cv.pth",
    batch_size: int = 16,
    random_state: int = 79
):
    """
    Test a cross-validation trained model on a holdout test set.
    """
    from sklearn.model_selection import train_test_split
    from train import get_raw_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load raw data and create a holdout test set
    sequences, labels = get_raw_data(data_path)

    # Create holdout test set (20% of data)
    _, X_test, _, y_test = train_test_split(
        sequences, labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels
    )

    # Create test dataset and loader
    test_dataset = DNADataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Holdout test samples: {len(test_dataset)}")

    # Load and test model
    full_model_path = os.path.join("../results/models", model_path)
    if not os.path.exists(full_model_path):
        print(f"Model not found: {full_model_path}")
        return None

    model = get_model().to(device)
    model.load_state_dict(torch.load(full_model_path, map_location=device))
    print(f"Loaded CV model from: {full_model_path}")

    # Evaluate
    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc, all_preds, all_targets = detailed_evaluation(
        model, device, test_loader, criterion
    )

    # Print results
    print("\n" + "="*37)
    print(" CROSS-VALIDATION MODEL TEST RESULTS")
    print("="*37)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=['Non-Promoter', 'Promoter']))

    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Test PromoterCNN model")
    parser.add_argument("--model_path", type=str, default="promoter_cnn.pth",
                        help="Path to the trained model file (e.g., promoter_cnn.pth)")
    parser.add_argument("--processed_dir", type=str, default="../data/processed",
                        help="Directory with processed data")
    parser.add_argument("--data_path", type=str, 
                        default="../data/molecular+biology+promoter+gene+sequences/promoters.data",
                        help="Path to raw data (used if processed data not found)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--method", type=str, choices=['split', 'cv'], default='split',
                        help="Test method: 'split' for train/val/test split, 'cv' for cross-validation model")
    parser.add_argument("--val_size", type=float, default=0.25)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    if args.method == 'cv':
        print("Testing Cross-Validation trained model")

        # Ensure model path ends with '_cv.pth'
        if not args.model_path.endswith('_cv.pth'):
            cv_model_path = args.model_path.replace('.pth', '_cv.pth')
        else:
            cv_model_path = args.model_path

        # Test the cross-validation model
        test_acc = test_cross_validation_model(
            args.data_path,
            cv_model_path,
            args.batch_size
        )
    else:
        print("Testing Train/Val/Test split model")
        test_acc = test_model(
            args.model_path,
            args.processed_dir,
            args.batch_size,
            args.data_path,
            args.val_size,
            args.test_size
        )

    if test_acc is not None:
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
