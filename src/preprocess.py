import numpy as np
import torch
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from typing import List, Tuple, Dict

# One-hot encoding map for nucleotides
MappingType = Dict[str, List[int]]
mapping: MappingType = {
    'A': [1, 0, 0, 0],
    'T': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'C': [0, 0, 0, 1],
}


def load_promoter_data(file_path: str) -> Tuple[List[str], List[int]]:
    """
    Load sequences and labels from the UCI Promoter dataset.
    Returns:
        sequences: list of str
        labels: list of int (1 for '+' promoter, 0 for '-' non-promoter)
    """
    sequences: List[str] = []
    labels: List[int] = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split by comma and take first and last parts
            parts = line.split(',')
            if len(parts) >= 3:
                label_str = parts[0].strip()
                # The sequence is in the last part, after tabs
                seq_part = parts[-1].strip()
                # Remove any tabs and extract only valid nucleotides
                seq = ''.join(c.upper() for c in seq_part if c.upper() in 'ATGC')

                if seq:  # Only add if we have a valid sequence
                    labels.append(1 if label_str == '+' else 0)
                    sequences.append(seq)

    print(f"Loaded {len(sequences)} sequences")
    if sequences:
        seq_lengths = [len(seq) for seq in sequences]
        if len(set(seq_lengths)) > 1:
            print("Warning: Sequences have different lengths!")
        else:
            print("All sequences have the same length.")
    return sequences, labels


def one_hot_encode(seqs: List[str]) -> np.ndarray:
    """
    Convert a list of DNA sequences to one-hot encoded numpy arrays.
    Returns:
        numpy array of shape (n_samples, seq_length, 4)
    """
    # each seq → list of [A,T,G,C] one-hots
    encoded: np.ndarray = np.array(
        [[mapping[nt] for nt in seq] for seq in seqs],
        dtype=np.float32
    )
    return encoded


class DNADataset(Dataset):
    """
    PyTorch Dataset for DNA sequences.
    """
    def __init__(self, sequences: np.ndarray, labels: List[int]) -> None:
        """
        sequences: numpy array, shape (n_samples, seq_len, 4)
        labels: list of 0/1 ints, length n_samples
        """
        super().__init__()
        # convert to torch.Tensor
        self.sequences: torch.Tensor = torch.tensor(sequences)
        self.labels: torch.Tensor = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def save_processed_data(train_dataset: DNADataset, test_dataset: DNADataset, 
                       mapping: Dict, output_dir: str = '../data/processed') -> None:
    """
    Save processed datasets for later use.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    torch.save(train_dataset, os.path.join(output_dir, 'train_dataset.pt'))
    torch.save(test_dataset, os.path.join(output_dir, 'test_dataset.pt'))

    # Save processed data/mapping
    with open(os.path.join(output_dir, 'nucleotide_mapping.pkl'), 'wb') as f:
        pickle.dump(mapping, f)

    # Save metadata
    metadata = {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'sequence_length': train_dataset.sequences.shape[1],
        'num_features': train_dataset.sequences.shape[2]
    }

    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Processed data saved to {output_dir}")
    print(f"Files created:")
    print(f"  - train_dataset.pt")
    print(f"  - test_dataset.pt") 
    print(f"  - nucleotide_mapping.pkl")
    print(f"  - metadata.pkl")


if __name__ == "__main__":
    # Load and preprocess
    file_path: str = '../data/molecular+biology+promoter+gene+sequences/promoters.data'
    sequences, labels = load_promoter_data(file_path)
    encoded_seqs = one_hot_encode(sequences)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        encoded_seqs,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )

    # Wrap in Datasets & DataLoaders
    train_dataset = DNADataset(X_train, y_train)
    test_dataset  = DNADataset(X_test,  y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=16, shuffle=False)

    print(f"\nNumber of training samples: {len(train_dataset)}")
    print(f"Number of testing  samples: {len(test_dataset)}\n")

    # Save processed data
    save_processed_data(train_dataset, test_dataset, mapping)
