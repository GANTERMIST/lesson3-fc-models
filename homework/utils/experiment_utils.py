import logging
import json
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def get_digits_loaders(batch_size: int = 64, test_size: float = 0.2, seed: int = 42):
    """Load sklearn digits dataset as PyTorch DataLoaders."""
    digits = load_digits()
    X, y = digits.data.astype(np.float32), digits.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    input_size = X.shape[1]   # 64
    output_size = len(np.unique(y))  # 10
    logger.info(f'Digits dataset: train={len(train_ds)}, test={len(test_ds)}, '
                f'input={input_size}, classes={output_size}')
    return train_loader, test_loader, input_size, output_size


def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert non-serializable to list
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj
    with open(path, 'w') as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    logger.info(f'Results saved to {path}')
