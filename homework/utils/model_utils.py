import torch
import torch.nn as nn
import time
import logging

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_fc_model(input_size: int, hidden_sizes: list, output_size: int,
                   dropout: float = 0.0, use_batchnorm: bool = False) -> nn.Module:
    """
    Build a fully-connected model.
    hidden_sizes: list of hidden layer sizes. Empty => linear classifier.
    """
    layers = []
    prev = input_size
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, output_size))
    return nn.Sequential(*layers)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item() * X.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += X.size(0)
    return total_loss / total, correct / total


def run_experiment(model, train_loader, test_loader, epochs: int = 20,
                   lr: float = 1e-3, weight_decay: float = 0.0, device: str = 'cpu'):
    """Train model and return history dict with train/test accuracy and timing."""
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    t0 = time.time()
    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        history['train_acc'].append(tr_acc)
        history['test_acc'].append(te_acc)
        history['train_loss'].append(tr_loss)
        history['test_loss'].append(te_loss)
        logger.debug(f'Epoch {epoch+1}/{epochs} | train_acc={tr_acc:.4f} | test_acc={te_acc:.4f}')
    history['train_time'] = time.time() - t0
    history['n_params'] = count_parameters(model)
    return history
