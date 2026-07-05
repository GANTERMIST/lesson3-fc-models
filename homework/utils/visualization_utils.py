import matplotlib.pyplot as plt
import numpy as np
import os


def plot_learning_curves(histories: dict, title: str, save_path: str = None):
    """
    histories: {label: history_dict}
    Plots train/test accuracy curves for each model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, h in histories.items():
        axes[0].plot(h['train_acc'], label=f'{label} train')
        axes[0].plot(h['test_acc'], '--', label=f'{label} test')
        axes[1].plot(h['train_loss'], label=f'{label} train')
        axes[1].plot(h['test_loss'], '--', label=f'{label} test')
    for ax, ylabel in zip(axes, ['Accuracy', 'Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close()


def plot_bar_comparison(labels, values, ylabel: str, title: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color='steelblue', edgecolor='black')
    ax.bar_label(bars, fmt='%.4f')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close()


def plot_heatmap(matrix: np.ndarray, row_labels, col_labels,
                 title: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Test Accuracy')
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Hidden size')
    ax.set_ylabel('Depth')
    ax.set_title(title)
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close()


def plot_weight_distribution(model, title: str, save_path: str = None):
    import torch
    weights = []
    for name, p in model.named_parameters():
        if 'weight' in name:
            weights.extend(p.detach().cpu().numpy().flatten().tolist())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(weights, bins=100, color='steelblue', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Weight value')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close()
