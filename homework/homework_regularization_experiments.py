"""
Задание 3: Эксперименты с регуляризацией полносвязных сетей.

Запуск:
    python homework/homework_regularization_experiments.py
"""
import logging
import os
import sys
import copy
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from homework.utils.experiment_utils import get_digits_loaders, save_results
from homework.utils.model_utils import build_fc_model, run_experiment
from homework.utils.visualization_utils import (
    plot_learning_curves, plot_bar_comparison, plot_weight_distribution
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

EPOCHS = 30
DEVICE = 'cpu'
HIDDEN = [256, 128, 64]  # фиксированная архитектура для всех экспериментов


def experiment_3_1_regularization_comparison(
        train_loader, test_loader, input_size, output_size):
    """3.1 Сравнение техник регуляризации."""
    logger.info('=== 3.1: Regularization comparison ===')
    configs = {
        'No reg': dict(dropout=0.0, use_batchnorm=False, weight_decay=0.0),
        'Dropout 0.1': dict(dropout=0.1, use_batchnorm=False, weight_decay=0.0),
        'Dropout 0.3': dict(dropout=0.3, use_batchnorm=False, weight_decay=0.0),
        'Dropout 0.5': dict(dropout=0.5, use_batchnorm=False, weight_decay=0.0),
        'BatchNorm': dict(dropout=0.0, use_batchnorm=True, weight_decay=0.0),
        'Drop0.3+BN': dict(dropout=0.3, use_batchnorm=True, weight_decay=0.0),
        'L2 1e-4': dict(dropout=0.0, use_batchnorm=False, weight_decay=1e-4),
        'L2 1e-3': dict(dropout=0.0, use_batchnorm=False, weight_decay=1e-3),
    }

    histories = {}
    models = {}
    for name, cfg in configs.items():
        wd = cfg.pop('weight_decay')
        model = build_fc_model(input_size, HIDDEN, output_size, **cfg)
        h = run_experiment(model, train_loader, test_loader,
                           epochs=EPOCHS, weight_decay=wd, device=DEVICE)
        histories[name] = h
        models[name] = model
        gap = h['train_acc'][-1] - h['test_acc'][-1]
        logger.info(f'{name}: train={h["train_acc"][-1]:.4f}, '
                    f'test={h["test_acc"][-1]:.4f}, gap={gap:.4f}')

    plot_learning_curves(
        histories, title='3.1 Regularization Techniques — Learning Curves',
        save_path='homework/plots/reg_learning_curves.png'
    )
    labels = list(histories.keys())
    test_accs = [h['test_acc'][-1] for h in histories.values()]
    plot_bar_comparison(labels, test_accs, 'Test Accuracy',
                        '3.1 Final Test Accuracy by Regularization',
                        save_path='homework/plots/reg_test_accuracy.png')

    # Визуализация распределения весов для базовой и лучшей модели
    plot_weight_distribution(models['No reg'], 'Weight Distribution — No Regularization',
                             save_path='homework/plots/reg_weights_no_reg.png')
    best_name = max(histories, key=lambda k: histories[k]['test_acc'][-1])
    plot_weight_distribution(models[best_name],
                             f'Weight Distribution — {best_name}',
                             save_path='homework/plots/reg_weights_best.png')
    return histories


class AdaptiveDropoutFC(nn.Module):
    """
    Полносвязная сеть с Dropout, чей коэффициент линейно уменьшается
    от start_p до end_p за total_epochs эпох.
    """
    def __init__(self, input_size, hidden_sizes, output_size,
                 start_p: float = 0.5, end_p: float = 0.1):
        super().__init__()
        self.start_p = start_p
        self.end_p = end_p
        self.dropout = nn.Dropout(p=start_p)
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(prev, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.dropout(x)
        return self.out(x)

    def set_dropout_p(self, p: float):
        self.dropout.p = p


def experiment_3_2_adaptive_regularization(
        train_loader, test_loader, input_size, output_size):
    """3.2 Адаптивная регуляризация."""
    logger.info('=== 3.2: Adaptive regularization ===')

    device = torch.device(DEVICE)
    model = AdaptiveDropoutFC(input_size, HIDDEN, output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs, dropout_ps = [], [], []
    for epoch in range(EPOCHS):
        # Линейный schedule: p убывает от 0.5 до 0.1
        p = 0.5 - (0.5 - 0.1) * epoch / max(EPOCHS - 1, 1)
        model.set_dropout_p(p)
        dropout_ps.append(p)

        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()
            total += X.size(0)
        train_accs.append(correct / total)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                correct += (out.argmax(1) == y).sum().item()
                total += X.size(0)
        test_accs.append(correct / total)
        logger.debug(f'Epoch {epoch+1}: p={p:.3f}, '
                     f'train={train_accs[-1]:.4f}, test={test_accs[-1]:.4f}')

    logger.info(f'Adaptive dropout final: train={train_accs[-1]:.4f}, '
                f'test={test_accs[-1]:.4f}')

    # --- BN с разными momentum ---
    bn_histories = {}
    for momentum in [0.01, 0.1, 0.5, 0.9]:
        # Строим модель с BN вручную (задаём momentum)
        layers = []
        prev = input_size
        for h in HIDDEN:
            layers += [nn.Linear(prev, h),
                       nn.BatchNorm1d(h, momentum=momentum),
                       nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        m = nn.Sequential(*layers)
        h = run_experiment(m, train_loader, test_loader, epochs=EPOCHS, device=DEVICE)
        key = f'BN momentum={momentum}'
        bn_histories[key] = h
        logger.info(f'{key}: test_acc={h["test_acc"][-1]:.4f}')

    plot_learning_curves(
        bn_histories,
        title='3.2 BatchNorm with Different Momentum Values',
        save_path='homework/plots/reg_bn_momentum.png'
    )
    return {
        'adaptive_dropout': {'train': train_accs, 'test': test_accs, 'dropout_p': dropout_ps},
        'bn_momentum': {k: {'final_test': v['test_acc'][-1]} for k, v in bn_histories.items()}
    }


def main():
    train_loader, test_loader, input_size, output_size = get_digits_loaders()

    h1 = experiment_3_1_regularization_comparison(
        train_loader, test_loader, input_size, output_size)
    h2 = experiment_3_2_adaptive_regularization(
        train_loader, test_loader, input_size, output_size)

    all_results = {
        '3_1_regularization': {
            name: {'final_train': h['train_acc'][-1], 'final_test': h['test_acc'][-1],
                   'gap': h['train_acc'][-1] - h['test_acc'][-1]}
            for name, h in h1.items()
        },
        '3_2_adaptive': h2
    }
    save_results(all_results, 'homework/results/regularization_experiments/results.json')
    logger.info('Task 3 complete.')


if __name__ == '__main__':
    main()
