"""
Задание 1: Эксперименты с глубиной полносвязных сетей.

Запуск:
    python homework/homework_depth_experiments.py
"""
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from homework.utils.experiment_utils import get_digits_loaders, save_results
from homework.utils.model_utils import build_fc_model, run_experiment, count_parameters
from homework.utils.visualization_utils import plot_learning_curves, plot_bar_comparison

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

EPOCHS = 30
BASE_WIDTH = 128
DEVICE = 'cpu'

DEPTH_CONFIGS = {
    '1 layer (linear)': [],
    '2 layers (1 hidden)': [BASE_WIDTH],
    '3 layers (2 hidden)': [BASE_WIDTH, BASE_WIDTH],
    '5 layers (4 hidden)': [BASE_WIDTH] * 4,
    '7 layers (6 hidden)': [BASE_WIDTH] * 6,
}


def experiment_1_1_depth_comparison(train_loader, test_loader, input_size, output_size):
    """1.1 Сравнение моделей разной глубины."""
    logger.info('=== 1.1: Depth comparison ===')
    histories = {}
    for name, hidden in DEPTH_CONFIGS.items():
        model = build_fc_model(input_size, hidden, output_size)
        h = run_experiment(model, train_loader, test_loader,
                           epochs=EPOCHS, device=DEVICE)
        h['n_hidden'] = len(hidden)
        histories[name] = h
        logger.info(f'{name}: test_acc={h["test_acc"][-1]:.4f}, '
                    f'params={count_parameters(model)}, '
                    f'time={h["train_time"]:.1f}s')

    plot_learning_curves(
        histories, title='1.1 Depth Comparison — Learning Curves',
        save_path='homework/plots/depth_learning_curves.png'
    )
    labels = list(histories.keys())
    test_accs = [h['test_acc'][-1] for h in histories.values()]
    train_times = [h['train_time'] for h in histories.values()]
    plot_bar_comparison(labels, test_accs, 'Test Accuracy', '1.1 Final Test Accuracy by Depth',
                        save_path='homework/plots/depth_test_accuracy.png')
    plot_bar_comparison(labels, train_times, 'Train Time (s)', '1.1 Training Time by Depth',
                        save_path='homework/plots/depth_train_time.png')
    return histories


def experiment_1_2_overfitting_analysis(train_loader, test_loader, input_size, output_size):
    """1.2 Анализ переобучения: базовый vs Dropout vs BatchNorm vs Dropout+BN."""
    logger.info('=== 1.2: Overfitting analysis ===')
    HIDDEN = [BASE_WIDTH] * 4  # 5-layer model prone to overfit

    configs = {
        'No regularization': dict(dropout=0.0, use_batchnorm=False),
        'Dropout 0.3': dict(dropout=0.3, use_batchnorm=False),
        'BatchNorm': dict(dropout=0.0, use_batchnorm=True),
        'Dropout + BN': dict(dropout=0.3, use_batchnorm=True),
    }
    histories = {}
    for name, cfg in configs.items():
        model = build_fc_model(input_size, HIDDEN, output_size, **cfg)
        h = run_experiment(model, train_loader, test_loader,
                           epochs=EPOCHS, device=DEVICE)
        histories[name] = h
        gap = h['train_acc'][-1] - h['test_acc'][-1]
        logger.info(f'{name}: train={h["train_acc"][-1]:.4f}, '
                    f'test={h["test_acc"][-1]:.4f}, gap={gap:.4f}')

    plot_learning_curves(
        histories, title='1.2 Overfitting Analysis — Regularization Effect',
        save_path='homework/plots/depth_overfitting.png'
    )
    return histories


def main():
    train_loader, test_loader, input_size, output_size = get_digits_loaders()

    h1 = experiment_1_1_depth_comparison(train_loader, test_loader, input_size, output_size)
    h2 = experiment_1_2_overfitting_analysis(train_loader, test_loader, input_size, output_size)

    all_results = {
        '1_1_depth_comparison': {
            name: {'final_train': h['train_acc'][-1], 'final_test': h['test_acc'][-1],
                   'train_time': h['train_time'], 'n_params': h['n_params']}
            for name, h in h1.items()
        },
        '1_2_overfitting': {
            name: {'final_train': h['train_acc'][-1], 'final_test': h['test_acc'][-1],
                   'gap': h['train_acc'][-1] - h['test_acc'][-1]}
            for name, h in h2.items()
        }
    }
    save_results(all_results, 'homework/results/depth_experiments/results.json')
    logger.info('Task 1 complete.')


if __name__ == '__main__':
    main()
