"""
Задание 2: Эксперименты с шириной полносвязных сетей.

Запуск:
    python homework/homework_width_experiments.py
"""
import logging
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from homework.utils.experiment_utils import get_digits_loaders, save_results
from homework.utils.model_utils import build_fc_model, run_experiment, count_parameters
from homework.utils.visualization_utils import (
    plot_learning_curves, plot_bar_comparison, plot_heatmap
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

EPOCHS = 25
DEVICE = 'cpu'

WIDTH_CONFIGS = {
    'Narrow [64,32,16]': [64, 32, 16],
    'Medium [256,128,64]': [256, 128, 64],
    'Wide [1024,512,256]': [1024, 512, 256],
    'Very wide [2048,1024,512]': [2048, 1024, 512],
}


def experiment_2_1_width_comparison(train_loader, test_loader, input_size, output_size):
    """2.1 Сравнение моделей разной ширины (3 слоя)."""
    logger.info('=== 2.1: Width comparison ===')
    histories = {}
    for name, hidden in WIDTH_CONFIGS.items():
        model = build_fc_model(input_size, hidden, output_size)
        h = run_experiment(model, train_loader, test_loader,
                           epochs=EPOCHS, device=DEVICE)
        h['hidden'] = hidden
        histories[name] = h
        logger.info(f'{name}: test_acc={h["test_acc"][-1]:.4f}, '
                    f'params={count_parameters(model)}, '
                    f'time={h["train_time"]:.1f}s')

    plot_learning_curves(
        histories, title='2.1 Width Comparison — Learning Curves',
        save_path='homework/plots/width_learning_curves.png'
    )
    labels = list(histories.keys())
    test_accs = [h['test_acc'][-1] for h in histories.values()]
    n_params = [h['n_params'] for h in histories.values()]
    plot_bar_comparison(labels, test_accs, 'Test Accuracy', '2.1 Final Test Accuracy by Width',
                        save_path='homework/plots/width_test_accuracy.png')
    plot_bar_comparison(labels, n_params, '# Parameters', '2.1 Parameter Count by Width',
                        save_path='homework/plots/width_params.png')
    return histories


def experiment_2_2_grid_search(train_loader, test_loader, input_size, output_size):
    """2.2 Grid search по глубине и ширине; heatmap результатов."""
    logger.info('=== 2.2: Grid search (depth x width) ===')
    depths = [1, 2, 3]          # число скрытых слоёв
    widths = [64, 256, 512]     # ширина каждого скрытого слоя

    matrix = np.zeros((len(depths), len(widths)))
    grid_results = {}
    for i, d in enumerate(depths):
        for j, w in enumerate(widths):
            hidden = [w] * d
            model = build_fc_model(input_size, hidden, output_size)
            h = run_experiment(model, train_loader, test_loader,
                               epochs=EPOCHS, device=DEVICE)
            matrix[i, j] = h['test_acc'][-1]
            key = f'd{d}_w{w}'
            grid_results[key] = {'test_acc': h['test_acc'][-1], 'n_params': h['n_params']}
            logger.info(f'depth={d}, width={w}: test_acc={h["test_acc"][-1]:.4f}')

    row_labels = [f'{d} hidden' for d in depths]
    col_labels = [str(w) for w in widths]
    plot_heatmap(matrix, row_labels, col_labels,
                 title='2.2 Grid Search: Test Accuracy (depth x width)',
                 save_path='homework/plots/width_grid_heatmap.png')
    return grid_results


def main():
    train_loader, test_loader, input_size, output_size = get_digits_loaders()

    h1 = experiment_2_1_width_comparison(train_loader, test_loader, input_size, output_size)
    h2 = experiment_2_2_grid_search(train_loader, test_loader, input_size, output_size)

    all_results = {
        '2_1_width_comparison': {
            name: {'final_test': h['test_acc'][-1], 'n_params': h['n_params'],
                   'train_time': h['train_time']}
            for name, h in h1.items()
        },
        '2_2_grid_search': h2
    }
    save_results(all_results, 'homework/results/width_experiments/results.json')
    logger.info('Task 2 complete.')


if __name__ == '__main__':
    main()
