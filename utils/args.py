import argparse

__all__ = ["parse_args"]


def parse_args():
    parser = argparse.ArgumentParser(description='CODE-CL algorithm')
    # General
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--aperture', nargs='+', type=float, default=[4, 4, 4, 4, 4, 4],
                        help='Aperture parameter to compute conceptors.')
    parser.add_argument('--data-aug', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--dropout', action='store_true',
                        help='Use dropout')
    parser.add_argument('--basis-batch-size', type=int, default=125,
                        help='Batch size for conceptor computation')
    parser.add_argument('--avg-pool', action='store_true',
                        help='Use average pooling in the conceptor computation')
    parser.add_argument('--threshold-conv', type=float, default=0.95,
                        help='Threshold for conceptors of conv layers')
    parser.add_argument('--threshold-linear', type=float, default=0.97,
                        help='Threshold for conceptors of linear layers')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--model', type=str, choices=['AlexNet', 'MLP', 'ResNet18'], default='AlexNet',
                        help='Choice of the model function')
    parser.add_argument('--dataset', type=str, choices=['SplitCIFAR100', 'PermutedMNIST', 'MiniIMAGENET'],
                        default='SplitCIFAR100',
                        help='Choice of dataset')
    parser.add_argument('--n-experiences', type=int, default=10,
                        help='Number of tasks')
    parser.add_argument('--print-freq', type=int, default=100,
                        help='Frequency to print results')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--aperture-gain', type=float, default=1.0,
                        help='Aperture gain')
    parser.add_argument('--save-path', type=str, default='./experiments_logs',
                        help='path to save the experiments logs')
    parser.add_argument('--experiment-name', type=str, default='CL_Conceptors',
                        help='experiment name for mlflow')
    parser.add_argument('--patience', type=float, default=6,
                        help='Patience (# of epochs to wait) used in learning rate decay')
    parser.add_argument('--lr-threshold', type=float, default=1e-5,
                        help='Minimum learning rate')
    parser.add_argument('--lr-decay', type=float, default=2.0,
                        help='Learning rate decay factor')
    parser.add_argument('--num-free-dim', type=int, default=0,
                        help='Number of free dimensions (K)')


    args = parser.parse_args()
    return args