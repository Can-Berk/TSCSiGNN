import os
import argparse

import numpy as np

from src.utils import read_dataset, read_multivariate_dataset

dataset_dir = './datasets/univariate'
multivariate_dir = './datasets/multivariate'
output_dir = './tmp'

multivariate_datasets = ['CharacterTrajectories', 'Handwriting', 'PhonemeSpectra']

def argsparser():
    parser = argparse.ArgumentParser("SiGNN data creator")
    parser.add_argument('--dataset', help='Dataset name', default='Handwriting')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--label_level', help='Percentage of the labeled time-series to the all dataset', type=str, default="full")

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    # Seeding
    np.random.seed(args.seed)

    # Create dirs
    if args.dataset in multivariate_datasets:
        output_dir = os.path.join(output_dir, 'multivariate_datasets_'+str(args.label_level)+'_label')  
    else:
        output_dir = os.path.join(output_dir, 'univariate_datasets_'+str(args.label_level)+'_label')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read data
    if args.dataset in multivariate_datasets:
        X, y, train_idx, test_idx = read_multivariate_dataset(multivariate_dir, args.dataset, args.label_level)
    else:
        X, y, train_idx, test_idx = read_dataset(dataset_dir, args.dataset, args.label_level)
    data = {
                'X': X,
                'y': y,
                'train_idx': train_idx,
                'test_idx': test_idx
            }
    np.save(os.path.join(output_dir, args.dataset), data)
