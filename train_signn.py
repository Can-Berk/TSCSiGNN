import os
import argparse

import numpy as np
import torch

from src.signn.model import SiGNN, SiGNNTrainer
from src.utils import read_dataset_from_npy, Logger

data_dir = './tmp'
log_dir = './logs'

multivariate_datasets = ['CharacterTrajectories', 'Handwriting', 'PhonemeSpectra']

def train(X, y, train_idx, test_idx, distances, device, logger, K, alpha, epochs, supervision, sim_measure, lr, gnn, dilated):
    nb_classes = len(np.unique(y, axis=0))
    input_size = X.shape[1]

    model = SiGNN(input_size, nb_classes, gnn, dilated)
    model = model.to(device)
    trainer = SiGNNTrainer(device, logger)
    model = trainer.fit(model, X, y, train_idx, distances, K, alpha, epochs, supervision, sim_measure, lr, test_idx)
    acc = trainer.test(model, test_idx)

    return acc


def argsparser():
    parser = argparse.ArgumentParser("SiGNN")
    parser.add_argument('--dataset', help='Dataset name', default='Handwriting')
    parser.add_argument('--seed', help='Random seed', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--label_level', help='label_level', type=str, default="full")
    parser.add_argument('--K', help='K', type=int, default=3)
    parser.add_argument('--alpha', help='alpha', type=float, default=1)
    parser.add_argument('--epochs', help='epochs', type=int, default=1500)
    parser.add_argument('--supervision', help='supervision', default="supervised")
    parser.add_argument('--sim_measure', help='sim_measure', default="dtw")
    parser.add_argument('--lr', help='lr', type=float, default=1e-4)
    parser.add_argument('--gnn', help='gnn', default="GATv2Conv")
    parser.add_argument('--dilated', help='dilated', action=argparse.BooleanOptionalAction, default=True)

    return parser

if __name__ == "__main__":
    # Get the arguments
    parser = argsparser()
    args = parser.parse_args()

    # Setup the gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("--> Running on the GPU")
    else:
        device = torch.device("cpu")
        print("--> Running on the CPU")

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    
    if args.sim_measure=='dtw':
        if args.dataset in multivariate_datasets:
            dtw_dir = os.path.join(data_dir, 'datasets_dtw')
            distances = np.load(os.path.join(dtw_dir, args.dataset+'_multi.npy'))
        else:
            dtw_dir = os.path.join(data_dir, 'datasets_dtw')
            distances = np.load(os.path.join(dtw_dir, args.dataset+'.npy'))
    elif args.sim_measure == 'attention_scores':
        att_dir = os.path.join(data_dir, 'datasets_att')
        distances = np.load(os.path.join(att_dir, args.dataset+'_att_scores.npy'))
    elif args.sim_measure == 'attention_weights':
        att_dir = os.path.join(data_dir, 'datasets_att')
        distances = np.load(os.path.join(att_dir, args.dataset+'_att_weights.npy'))
    elif args.sim_measure == 'attention_outputs':
        att_dir = os.path.join(data_dir, 'datasets_att')
        distances = np.load(os.path.join(att_dir, args.dataset+'_att_outputs.npy'))
        
    
    log_dir = './results/logs' + '_' + str(args.dataset) + '_' + str(args.gnn) + '_' + (args.label_level)
    if args.dilated is True:
        log_dir = log_dir + '_' + 'dilated'
    out_dir = os.path.join(log_dir, 'signn_log_'+str(args.label_level)+'_label'+str(args.K)+'_'+str(args.alpha)+'_'+str(args.epochs)+'epochs'+'_'+str(args.lr))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, args.dataset+'_'+str(args.seed)+'_'+str(args.supervision)+'_'+str(args.sim_measure)+'.txt')

    with open(out_path, 'w') as f:
        logger = Logger(f)
        # Read data
        if args.dataset in multivariate_datasets:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'multivariate_datasets_'+str(args.label_level)+'_label', args.dataset+'.npy'))
        else:
            X, y, train_idx, test_idx = read_dataset_from_npy(os.path.join(data_dir, 'univariate_datasets_'+str(args.label_level)+'_label', args.dataset+'.npy'))

        # Train the model
        acc = train(X, y, train_idx, test_idx, distances, device, logger, args.K, args.alpha, args.epochs, args.supervision, args.sim_measure, args.lr, args.gnn, args.dilated)

        logger.log('--> {} Test Accuracy: {:5.4f}'.format(args.dataset, acc))
        logger.log(str(acc))
