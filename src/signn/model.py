import os
import uuid
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv, GATv2Conv, ARMAConv, ResGatedGraphConv, WLConvContinuous, GatedGraphConv, PDNConv
import torch.utils.data
import copy
import random
random.seed(0)

class EarlyStopping:
    def __init__(self, patience=30, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False

class SiGNNTrainer:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger
        self.tmp_dir = 'tmp'
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def fit(self, model, X, y, train_idx, distances, K, alpha, epochs, supervision, sim_measure, lr, test_idx=None, report_test=False, batch_size=128):
        self.K = K
        self.alpha = alpha
        self.sim_measure = sim_measure
        self.lr = lr

        n_val = int(len(train_idx)*0.10)
        val_idx = random.sample(train_idx, n_val)
        train_idx = [x for x in train_idx if x not in val_idx]
        train_batch_size = min(batch_size//2, len(train_idx))
        if supervision=="supervised":
            other_idx = np.array([])
        elif supervision=="inductive":
            other_idx = np.array([i for i in range(len(X)) if (i not in train_idx)&(i not in test_idx)])
        elif supervision=="transductive":
            other_idx = np.array([i for i in range(len(X)) if i not in train_idx])
        other_batch_size = min(batch_size - train_batch_size, len(other_idx))
        train_dataset = Dataset(train_idx)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
        val_batch_size = min(batch_size//2, len(val_idx))
        val_dataset = Dataset(val_idx)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=1)
        other_idx_val = np.array([])
        other_batch_size_val = 0

        if report_test:
            test_batch_size = min(batch_size//2, len(test_idx))
            other_idx_test = np.array([i for i in range(len(X)) if i not in test_idx])
            other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))
            test_dataset = Dataset(test_idx)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)


        self.adj = torch.from_numpy(distances.astype(np.float32))

        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
        file_path = os.path.join(self.tmp_dir, str(uuid.uuid4()))

        optimizer = optim.Adam(model.parameters(), lr, weight_decay=4e-3)
        es = EarlyStopping()
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print('Number of trainable parameters=', pytorch_total_params)

        best_loss = 100.0

        # Training
        epoch = 0
        done = False
        while epoch < epochs and not done:
            epoch += 1
            model.train()
            optimizer.zero_grad()

            for sampled_train_idx in train_loader:
                sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)
                idx = np.concatenate((sampled_train_idx, sampled_other_idx))
                _X, _y, _adj = self.X[idx].to(self.device), self.y[sampled_train_idx].to(self.device), self.adj[idx][:,idx]
                outputs = model(_X, _adj, K, alpha, sim_measure)
                loss = F.nll_loss(outputs[:len(sampled_train_idx)], _y)

                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for sampled_val_idx in val_loader:
                    _X, _y, _adj = self.X[sampled_val_idx].to(self.device), self.y[sampled_val_idx].to(self.device), self.adj[sampled_val_idx][:,sampled_val_idx]
                    outputs = model(_X, _adj, K, alpha, sim_measure)
                    vloss = F.nll_loss(outputs[:len(sampled_val_idx)], _y)
                    vacc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, self.sim_measure, val_loader, self.device, other_idx_val, other_batch_size_val)
            acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, self.sim_measure, train_loader, self.device, other_idx, other_batch_size)
            if es(model, vloss.item()):
                done = True
            if vloss <= best_loss:
                best_loss = vloss
                torch.save(model.state_dict(), file_path)
            if report_test:
                test_acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, self.sim_measure, test_loader, self.device, other_idx_test, other_batch_size_test)
                self.logger.log('--> Epoch {}: Training loss {:5.4f}; Training accuracy: {:5.4f}; Validation loss {:5.4f}; Validation accuracy: {:5.4f}; test accuracy: {:5.4f}'.format(epoch, loss.item(), acc, vloss.item(), vacc.item(), test_acc))
            else:
                self.logger.log('--> Epoch {}: Training loss {:5.4f}; Training accuracy: {:5.4f}; Validation loss {:5.4f}; Validation accuracy: {:5.4f}'.format(epoch, loss.item(), acc, vloss.item(), vacc.item()))
        
        # Load the best model
        model.load_state_dict(torch.load(file_path))
        model.eval()
        os.remove(file_path)

        return model
    
    def test(self, model, test_idx, batch_size=128):
        test_batch_size = min(batch_size//2, len(test_idx))
        other_idx_test = np.array([i for i in range(len(self.X)) if i not in test_idx])
        other_batch_size_test = min(batch_size - test_batch_size, len(other_idx_test))
        test_dataset = Dataset(test_idx)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=1)
        acc = compute_accuracy(model, self.X, self.y, self.adj, self.K, self.alpha, self.sim_measure, test_loader, self.device, other_idx_test, other_batch_size_test)
        return acc.item()

def compute_accuracy(model, X, y, adj, K, alpha, sim_measure, loader, device, other_idx, other_batch_size):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx in loader:
            sampled_other_idx = np.random.choice(other_idx, other_batch_size, replace=False)
            idx = np.concatenate((batch_idx, sampled_other_idx))
            _X, _y, _adj = X[idx].to(device), y[idx][:len(batch_idx)].to(device), adj[idx][:,idx]
            outputs = model(_X, _adj, K, alpha, sim_measure)
            preds = outputs[:len(batch_idx)].max(1)[1].type_as(_y)
            _correct = preds.eq(_y).double()
            correct += _correct.sum()
            total += len(batch_idx)
    acc = correct / total
    return acc

class SiGNN(nn.Module):
    def __init__(self, input_size, nb_classes, gnn, dilated, num_layers=1, n_feature_maps=64, dropout=0.5):
        super(SiGNN, self).__init__()
        self.num_layers = num_layers
        self.gnn = gnn

        self.block_1 = ResNetBlock(input_size, n_feature_maps, dilated)
        self.block_2 = ResNetBlock(n_feature_maps, n_feature_maps, dilated)
        self.block_3 = ResNetBlock(n_feature_maps, n_feature_maps, dilated)

        if self.num_layers == 1:
            if gnn=='GraphConvolution':
                self.gc1 = GraphConvolution(n_feature_maps, nb_classes)
            elif gnn in ['GATConv']:
                self.gc1 = GATConv(n_feature_maps, nb_classes, edge_dim=1)
            elif gnn in ['GATv2Conv']:
                self.gc1 = GATv2Conv(n_feature_maps, nb_classes, edge_dim=1)
            elif gnn in ['GraphSage', 'GCNConv']:   
                # https://github.com/pyg-team/pytorch_geometric/discussions/5983 weighted SAGEConv() == GraphConv(aggr="mean")
                self.gc1 = GCNConv(n_feature_maps, nb_classes, aggr="mean")
            elif gnn in ['ARMAConv']:
                self.gc1 = ARMAConv(n_feature_maps, nb_classes)
            elif gnn in ['PDNConv']:
                self.gc1 = PDNConv(n_feature_maps, nb_classes, edge_dim=1, hidden_channels= 16)
            elif gnn in ['GatedGraphConv']:
                self.gc1 = GatedGraphConv(n_feature_maps, nb_classes)
            elif gnn in ['WLConvContinuous']:
                self.gc1 = WLConvContinuous()
            elif gnn in ['ResGatedGraphConv']:
                self.gc1 = ResGatedGraphConv(n_feature_maps, nb_classes, edge_dim=1)
        elif self.num_layers == 2:
            if gnn=='GraphConvolution':
                self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
                self.gc2 = GraphConvolution(n_feature_maps, nb_classes)
                self.dropout = dropout
            elif gnn in ['GATConv']:
                self.gc1 = GATConv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc2 = GATConv(n_feature_maps, nb_classes, edge_dim=1)
                self.dropout = dropout
            elif gnn in ['GATv2Conv']:
                self.gc1 = GATv2Conv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc2 = GATv2Conv(n_feature_maps, nb_classes, edge_dim=1)
                self.dropout = dropout
            elif gnn in ['GraphSage', 'GCNConv']:
                self.gc1 = GCNConv(n_feature_maps, n_feature_maps, aggr="mean")
                self.gc2 = GCNConv(n_feature_maps, nb_classes, aggr="mean")
                self.dropout = dropout
            elif gnn in ['ARMAConv']:
                self.gc1 = ARMAConv(n_feature_maps, n_feature_maps)
                self.gc2 = ARMAConv(n_feature_maps, nb_classes)
                self.dropout = dropout
            elif gnn in ['PDNConv']:
                self.gc1 = PDNConv(n_feature_maps, n_feature_maps, edge_dim=1, hidden_channels= 16)
                self.gc2 = PDNConv(n_feature_maps, nb_classes, edge_dim=1, hidden_channels= 16)
                self.dropout = dropout
            elif gnn in ['GatedGraphConv']:
                self.gc1 = GatedGraphConv(n_feature_maps, n_feature_maps)
                self.gc2 = GatedGraphConv(n_feature_maps, nb_classes)
                self.dropout = dropout
            elif gnn in ['WLConvContinuous']:
                self.gc1 = WLConvContinuous()
                self.gc2 = WLConvContinuous()
                self.dropout = dropout
            elif gnn in ['ResGatedGraphConv']:
                self.gc1 = ResGatedGraphConv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc2 = ResGatedGraphConv(n_feature_maps, nb_classes, edge_dim=1)
                self.dropout = dropout
        elif self.num_layers == 3:
            if gnn=='GraphConvolution':
                self.gc1 = GraphConvolution(n_feature_maps, n_feature_maps)
                self.gc2 = GraphConvolution(n_feature_maps, n_feature_maps)
                self.gc3 = GraphConvolution(n_feature_maps, nb_classes)
                self.dropout = dropout
            elif gnn in ['GATConv']:
                self.gc1 = GATConv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc2 = GATConv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc3 = GATConv(n_feature_maps, nb_classes, edge_dim=1)
                self.dropout = dropout
            elif gnn in ['GATv2Conv']:
                self.gc1 = GATv2Conv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc2 = GATv2Conv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc3 = GATv2Conv(n_feature_maps, nb_classes, edge_dim=1)
                self.dropout = dropout
            elif gnn in ['GraphSage', 'GCNConv']:
                self.gc1 = GCNConv(n_feature_maps, n_feature_maps, aggr="mean")
                self.gc2 = GCNConv(n_feature_maps, n_feature_maps, aggr="mean")
                self.gc3 = GCNConv(n_feature_maps, nb_classes, aggr="mean")
                self.dropout = dropout
            elif gnn in ['ARMAConv']:
                self.gc1 = ARMAConv(n_feature_maps, n_feature_maps)
                self.gc2 = ARMAConv(n_feature_maps, n_feature_maps)
                self.gc3 = ARMAConv(n_feature_maps, nb_classes)
                self.dropout = dropout
            elif gnn in ['PDNConv']:
                self.gc1 = PDNConv(n_feature_maps, n_feature_maps, edge_dim=1, hidden_channels= 16)
                self.gc2 = PDNConv(n_feature_maps, n_feature_maps, edge_dim=1, hidden_channels= 16)
                self.gc3 = PDNConv(n_feature_maps, nb_classes, edge_dim=1, hidden_channels= 16)
                self.dropout = dropout
            elif gnn in ['GatedGraphConv']:
                self.gc1 = GatedGraphConv(n_feature_maps, n_feature_maps)
                self.gc2 = GatedGraphConv(n_feature_maps, n_feature_maps)
                self.gc3 = GatedGraphConv(n_feature_maps, nb_classes)
                self.dropout = dropout
            elif gnn in ['WLConvContinuous']:
                self.gc1 = WLConvContinuous()
                self.gc2 = WLConvContinuous()
                self.gc3 = WLConvContinuous()
                self.dropout = dropout
            elif gnn in ['ResGatedGraphConv']:
                self.gc1 = ResGatedGraphConv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc2 = ResGatedGraphConv(n_feature_maps, n_feature_maps, edge_dim=1)
                self.gc3 = ResGatedGraphConv(n_feature_maps, nb_classes, edge_dim=1)
                self.dropout = dropout

    def forward(self, x, adj, K, alpha, sim_measure):
        if sim_measure in ['dtw','attention_scores', 'attention_weights']:
            ranks = torch.argsort(adj, dim=1)
        elif sim_measure in ['attention_outputs', 'attention_outputs_dotp']:
            ranks = torch.argsort(adj, dim=1, descending=True)
        sparse_index = [[], []]
        sparse_value = []
        for i in range(len(adj)):
            _sparse_value = []
            for j in ranks[i][:K]:
                sparse_index[0].append(i)
                sparse_index[1].append(j)
                if sim_measure in ['dtw','attention_scores', 'attention_weights']:
                    _sparse_value.append(1/np.exp(alpha*adj[i][j]))
                elif sim_measure in ['attention_outputs', 'attention_outputs_dotp']:
                    _sparse_value.append(np.exp(alpha*adj[i][j]))
            _sparse_value = np.array(_sparse_value)
            _sparse_value /= _sparse_value.sum()
            sparse_value.extend(_sparse_value.tolist())
        sparse_index = torch.LongTensor(sparse_index)
        sparse_value = torch.FloatTensor(sparse_value)
        adj = torch.sparse.FloatTensor(sparse_index, sparse_value, adj.size())
        # device = self.gc1.bias.device
        device = torch.device('cuda:0')
        sparse_index = sparse_index.to(device)
        if self.gnn in ['GATConv', 'GATv2Conv', 'ResGatedGraphConv', 'PDNConv']:
            sparse_value = sparse_value.reshape(-1,1)
        sparse_value = sparse_value.to(device)
        adj = adj.to(device)

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = F.avg_pool1d(x, x.shape[-1]).squeeze()

        if self.num_layers == 1:
            if self.gnn=="GraphConvolution":
                x = self.gc1(x, adj)
            elif self.gnn in ['GraphSage', 'GCNConv', 'GATConv', 'GATv2Conv', 'ARMAConv', 'PDNConv', 'GatedGraphConv', 'WLConvContinuous', 'ResGatedGraphConv']:
                x = self.gc1(x, sparse_index, sparse_value)
        elif self.num_layers == 2:
            if self.gnn=="GraphConvolution":
                x = F.relu(self.gc1(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gc2(x, adj)
            elif self.gnn in ['GraphSage', 'GCNConv', 'GATConv', 'GATv2Conv', 'ARMAConv', 'ResGatedGraphConv']:
                x = F.relu(self.gc1(x, sparse_index, sparse_value))
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gc2(x, sparse_index, sparse_value)
        elif self.num_layers == 3:
            if self.gnn=="GraphConvolution":
                x = F.relu(self.gc1(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
                x = F.relu(self.gc2(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gc3(x, adj)
            elif self.gnn in ['GraphSage', 'GCNConv', 'GATConv', 'GATv2Conv', 'ARMAConv', 'ResGatedGraphConv']:
                x = F.relu(self.gc1(x, sparse_index, sparse_value))
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gc2(x, sparse_index, sparse_value)
                x = F.dropout(x, self.dropout, training=self.training)
                x = self.gc3(x, sparse_index, sparse_value)

        x = F.log_softmax(x, dim=1)

        return x

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilated):
        super(ResNetBlock, self).__init__()
        self.expand = True if in_channels < out_channels else False
        if dilated is False:
            self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
            self.bn_x = nn.BatchNorm1d(out_channels)
            self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=2)
            self.bn_y = nn.BatchNorm1d(out_channels)
            self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=1)
            self.bn_z = nn.BatchNorm1d(out_channels)

        elif dilated is True:
            self.conv_x = nn.Conv1d(in_channels, out_channels, 7, padding=3)
            self.bn_x = nn.BatchNorm1d(out_channels)
            self.conv_y = nn.Conv1d(out_channels, out_channels, 5, padding=4, dilation=2)
            self.bn_y = nn.BatchNorm1d(out_channels)
            self.conv_z = nn.Conv1d(out_channels, out_channels, 3, padding=4, dilation=4)
            self.bn_z = nn.BatchNorm1d(out_channels)

        if self.expand:
            self.shortcut_y = nn.Conv1d(in_channels, out_channels, 1)
        self.bn_shortcut_y = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        B, _, L = x.shape
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        if self.expand:
            x = self.shortcut_y(x)
        x = self.bn_shortcut_y(x)
        out += x
        out = F.relu(out)
       
        return out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, idx):
        self.idx = idx

    def __getitem__(self, index):
        return self.idx[index]

    def __len__(self):
        return len(self.idx)
