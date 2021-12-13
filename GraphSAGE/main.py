import os.path as osp

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv

import numpy as np
import time
import random
from sklearn import metrics
from sklearn.model_selection import train_test_split
from utils import tab_printer, graph_reader, field_reader, target_reader

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SAGE')
parser.add_argument(
    '--epochs', type=int, default=9999, help='Number of epochs to train.')
parser.add_argument(
    '--batch', type=int, default=512, help="Batch size.")
parser.add_argument(
    '--seed', type=int, default=42, help='Random seed.')
parser.add_argument(
    '--patience', type=int, default=10, help="Patience.")
parser.add_argument(
    "--edge-path", nargs = "?", default = "./input/user_edge.csv", help = "Edge list csv.")
parser.add_argument(
    "--field-path", nargs = "?", default = "./input/user_field.npy", help = "Field npy.")
parser.add_argument(
    "--target-path", nargs = "?", default = "./input/user_age.csv", help = "Target classes csv.")


parser.add_argument(
    '--lr', type=float, default=0.1, help='Initial learning rate.')
parser.add_argument(
    '--weight-decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument(
    '--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
assert args.model in ['SAGE', 'GAT']

seed=args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

graph = graph_reader(args.edge_path)
edges = np.array(list(graph.edges()))
field_index = field_reader(args.field_path)
labels = target_reader(args.target_path)

field_dim = 64
user_count = labels.shape[0]
class_count = np.max(labels)+1
field_count = np.max(field_index)+1
class_weight = labels.shape[0] / (class_count * np.bincount(labels.squeeze()))

rand_indices = np.random.permutation(user_count)
train, val_test = train_test_split(rand_indices, test_size = 0.2, random_state=42, shuffle=True)
val, test = train_test_split(val_test, test_size = 0.5, random_state=42, shuffle=True)

field_index = torch.LongTensor(field_index)
edges = torch.LongTensor(edges)
labels = torch.LongTensor(labels.squeeze())
class_weight = torch.FloatTensor(class_weight)

data = Data(x=field_index, edge_index=edges.t(), y=labels)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.train_mask[train] = 1
data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.val_mask[val] = 1
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
data.test_mask[test] = 1

loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=args.batch,
                         shuffle=True, add_self_loops=False)

class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, field_dim, normalize=False)
        self.conv2 = SAGEConv(field_dim, out_channels, normalize=False)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.dropout(x, p=args.dropout, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size)
        return F.log_softmax(x, dim=1)

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, x, data_flow):
        block = data_flow[0]
        x = x[block.n_id]
        x = F.elu(
            self.conv1((x, x[block.res_n_id]), block.edge_index,
                       size=block.size))
        x = F.dropout(x, p=0.6, training=self.training)
        block = data_flow[1]
        x = self.conv2((x, x[block.res_n_id]), block.edge_index,
                       size=block.size)
        return F.log_softmax(x, dim=1)

class StackedGNN(nn.Module):
    def __init__(self, field_count, field_dim, output_channels):
        super(StackedGNN, self).__init__()
        self.field_embedding = nn.Embedding(field_count, field_dim)
        self.field_embedding.weight.requires_grad = True

        self.layer = SAGENet(field_dim, output_channels)

    def forward(self, field_index, data_flow):
        field_feature = self.field_embedding(field_index)
        user_feature = torch.mean(field_feature, dim=-2)
        scores = self.layer(user_feature, data_flow)
        return scores
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StackedGNN(field_count, field_dim, class_count).to(device)
class_weight = class_weight.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def classification_metrics(y_true, y_pred):
    acc = float(metrics.accuracy_score(y_true, y_pred))
    macro_f1 = float(metrics.f1_score(y_true, y_pred, average="macro"))
    return acc, macro_f1

def train(epoch):
    epoch_start_time = time.time()
    model.train()
    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device), class_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size

    test_loss =  total_loss / data.train_mask.sum().item()

    model.eval()
    total_loss = 0
    for data_flow in loader(data.val_mask):
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device), class_weight)
        total_loss += loss.item() * data_flow.batch_size

    val_loss = total_loss / data.val_mask.sum().item()

    print('Epoch: {:04d}'.format(epoch),
          "||",
          "time cost: {:.2f}s".format(time.time() - epoch_start_time), 
          'loss_train: {:.4f}'.format(test_loss),
          '||',
          'loss_val: {:.4f}'.format(val_loss))

    return val_loss

def test():
    model.eval()
    total_loss = 0
    y_true, y_pred, y_score = [], [], []
    for data_flow in loader(data.test_mask):
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device), class_weight)
        total_loss += loss.item() * data_flow.batch_size

        y_score += out.data.tolist()
        y_pred += out.max(1)[1].data.tolist()
        y_true += data.y[data_flow.n_id].to(device).data.tolist()

    test_loss = total_loss / data.test_mask.sum().item()

    acc_test, macro_f1_test = classification_metrics(y_true, y_pred)
    classification_report = metrics.classification_report(y_true, y_pred, digits=4)
    print(classification_report)

    print("Test set results:",
          "loss= {:.4f}".format(test_loss),
          "accuracy= {:.4f}".format(acc_test),
          'macro_f1= {:.4f}'.format(macro_f1_test))

# Train model
train_start_time = time.time()
bad_counter = 0
best_loss = np.inf
best_epoch = 0
patience = args.patience
for epoch in range(args.epochs):
    val_loss = train(epoch)
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = epoch
        bad_counter = 0
        best_model_state = model.state_dict()
    else:
        bad_counter += 1

    if bad_counter == patience:
        break

print("Optimization Finished!")
print("Total time elapsed: {:.2f}min".format((time.time() - train_start_time)/60))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(best_model_state)

# Testing
test()