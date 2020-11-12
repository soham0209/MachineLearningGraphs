import argparse
import time

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.nn as pyg_nn

import models
import utils
import matplotlib.pyplot as plt


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')

    parser.set_defaults(model_type='GCN',
                        dataset='cora',
                        num_layers=2,
                        batch_size=32,
                        hidden_dim=32,
                        dropout=0.0,
                        epochs=200,
                        opt='adam',  # opt_parser
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()


def train(dataset, task, args):
    # use mask to split train/validation/test
    test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # build model
    if args.model_type != 'APPNP':
        model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
                                args, task=task)
    else:
        alpha = 0.1  # Change here if you need to change alpha
        niter = 10   # Change here if you need to change niterations of Pagerank
        appnp_prop = models.PPRPowerIteration(dataset.data.edge_index, alpha, niter, args.dropout)
        model = models.APPNP(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
                             appnp_prop, args, task=task)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    accuracy = []
    # train
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print('Epoch: ', epoch, 'Training loss: ', total_loss)

        if epoch % 100 == 0:
            test_acc = test(loader, model)
            print('Test acc: ', test_acc)
            accuracy.append([epoch, test_acc])
    test_acc = test(loader, model)
    accuracy.append([args.epochs, test_acc])
    plot_accuracy(np.array(accuracy), args)
    print('Final test acc: ', test_acc)


def test(loader, model, is_validation=False):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y
        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]

        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.test_mask).item()
    return correct / total


def main():
    args = arg_parse()

    if args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        train(dataset, 'node', args)
    else:
        raise RuntimeError('Unknown datasets')


def plot_accuracy(values, args):
    fig = plt.figure()
    plt.plot(values[:, 0], values[:, 1])
    plt.suptitle(args.model_type)
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.savefig(args.model_type + '_acc.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
