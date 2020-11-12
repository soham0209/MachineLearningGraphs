import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from utils import calc_A_hat


class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, task='node'):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers - 1):
            self.convs.append(conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.task = task

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        ############################################################################
        # TODO: Your code here! 
        # Each layer in GNN should consist of a convolution (specified in model_type),
        # a non-linearity (use RELU), and dropout. 
        # HINT: the __init__ function contains parameters you will need. 
        # Our implementation is ~6 lines, but don't worry if you deviate from this.

        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)
        for l in range(1, len(self.convs)):
            x = self.convs[l](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)

        ############################################################################

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""

    def __init__(self, in_channels, out_channels, reducer='mean',
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        ############################################################################
        # TODO: Your code here!
        # Define the layers needed for the forward function.
        # Our implementation is ~2 lines, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels, out_channels)  # TODO
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)  # TODO

        ############################################################################

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, mysize=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, mysize):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]

        ############################################################################
        # TODO: Your code here!
        # Given x_j, perform the aggregation of a dense layer followed by a RELU non-linearity.
        # Notice that the aggregator operation will be done in self.propagate.
        # HINT: It may be useful to read the pyg_nn implementation of GCNConv,
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        x_j = self.lin(x_j)
        x_j = F.relu(x_j)
        ############################################################################

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]

        ############################################################################
        # TODO: Your code here! Perform the update step here.
        # Perform a MLP with skip-connection, that is a concatenation followed by
        # a linear layer and a RELU non-linearity.
        # Finally, remember to normalize as vector as shown in GraphSage algorithm.
        # Our implementation is ~4 lines, but don't worry if you deviate from this.

        aggr_out = torch.cat([x, aggr_out], dim=1)
        aggr_out = self.agg_lin(aggr_out)
        aggr_out = F.relu(aggr_out)

        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2)

        ############################################################################

        return aggr_out


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat
        self.dropout = dropout

        ############################################################################
        #  TODO: Your code here!
        # Define the layers needed for the forward function. 
        # Remember that the shape of the output depends the number of heads.
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.lin = nn.Linear(in_channels, self.heads * out_channels)  # TODO

        ############################################################################

        ############################################################################
        #  TODO: Your code here!
        # The attention mechanism is a single feed-forward neural network parametrized
        # by weight vector self.att. Define the nn.Parameter needed for the attention
        # mechanism here. Remember to consider number of heads for dimension!
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        self.att = nn.Parameter(torch.Tensor(self.heads, 2 * out_channels))  # TODO

        ############################################################################

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(self.heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

        ############################################################################

    def forward(self, x, edge_index, size=None):
        ############################################################################
        #  TODO: Your code here!
        # Apply your linear transformation to the node feature matrix before starting
        # to propagate messages.
        # Our implementation is ~1 line, but don't worry if you deviate from this.

        x = self.lin(x)  # TODO
        ############################################################################

        # Start propagating messages.
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        #  Constructs messages to node i for each edge (j, i).
        ############################################################################
        #  TODO: Your code here! Compute the attention coefficients alpha as described
        # in equation (7). Remember to be careful of the number of heads with 
        # dimension!
        # Our implementation is ~5 lines, but don't worry if you deviate from this.
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        comb = torch.cat([x_i, x_j], dim=-1)
        alpha = F.leaky_relu((comb * self.att).sum(-1), 0.2)
        alpha = pyg_utils.softmax(alpha, edge_index_i, num_nodes=size_i)
        ############################################################################

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return (x_j * alpha.unsqueeze(-1)).view(-1, self.heads * self.out_channels)

    def update(self, aggr_out):
        # Updates node embedings.
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class APPNP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, propagation, args, task='node'):
        super().__init__()
        mlp = [nn.Linear(input_dim, hidden_dim)]
        for i in range(1, args.num_layers):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
        mlp.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.ModuleList(mlp)

        self.dropout = args.dropout
        self.propagation = propagation

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for l in range(len(self.mlp) - 1):
            x = self.mlp[l](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        final_logits = self.propagation(x)
        return F.log_softmax(final_logits, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix, alpha: float, niter: int, dropout):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', torch.FloatTensor((1 - alpha) * M))
        self.dropout = nn.Dropout(dropout)

    def forward(self, local_preds: torch.FloatTensor):
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds
        return preds



