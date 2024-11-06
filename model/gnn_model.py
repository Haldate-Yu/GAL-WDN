import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GCNConv, GATConv, SimpleConv
from .layer import GENConvolution, SAGEConv, SSGConv, GResBlockMeanConv, CombineAttention, SimpleGraphResidual

"""
    Graph convolution based model using multiple GCN layers (with multiple hops).
"""


class m_GCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim, latent_dim=96, n_aggr=45, n_hops=1, bias=False, num_layers=2,
                 dropout=0., batch_size=48):
        super(m_GCN, self).__init__()
        self.n_aggr = n_aggr
        self.n_hops = n_hops
        self.batch_size = batch_size
        self.latent = latent_dim
        self.out_dim = out_dim
        self.node_in = torch.nn.Linear(in_dim, latent_dim, bias=bias)
        self.node_out = torch.nn.Linear(latent_dim, out_dim, bias=bias)
        self.edge = torch.nn.Linear(edge_dim, latent_dim, bias=bias)

        self.gcn_aggrs = torch.nn.ModuleList()
        for _ in range(n_aggr):
            gcn = GENConvolution(latent_dim, latent_dim, latent_dim, aggr="add", bias=bias, num_layers=num_layers,
                                 dropout=dropout)
            self.gcn_aggrs.append(gcn)

    def forward(self, data):
        x, y, edge_index, edge_attr = data.x, data.y, data.edge_index, data.edge_attr

        """ Embedding for edge features. """
        edge_attr = self.edge(edge_attr)

        """ Embedding for node features. """
        Z = self.node_in(x)

        # gcn_aggr * multi_hops GCN layers
        """ 
            Mutiple GCN layers.
        """
        for gcn in self.gcn_aggrs:
            """
                Multiple Hops.
            """
            for _ in range(self.n_hops - 1):
                Z = torch.selu(gcn(Z, edge_index, edge_attr, mlp=False))
            Z = torch.selu(gcn(Z, edge_index, edge_attr, mlp=True))

        """ Reconstructing node features through a final dense layer. """
        y_predict = self.node_out(Z)

        del data

        return y_predict

    def cal_loss(self, y, y_predict):
        """ L1 Loss. """
        l1_loss = torch.nn.L1Loss(reduction='mean')
        loss = l1_loss(y, y_predict)
        del y, y_predict
        return loss


class GATResMeanConv(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=5, name='GATResMeanConv'):
        super(GATResMeanConv, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.num_blocks = num_layers

        self.lin0 = torch.nn.Linear(self.in_dim, self.hid_dim)
        self.blocks = torch.nn.ModuleList()
        self.name = name
        for _ in range(self.num_blocks):
            block = GResBlockMeanConv(self.hid_dim, self.hid_dim, self.hid_dim)
            self.blocks.append(block)

        self.lin1 = torch.nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, data):
        x, y, edge_index, edge_attr = data.x, data.y, data.edge_index, data.edge_attr
        x = self.lin0(x)

        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr)

        x = self.lin1(x)
        # x = torch.sigmoid(x)
        return x


class GraphSage(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout=0., use_weight=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_weight = use_weight

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(SAGEConv(self.hid_dim, self.hid_dim, normalize=True))
        self.lin1 = torch.nn.Linear(self.in_dim, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.out_dim)

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, y, edge_index, edge_attr, edge_weight = data.x, data.y, data.edge_index, data.edge_attr, data.weight

        x = self.lin1(x)

        for conv in self.convs:
            # todo using silu
            if self.use_weight:
                x = F.relu(conv(x, edge_index, edge_weight))
            else:
                x = F.relu(conv(x, edge_index))

        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        # todo torch.sigmoid(x)
        return x


class SSGC(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, alpha=0.6, aggr_type='ssgc', norm_type=False, dropout=0.,
                 use_weight=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.aggr_type = aggr_type
        self.norm_type = norm_type
        self.dropout = dropout
        self.use_weight = use_weight

        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(GCNConv(self.hid_dim, self.hid_dim))

        self.lin1 = torch.nn.Linear(self.in_dim, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.out_dim)

        if self.aggr_type == 'attn':
            self.attention = CombineAttention(hid_dim, 2 * hid_dim)
            self.gcn = GCNConv(self.hid_dim, self.hid_dim)
            self.gat = GATConv(self.hid_dim, self.hid_dim, 1, concat=False)
            self.simple = SimpleConv(aggr='mean')

    def reset_parameters(self):
        pass

    def forward(self, data):
        x, y, edge_index, edge_attr, edge_weight = data.x, data.y, data.edge_index, data.edge_attr, data.weight

        Z = self.lin1(x)
        z_init = Z

        z_list = []
        for i in range(self.num_layers):
            Z_residual = Z
            if self.use_weight:
                Z = self.convs[i](Z, edge_index, edge_weight).relu()
            else:
                Z = self.convs[i](Z, edge_index).relu()

            if self.aggr_type == 'attn':
                # z_comb = torch.stack([z_init, Z], dim=1)
                # Z, _ = self.attention(z_comb)
                Z = F.relu(self.gat(Z, edge_index) + z_init)
                # Z = F.relu(self.gcn(Z, edge_index) + Z_residual)
                # Z = Z + z_init
            z_list.append(Z)

        if self.aggr_type == 'ssgc':
            out_x = self.alpha * z_init + (1 - self.alpha) * sum(z_list) / self.num_layers
        elif self.aggr_type == 'ssgc_no_avg':
            out_x = self.alpha * z_init + (1 - self.alpha) * sum(z_list)
        elif self.aggr_type == 'sgc':
            out_x = self.alpha * z_init + (1 - self.alpha) * Z
        elif self.aggr_type == 'attn':
            out_x = sum(z_list) / self.num_layers
            # z_comb = torch.stack([self.alpha * z_init, (1 - self.alpha) * sum(z_list) / self.num_layers], dim=1)
            # z_comb = torch.stack([z_init, sum(z_list)], dim=1)
            # z_comb = torch.stack([z_init, z_list[-1]], dim=1)
            # out_x, _ = self.attention(z_comb)
    
        # out_x = F.dropout(out_x, p=self.dropout, training=self.training)
        out_x = self.lin2(out_x)
    
        if self.norm_type:
            out_x = torch.sigmoid(out_x)

        return out_x

class MarkovDiffusionResidual(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers=5, name='MarkovDiffusionResidual'):
        super(MarkovDiffusionResidual, self).__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.num_blocks = num_layers

        self.lin0 = torch.nn.Linear(self.in_dim, self.hid_dim)
        self.blocks = torch.nn.ModuleList()
        self.name = name
        for _ in range(self.num_blocks):
            block = GResBlockMeanConv(self.hid_dim, self.hid_dim, self.hid_dim)
            self.blocks.append(block)

        self.lin1 = torch.nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, data):
        x, y, edge_index, edge_attr = data.x, data.y, data.edge_index, data.edge_attr
        x = self.lin0(x)

        x_list = []
        for i in range(self.num_blocks):
            x = self.blocks[i](x, edge_index, edge_attr)
            x_list.append(x)
        x = sum(x_list) / self.num_blocks
        
        x = self.lin1(x)
        # x = torch.sigmoid(x)
        return x