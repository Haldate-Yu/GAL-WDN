import numpy as np
from .anytown import ChebNet as Anytown_Ori_GCN
from .ctown import ChebNet as Ctown_Ori_GCN
from .richmond import ChebNet as Richmond_Ori_GCN
from .gnn_model import *


def load_model(args, trn_x, trn_y):
    # the original model
    if args.model == 'ori':
        if args.wds == 'anytown':
            model = Anytown_Ori_GCN(np.shape(trn_x)[-1], np.shape(trn_y)[-1])
        elif args.wds == 'ctown':
            model = Ctown_Ori_GCN(np.shape(trn_x)[-1], np.shape(trn_y)[-1])
        elif args.wds == 'richmond':
            model = Richmond_Ori_GCN(np.shape(trn_x)[-1], np.shape(trn_y)[-1])
        else:
            print('Water distribution system is unknown.\n')
            raise Exception('Water distribution system is unknown.\n')
    # m-gcn
    elif args.model == 'm_gcn':
        model = m_GCN(in_dim=np.shape(trn_x)[-1], out_dim=np.shape(trn_y)[-1], edge_dim=3, latent_dim=args.hidden_dim, 
                      n_aggr=args.n_layers, n_hops=args.m_gcn_n_hops, bias=False, num_layers=args.m_gcn_n_layers, dropout=args.dropout)
    # GAT-Res
    elif args.model == 'GAT_RES_Small':
        model = GATResMeanConv(np.shape(trn_x)[-1], args.hidden_dim, np.shape(trn_y)[-1], num_layers=args.n_layers)
    elif args.model == 'GAT_RES_Large':
        model = GATResMeanConv(np.shape(trn_x)[-1], args.hidden_dim, np.shape(trn_y)[-1], num_layers=args.n_layers)
    # sage
    elif args.model == 'sage':
        model = GraphSage(np.shape(trn_x)[-1], args.hidden_dim, np.shape(trn_y)[-1], num_layers=args.n_layers,
                          dropout=args.dropout, use_weight=args.use_weight)
    elif args.model == 'ssgc':
        model = SSGC(np.shape(trn_x)[-1], args.hidden_dim, np.shape(trn_y)[-1], num_layers=args.n_layers,
                     alpha=args.alpha, aggr_type=args.aggr_type, norm_type=args.norm_type,
                     dropout=args.dropout, use_weight=args.use_weight)
    elif args.model == 'mdkres':
        model = MarkovDiffusionResidual(np.shape(trn_x)[-1], args.hidden_dim, np.shape(trn_y)[-1], num_layers=args.n_layers)

    return model
