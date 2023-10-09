import numpy as np

import torch
from scipy.sparse import csr_matrix,lil_matrix
from models.rwgraph import RW_NN
# import utils


def fea_to_graph(feat):
    '''
    image patches feature -> graph
    To get hidden_graph

    Inputs:
        -- 'feat' (B x C x T x N), image patches feature
    Outputs:
        -- 'adj' (BT x NN x NN), node embeddings
        -- 'features'  (BT x N x C), node feature maps
    '''
    B, C, T, N = feat.shape
    features = feat.permute(0, 2, 1, 3).reshape(B * T, C, N)
    adj = torch.einsum('bcn,bcm->bnm', features, features)

    return adj, features.permute(0, 2, 1)

def generate_batches(adj, features, batch_size, shuffle=False):
    N = adj.shape[0]
    if shuffle:
        index = np.random.permutation(N)
    else:
        index = np.array(range(N), dtype=np.int32)

    adj_lst = list()
    features_lst = list()
    graph_indicator_lst = list()

    for i in range(0, N, batch_size):
        n_nodes = sum([adj[index[j]].shape[0] for j in range(i, min(i + batch_size, N))])

        # adj_batch = lil_matrix((n_nodes, n_nodes))
        # features_batch = np.zeros((n_nodes, features[0].shape[1]))
        adj_batch = torch.zeros((n_nodes, n_nodes))
        features_batch = torch.zeros((n_nodes, features[0].shape[1]))
        graph_indicator_batch = np.zeros(n_nodes)

        idx = 0
        for j in range(i, min(i + batch_size, N)):
            n = adj[index[j]].shape[0]
            adj_batch[idx:idx + n, idx:idx + n] = adj[index[j]]
            features_batch[idx:idx + n, :] = features[index[j]]
            graph_indicator_batch[idx:idx + n] = j - i

            idx += n

        # adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_batch).to(device))
        adj_batch = torch.FloatTensor(adj_batch).cuda()
        features_batch = torch.FloatTensor(features_batch).cuda()
        graph_indicator_batch = torch.LongTensor(graph_indicator_batch).cuda()

    return adj_batch, features_batch, graph_indicator_batch

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# if __name__ == '__main__':
#     args = utils.arguments.train_args()
#     adj = torch.rand(64, 197, 197).to('cuda') # 64,197,197
#     features = torch.rand(64, 197, 128).to('cuda') # 64 197 128
#     adj, features, graph_indicator = \
#         generate_batches(adj, features, 64, 'cuda', shuffle=False)
#
#     rw_gk = RW_NN(128, args.max_step, args.hidden_graphs,
#                   args.size_hidden_graphs, args.hidden_dim,
#                   args.penultimate_dim, args.rw_normalize,
#                   args.rw_dropout, args.device).to(args.device)
#
#     rw_kernel = rw_gk(adj, features, graph_indicator)
#     print(rw_kernel.shape)