import torch
import numpy as np
from torch import Tensor


def gh(P):
    b = P.shape[0]
    n = P.shape[1]
    A = np.stack([np.ones((n, n)) - np.eye(n) for i in range(b)])

    edge_num = int(np.sum(A[0], axis=(0, 1)))

    n_pad = n
    edge_pad = edge_num

    G = np.zeros((b, n_pad, edge_pad), dtype=np.float32)
    H = np.zeros((b, n_pad, edge_pad), dtype=np.float32)

    for bi in range(b):
        edge_idx = 0
        for i in range(n):
            for j in range(n):
                if A[bi, i, j] == 1:
                    G[bi, i, edge_idx] = 1
                    H[bi, j, edge_idx] = 1
                    edge_idx += 1

    return G, H, edge_num


def kronecker(A, B):
    AB = torch.einsum("eab,ecd->eacbd", A, B)
    AB = AB.view(A.size(0), A.size(1)*B.size(1), A.size(2)*B.size(2))
    return AB


def moving_average(feat, saved_ma, alpha):
    if len(saved_ma) == 0:
        ema = feat
    else:
        ema = saved_ma * alpha + feat * (1 - alpha)
    return ema


def nanorinf_replace(a):
    EPS = 1e-20
    if torch.isnan(a).int().sum().item() > 0:
        a = torch.where(torch.isnan(a), torch.full_like(a, EPS), a)
    if torch.isinf(a).int().sum().item() > 0:
        a = torch.where(torch.isinf(a), torch.full_like(a, EPS), a)

    return a


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union

def reshape_edge_feature(F: Tensor, G: Tensor, H: Tensor, device=None):
    """
    Reshape edge feature matrix into X, where features are arranged in the order in G, H.
    :param F: raw edge feature matrix
    :param G: factorized adjacency matrix, where A = G * H^T
    :param H: factorized adjacancy matrix, where A = G * H^T
    :param device: device. If not specified, it will be the same as the input
    :return: X
    """
    if device is None:
        device = F.device

    batch_num = F.shape[0]
    feat_dim = F.shape[1]
    point_num, edge_num = G.shape[1:3]
    X = torch.zeros(batch_num, 2 * feat_dim, edge_num, dtype=torch.float32, device=device)
    #X = torch.matmul(F, G) - torch.matmul(F, H)
    #X = X.squeeze(0)
    #X = torch.nn.functional.normalize(X, p=2, dim=0).unsqueeze(0)
    X[:, 0:feat_dim, :] = torch.matmul(F, G)
    X[:, feat_dim:2*feat_dim, :] = torch.matmul(F, H)
    #X = torch.nn.functional.normalize(X.squeeze(0), p=2, dim=0).unsqueeze(0)
    #print('11111')

    return X
