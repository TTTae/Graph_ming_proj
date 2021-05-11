import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


# combine final embedding
class Encoder(nn.Module):

    def __init__(self, features, embed_dim, uv_lists, r_lists, aggregator, cuda="cpu", uv=True):
        super(Encoder, self).__init__()

        self.features = features
        self.uv = uv
        self.uv_lists = uv_lists
        self.r_lists = r_lists
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        tmp_uv = []
        tmp_r = []
        for node in nodes:
            tmp_uv.append(self.uv_lists[int(node)])
            tmp_r.append(self.r_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_uv, tmp_r)  # user-item network

        self_feats = self.features.weight[nodes]
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
