import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch.utils import Sequential

from src.nn.gn import HeteroGN

EPS = 1e-8


def update_target(source, target, tau):
    for src_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * src_param.data + (1.0 - tau) * target_param.data)


class GNN(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 latent_dim: int,
                 n_layer: int,
                 gn_params: dict = None):
        super(GNN, self).__init__()

        if gn_params is None:
            gn_params = {}

        self.node_encoder = nn.Linear(node_dim, latent_dim)
        self.edge_encoder = nn.Linear(edge_dim, latent_dim)

        self.gn_layers = Sequential(
            *tuple([HeteroGN(dim=latent_dim, **gn_params) for _ in range(n_layer)])
        )

    def forward(self, g: dgl.graph) -> torch.tensor:
        unf = self.node_encoder(g.ndata['nf'].float())
        uef = self.edge_encoder(g.edata['ef'].float())
        unf, uef = self.gn_layers(g, unf, uef)

        return unf, uef
