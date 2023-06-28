import dgl
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from src.nn.gnn import GNN
from src.nn.mlp import MLP

EPS = 1e-8


class IndependentBCAgent(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 latent_dim: int,
                 n_layer: int,
                 gn_params: dict = None,
                 policy_params: dict = None):
        super(IndependentBCAgent, self).__init__()

        if gn_params is None:
            gn_params = {}

        if policy_params is None:
            policy_params = {'out_act': 'Sigmoid'}

        self.gnn = GNN(node_dim=node_dim,
                       edge_dim=edge_dim,
                       latent_dim=latent_dim,
                       n_layer=n_layer,
                       **gn_params)
        self.policy = MLP(input_dim=latent_dim,
                          output_dim=1, **policy_params)

    def forward(self, g: dgl.graph) -> torch.tensor:
        unf, uef = self.gnn(g)
        probs = self.policy(unf)

        return probs

    def get_batch_probs(self, g):
        with g.local_scope():
            probs = self.forward(g)
            probs = probs.view(-1)
            num_nodes_per_graph = g.batch_num_nodes()

            prob_chunks = probs.split(num_nodes_per_graph.tolist())
            prob_chunks_padded = pad_sequence(prob_chunks, batch_first=True)

            assert prob_chunks_padded.shape[0] == g.batch_size
            assert prob_chunks_padded.shape[1] == int(max(num_nodes_per_graph))

            # dist = Categorical(probs=prob_chunks_padded)

            return prob_chunks_padded

    def get_batch_probs_penalty(self, g):
        with g.local_scope():
            probs = self.forward(g)
            probs = probs.view(-1)
            num_nodes_per_graph = g.batch_num_nodes()

            demand = (g.ndata['demand'] / g.ndata['capacity']).view(-1)
            demand_chunk = demand.split(num_nodes_per_graph.tolist())
            demand_chunk_padded = pad_sequence(demand_chunk, batch_first=True)

            prob_chunks = probs.split(num_nodes_per_graph.tolist())
            prob_chunks_padded = pad_sequence(prob_chunks, batch_first=True)

            assert prob_chunks_padded.shape[0] == g.batch_size
            assert prob_chunks_padded.shape[1] == int(max(num_nodes_per_graph))
            assert demand_chunk_padded.shape[1] == int(max(num_nodes_per_graph))

            return prob_chunks_padded, demand_chunk_padded.float(), g.ndata['M'][g.ndata['demand'] == 0.]
