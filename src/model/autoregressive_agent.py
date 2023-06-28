import dgl
import torch
import torch.nn as nn

from torch_scatter import scatter_softmax, scatter
from torch.nn.utils.rnn import pad_sequence

from src.nn.gnn import GNN
from src.nn.mlp import MLP


class ConstructiveBCAgent(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 latent_dim: int,
                 n_layer: int,
                 gn_params: dict = {},
                 policy_params: dict = {}):
        super(ConstructiveBCAgent, self).__init__()

        self.gnn = GNN(node_dim=node_dim, edge_dim=edge_dim, latent_dim=latent_dim, n_layer=n_layer, **gn_params)
        self.policy = MLP(input_dim=latent_dim, output_dim=1, **policy_params)

    def forward(self, g: dgl.graph):
        unf, uef = self.gnn(g)

        active_nid = g.nodes()[g.ndata['active'].bool().squeeze()]
        logits = self.policy(unf[active_nid])

        return logits, active_nid

    def get_batch_probs(self, g):
        with g.local_scope():
            logits, active_nid = self.forward(g)
            num_action_per_graph = [a.sum().long() for a in g.ndata['active'].split(g.batch_num_nodes().tolist())]
            num_action_per_graph = torch.tensor(num_action_per_graph, device=g.device)
            batch_idx = torch.arange(g.batch_size, device=g.device).long()
            batch_idx = batch_idx.repeat_interleave(num_action_per_graph, dim=0)

            # batch_idx = batch_idx[active_nid]
            probs = scatter_softmax(logits.view(-1), batch_idx)
            prob_chunks = probs.split(num_action_per_graph.tolist())
            prob_chunks_padded = pad_sequence(prob_chunks, batch_first=True)

            assert prob_chunks_padded.shape[0] == g.batch_size
            assert prob_chunks_padded.shape[1] == int(max(num_action_per_graph))

        return prob_chunks_padded, num_action_per_graph

    def get_batch_actions(self, gs: dgl.graph):
        active_nid = torch.nonzero(gs.ndata['active']).view(-1)

        if self.training:  # have to select from labels to keep labels valid
            dummy = torch.nonzero(gs.ndata['is_dummy']).squeeze()

            labels = gs.ndata['s_hat'].clone().reshape(-1)  # gs.ndata['s_hat'][dummy] = 0
            active_labels = labels[gs.ndata['active'] == 1]

            # EOS
            sum_label = scatter(active_labels, gs.ndata['batch'][gs.ndata['active'] == 1].view(-1), reduce='sum')
            done_bid = torch.where(sum_label == 0)[0]
            labels[dummy[done_bid]] = 1
            active_labels = labels[gs.ndata['active'] == 1]

            num_action_per_graph = [a.sum().long() for a in gs.ndata['active'].split(gs.batch_num_nodes().tolist())]
            num_action_per_graph = torch.tensor(num_action_per_graph, device=gs.device)

            label_chunks = active_labels.split(num_action_per_graph.tolist())
            label_chunks_padded = pad_sequence(label_chunks, batch_first=True)

            sum_label = scatter(active_labels, gs.ndata['batch'][gs.ndata['active'] == 1].view(-1), reduce='sum')
            prob = label_chunks_padded / sum_label.view(-1, 1)
        else:
            with torch.no_grad():
                prob, num_action_per_graph = self.get_batch_probs(gs)

        actions = prob.multinomial(1).view(-1)
        cum_num_actions = torch.cumsum(num_action_per_graph, dim=0)
        actions = torch.cat([actions[0].view(-1, 1), (actions[1:] + cum_num_actions[:-1]).view(-1, 1)], dim=0).view(-1)

        return active_nid[actions]
