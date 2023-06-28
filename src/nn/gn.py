import dgl
import torch
import torch.nn as nn
import torch_scatter

from src.nn.mlp import MLP


class HeteroGN(nn.Module):
    def __init__(self,
                 dim: int,
                 num_etypes: int = 2,
                 num_ntypes: int = 2,
                 residual: bool = True,
                 updater_params: dict = {}):
        super(HeteroGN, self).__init__()

        self.dim = dim
        self.num_etypes = num_etypes
        self.num_ntypes = num_ntypes
        self.residual = residual

        self.outer_em = MLP(input_dim=3*dim, output_dim=dim, **updater_params)  # ef + 2nf
        self.inter_em = MLP(input_dim=3*dim, output_dim=dim, **updater_params)

        self.cust_nm = MLP(input_dim=(1+self.num_etypes)*dim, output_dim=dim, **updater_params)
        if self.num_ntypes > 1:
            self.dummy_nm = MLP(input_dim=(1+self.num_etypes)*dim, output_dim=dim, **updater_params)

    def forward(self, g: dgl.graph, nf, ef):
        with g.local_scope():
            g.ndata['h'] = nf
            g.edata['h'] = ef

            g.apply_edges(func=self.edge_update)
            g.update_all(self.msg_func,
                         self.reduce_function,
                         self.node_update)

            unf = g.ndata['h']
            uef = g.edata['msg']

            if self.residual:
                unf = unf + nf
                uef = uef + ef

            return unf, uef

    def edge_update(self, edges):  # for all edges
        u_nf, ef, v_nf = edges.src['h'], edges.data['h'], edges.dst['h']
        edge_input = torch.cat([ef, u_nf, v_nf], dim=-1)
        msg = torch.zeros((ef.shape[0], self.dim), device=edge_input.device)
        idx = edges.data['etypes'].long()
        msg[idx == 0] = self.outer_em(edge_input[idx == 0])
        msg[idx == 1] = self.inter_em(edge_input[idx == 1])
        # msg[idx == 2] = self.inner_em(edge_input[idx == 2])

        return {'msg': msg, 'idx': idx}

    def msg_func(self, edges):
        return {'msg': edges.data['msg'], 'idx': edges.data['idx']}

    def reduce_function(self, nodes):
        msg = nodes.mailbox['msg']  # (n_node, #incomingedge, hidden_dim)
        msg_type = nodes.mailbox['idx']  # (n_node, #incomingedge)
        reduced_msg = torch.zeros(msg.shape[0], self.num_etypes, msg.shape[2], device=msg.device).float()

        # aggregate function sum, max, mean
        reduced_msg = torch_scatter.scatter_sum(msg, msg_type, out=reduced_msg, dim=-2)  # (n_node, # type, hidden_dim)
        reduced_msg = reduced_msg.flatten(start_dim=1)
        # reduced_msg = torch_scatter.scatter_max(msg, msg_type, out=reduced_msg, dim=-2)
        # reduced_msg = reduced_msg[0].flatten(start_dim=1)  # (n nodes, # types x hidden)  # for scatter_max
        return {'agg_m': reduced_msg}  # (n_node, # types x hidden_dim)

    def node_update(self, nodes):
        agg_m = nodes.data['agg_m']
        nf = nodes.data['h']
        node_input = torch.cat([nf, agg_m], dim=-1)

        unf = torch.zeros((nf.shape[0], self.dim), device=node_input.device)
        idx = nodes.data['is_dummy'].long()
        unf[idx == 0] = self.cust_nm(node_input[idx == 0])
        unf[idx == 1] = self.dummy_nm(node_input[idx == 1])

        return {'h': unf}
