from typing import List

import dgl
import numpy as np
import torch
from torch_scatter import scatter_max, scatter
from torch.nn.utils.rnn import pad_sequence


def get_graph_label(g: dgl.graph,
                    depot_complete: bool = True,
                    with_label: bool = True,
                    add_dummy: bool = False):
    g = g.clone()
    if with_label:
        s_hat = g.ndata['s_hat'].T  # [N X K].T
        K = s_hat.shape[0]
    else:
        K = np.ceil(sum(g.ndata['demand']) / g.ndata['capacity'][0])
        K = int(K)

    etypes = torch.zeros(g.num_edges())
    g.edata['etypes'] = etypes.int()

    if depot_complete:
        u, v = g.in_edges(0)  # v = [0, 0, 0, -----]
        u_remain = np.setdiff1d(np.arange(1, g.num_nodes()), u.view(-1).numpy())
        u_remain = torch.tensor(u_remain, dtype=torch.int64)
        v_remain = torch.zeros_like(u_remain, dtype=torch.int64)
        g.add_edges(u_remain, v_remain)  # will be zero-initialized
        g.add_edges(v_remain, u_remain)  #

    if add_dummy:
        dummy_idx = g.num_nodes()
        g.add_edges([i for i in range(dummy_idx)], [dummy_idx for _ in range(dummy_idx)])

    gs = dgl.batch([g for _ in range(K)])
    gs.ndata['batch'] = torch.arange(K).repeat_interleave(g.num_nodes()).view(-1, 1)
    gs.ndata['M'] = torch.arange(K).repeat_interleave(g.num_nodes()).view(-1, 1)
    gs.ndata['scaled_M'] = torch.arange(K).repeat_interleave(g.num_nodes()).view(-1, 1) / K
    gs.ndata['is_dummy'] = torch.zeros(gs.num_nodes()).view(-1)

    if add_dummy:
        gs.ndata['is_dummy'][gs.nodes() % g.num_nodes() == dummy_idx] = 1
        if with_label:
            s_hat = torch.cat([s_hat, torch.zeros(K, 1)], dim=1)

    if with_label:
        gs.ndata['s_hat'] = s_hat.reshape(-1, 1)

    gs.edata['batch'] = torch.arange(K).repeat_interleave(g.num_edges()).view(-1, 1)
    gs.ndata['nf'] = torch.cat([gs.ndata['demand'] / gs.ndata['capacity'], gs.ndata['scaled_M']], dim=1)
    gs.edata['ef'] = gs.edata['x_val']

    return gs


def set_etypes(gs, prob_list):
    s_probs = []
    for p in prob_list:
        p = p.detach()
        p[0] = 0.0
        s_probs.extend(p)

    s_probs = torch.tensor(s_probs)

    dist = torch.distributions.Bernoulli(probs=s_probs)
    sample = dist.sample()
    sample = sample.to(gs.device)

    gs.ndata['s'] = sample
    etypes = torch.zeros(gs.num_edges(), device=gs.device)
    etypes[gs.out_edges(torch.nonzero(sample).view(-1), 'eid')] = 1
    gs.edata['etypes'] = etypes.int()


def get_current_g(g: dgl.graph, cur_s: List, m, aggr_s: bool = True):
    cur_g = g.clone()
    # cur_g = copy.deepcopy(g)

    if aggr_s:
        cur_g.remove_edges(cur_g.in_edges(cur_s, 'eid'))

        etypes = torch.zeros(cur_g.num_edges(), device=g.device)
        etypes[cur_g.out_edges(cur_s, 'eid')] = 1
        cur_g.edata['etypes'] = etypes.int()
    else:  # not remove edges in S for sequential decoding only
        cur_g.edata['etypes'] = torch.zeros(cur_g.num_edges(), device=g.device).int()

    cur_g.ndata['s'] = torch.zeros(cur_g.num_nodes(), device=g.device)
    cur_g.ndata['s'][cur_s] = 1

    cur_g.ndata['M'] = torch.tensor([m for _ in range(cur_g.num_nodes())], device=g.device).view(-1, 1)
    cur_g.ndata['active'] = torch.ones(cur_g.num_nodes(), device=g.device)
    cur_g.ndata['active'][cur_s] = 0
    cur_g.ndata['active'][0] = 0

    dummy = torch.nonzero(cur_g.ndata['is_dummy']).squeeze()
    cur_g.ndata['capacity'][dummy] = cur_g.ndata['capacity'][0]
    cur_g.ndata['demand'][dummy] = cur_g.ndata['demand'][cur_s].sum()

    # infeasible => dummy is not active
    if cur_g.ndata['demand'][dummy] < m * cur_g.ndata['capacity'][dummy] + 1:
        cur_g.ndata['active'][dummy] = 0

    cur_g.edata['ef'] = torch.cat([cur_g.edata['x_val']])
    cur_g.ndata['nf'] = torch.cat([cur_g.ndata['demand'] / cur_g.ndata['capacity'], cur_g.ndata['M']], dim=-1)

    return cur_g


def get_current_gs(gs: dgl.graph, cur_s: List = [], with_label: bool = False):
    gs.ndata['active'] = torch.ones(gs.num_nodes(), device=gs.device)
    gs.ndata['active'][cur_s] = 0
    gs.ndata['active'][0] = 0

    dummy = torch.nonzero(gs.ndata['is_dummy']).squeeze()
    gs.ndata['capacity'][dummy] = scatter(gs.ndata['capacity'].view(-1), gs.ndata['batch'].view(-1), reduce='max').view(-1, 1)
    if len(cur_s) > 0:
        gs.ndata['demand'][dummy] = scatter(gs.ndata['demand'][cur_s].view(-1), gs.ndata['batch'][cur_s].view(-1), reduce='sum').view(-1, 1)

    etypes = torch.zeros(gs.num_edges(), device=gs.device)
    etypes[gs.out_edges(cur_s, 'eid')] = 1
    gs.edata['etypes'] = etypes.int()

    # infeasible => dummy is not active
    infeasible = gs.ndata['demand'][dummy] < gs.ndata['M'][dummy] * gs.ndata['capacity'][dummy] + 1
    gs.ndata['active'][dummy[infeasible.view(-1)]] = 0

    gs.edata['ef'] = torch.cat([gs.edata['x_val']])
    gs.ndata['nf'] = torch.cat([gs.ndata['demand'] / gs.ndata['capacity'], gs.ndata['scaled_M']], dim=1)

    if with_label:
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
        label_chunks_padded /= sum_label.view(-1, 1)

    else:
        label_chunks_padded = None

    return gs, label_chunks_padded


def get_cut_value(g, s):
    s_nid = torch.nonzero(torch.tensor(s, device=g.device)).view(-1)
    cut = 0.0
    us, vs, eids = g.out_edges(s_nid, 'all')

    for u, v, eid in zip(us, vs, eids):
        if v not in s_nid:
            cut += g.edata['x_val'][eid].item()

    return cut


def get_expected_cuts(gs, s):
    # x_mat = get_x_mat(g)
    # w_mat = torch.matmul(s.view(-1, 1), (1 - s).view(-1, 1).T)
    # cut = (x_mat * w_mat).sum()
    cuts = [0.0 for _ in range(gs.batch_size)]
    for u in range(gs.num_nodes()):
        _, vs, eids = gs.out_edges(u, 'all')
        for v, e in zip(vs, eids):
            cuts[gs.edata['batch'][e]] += (gs.edata['x_val'][e].item() * (1 - s[v].item()) * s[u].item()) / 2
    return cuts


def get_x_mat(g, return_dense=False):
    adj = g.adj().to(g.device)
    x_adj = torch.sparse_coo_tensor(adj._indices(), g.edata['x_val'].view(-1), adj.size())

    if return_dense:
        x_adj = x_adj.to_dense()
    return x_adj


def get_cut_by_x_mat(x_mat, s_prob):
    w_mat = torch.matmul(s_prob.view(-1, 1), (1 - s_prob).view(-1, 1).T)
    return (x_mat * w_mat).sum()


def get_contracted_gs(gs, s_prob, new_size, merged_nid, eval_flag=True):
    mapping = gs.nodes().clone()

    gs.edata['prob'] = _get_edge_prob(gs, s_prob)
    gs.ndata['prob'] = s_prob.clone()
    gs.ndata['nid'] = gs.nodes().clone()

    out, eid = scatter_max(gs.edata['prob'], gs.edata['batch'].view(-1), 0)

    if torch.count_nonzero(out > 0.) == 0:
        return gs, merged_nid, mapping

    while gs.batch_num_nodes().min() > new_size and torch.count_nonzero(out > 0.) > 0:
        eid = eid[torch.nonzero(out > 0.).view(-1)]
        gs, mapping = _merge_nodes(gs, mapping, eid)
        out, eid = scatter_max(gs.edata['prob'], gs.edata['batch'].view(-1), 0)

    if eval_flag:
        nodes = mapping.unique()
        find_nid = {}
        new_merged_nid = {}
        for i, n in enumerate(nodes):
            find_nid[n.item()] = i
            nids = torch.nonzero(mapping == n).view(-1)
            new_m_nids = [merged_nid[nid.item()] for nid in nids if nid.item() in merged_nid.keys()]
            new_m_nids.extend([torch.tensor([nid]) for nid in nids if nid.item() not in merged_nid.keys()])
            new_merged_nid[i] = torch.cat(new_m_nids, dim=0).unique()
    else:
        new_merged_nid = merged_nid

    gs.ndata['nf'] = torch.cat([gs.ndata['demand'] / gs.ndata['capacity'], gs.ndata['scaled_M']], dim=1)
    gs.edata['ef'] = gs.edata['x_val']

    return gs, new_merged_nid, mapping


def _merge_nodes(gs, mapping, selected):
    us, vs = gs.find_edges(selected.view(-1))
    for u, v in zip(us, vs):
        mapping[torch.nonzero(mapping == gs.ndata['nid'][u]).view(-1)] = mapping[gs.ndata['nid'][v]].item()
    gs.ndata['demand'][vs] += gs.ndata['demand'][us]
    gs.ndata['prob'][vs] += gs.ndata['prob'][us]
    gs.ndata['prob'][vs] = gs.ndata['prob'][vs] / 2

    # remove inner edges
    reverse = gs.edge_ids(vs, us)
    gs.remove_edges(torch.cat([selected.view(-1), reverse.view(-1)]))

    # update edge connections
    uu, vv, eid = gs.out_edges(us, 'all')
    new_uu = torch.tensor([vs[torch.nonzero(us == u).view(-1)][0] for u in uu], device=gs.device)
    start_eid = gs.num_edges()
    gs.add_edges(new_uu, vv)
    gs.edata['x_val'][start_eid:], gs.edata['etypes'][start_eid:] = gs.edata['x_val'][eid], gs.edata['etypes'][eid]
    gs.edata['batch'][start_eid:], gs.edata['prob'][start_eid:] = gs.edata['batch'][eid], gs.edata['prob'][eid]
    gs.edata['ef'][start_eid:] = gs.edata['ef'][eid]

    uu, vv, eid = gs.in_edges(us, 'all')
    new_vv = torch.tensor([vs[torch.nonzero(us == v).view(-1)][0] for v in vv], device=gs.device)
    start_eid = gs.num_edges()
    gs.add_edges(uu, new_vv)
    gs.edata['x_val'][start_eid:], gs.edata['etypes'][start_eid:] = gs.edata['x_val'][eid], gs.edata['etypes'][eid]
    gs.edata['batch'][start_eid:], gs.edata['prob'][start_eid:] = gs.edata['batch'][eid], gs.edata['prob'][eid]
    gs.edata['ef'][start_eid:] = gs.edata['ef'][eid]

    # remove parallel edges
    gs = gs.to('cpu')
    gs.edata['etypes'] = gs.edata['etypes'].float()
    gs.edata['batch'] = gs.edata['batch'].float()
    gs = gs.to_simple(copy_ndata=True, copy_edata=True, aggregator='sum')
    gs.edata['etypes'] = gs.edata['etypes'].int()
    gs.edata['batch'] = (gs.edata['batch'] / gs.edata['count'].view(-1, 1)).long()
    gs = gs.to(new_uu.device)

    gs = gs.remove_self_loop()
    gs.remove_nodes(us)

    # set batch info
    batch_num_nodes, batch_num_edges = [], []
    for b in range(gs.ndata['batch'].max() + 1):
        batch_num_nodes.append(torch.count_nonzero(gs.ndata['batch'].view(-1) == b).item())
        batch_num_edges.append(torch.count_nonzero(gs.edata['batch'].view(-1) == b).item())
    gs.set_batch_num_nodes(torch.tensor(batch_num_nodes, device=gs.device))
    gs.set_batch_num_edges(torch.tensor(batch_num_edges, device=gs.device))
    _ = gs.edata.pop('count')

    return gs, mapping


def _get_edge_prob(gs, s_prob):
    s_chunk = s_prob.split(gs.batch_num_nodes().tolist())

    e_prob_list = []
    for s in s_chunk:
        e_prob = torch.matmul(s.view(-1, 1), s.view(-1, 1).T)
        e_prob += torch.matmul((1 - s).view(-1, 1), (1 - s).view(-1, 1).T)
        e_prob_list.append(e_prob)

    e_prob_block = torch.block_diag(*e_prob_list)
    src_nodes = torch.nonzero(gs.ndata['demand'].view(-1) == 0.)
    e_prob_block[src_nodes, :] = 0.
    e_prob_block[:, src_nodes] = 0.

    return e_prob_block[gs.edges()]


def _get_big_edge_prob(gs, s_prob):
    e_prob = torch.zeros(gs.num_edges(), device=gs.device)
    src_nodes = torch.nonzero(gs.ndata['demand'].view(-1) == 0.)
    for i, j, eid in zip(*gs.edges('all')):
        if i in src_nodes or j in src_nodes:
            e_prob[eid] = 0
        else:
            e_prob[eid] = s_prob[i] * s_prob[j] + (1 - s_prob[i]) * (1 - s_prob[j])

    return e_prob

