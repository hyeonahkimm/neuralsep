import os
from itertools import combinations
from time import perf_counter

import dgl
import gurobipy as gp
import numpy
import torch
from box import Box
# from gurobipy import GRB
from torch_scatter import scatter

from src.model.autoregressive_agent import ConstructiveBCAgent
from src.model.coarsenig_agent import IndependentBCAgent
from src.utils.graph_utils import get_current_gs, get_cut_value
from src.utils.graph_utils import get_graph_label, get_x_mat, get_contracted_gs
from src.utils.train_utils import set_seed

EPS = 1e-6


def get_learned_autoregressive_RCI(edge, edge_x, demand, capacity):
    config = Box.from_yaml(filename=os.path.join(os.getcwd(), os.pardir, os.pardir, 'config', 'autoregressive_config.yaml'))
    model_pt = "../../checkpoints/autoregressive_pretrained.pt"

    device = config.train.device if torch.cuda.is_available() else 'cpu'
    set_seed(seed=config.train.seed,
             use_cuda='cuda' in device)

    tot_start = perf_counter()
    load_start = perf_counter()
    model = ConstructiveBCAgent(**config.model)
    model.load_state_dict(torch.load(model_pt, map_location='cpu'))
    model.eval()
    load_time = perf_counter() - load_start

    k = numpy.ceil(sum(demand) / capacity)
    graph_start = perf_counter()
    graph = get_graph_based_on_x(edge, edge_x, demand, capacity, False, False)
    gs = get_graph_label(graph, with_label=False, add_dummy=True)
    graph_time = perf_counter() - graph_start

    model = model.to(device)
    graph, gs = graph.to(device), gs.to(device)

    forward_start = perf_counter()
    with torch.no_grad():
        batch_idx = torch.arange(gs.batch_size, device=device).repeat_interleave(gs.batch_num_nodes())
        done = torch.zeros(gs.batch_size, device=device)
        cur_s = []

        for _ in range(graph.num_nodes()):
            gs, _ = get_current_gs(gs=gs, cur_s=cur_s, with_label=False)
            no_active = 1 - scatter(gs.ndata['active'].view(-1), batch_idx, reduce='max')
            done = torch.where(done > no_active, done, no_active)
            if done.min().bool():
                break
            actions = model.get_batch_actions(gs)
            dummy_selected = gs.ndata['is_dummy'][actions]
            done = torch.where(done > dummy_selected, done, dummy_selected)
            cur_s.extend(actions[(1 - done).bool()].tolist())

    s_pred = torch.tensor([int(i in cur_s) for i in range(gs.num_nodes())], device=device)
    s_pred = s_pred.reshape(gs.batch_size, -1)
    cut_pred = [get_cut_value(graph, s) for s in s_pred.tolist()]

    forward_time = perf_counter() - forward_start
    # print(cut_pred)

    list_s, list_rhs, list_z = [], [], []
    demand = graph.ndata['demand']
    capacity = graph.ndata['capacity'][0]
    for m in range(int(k)):
        rhs = (demand.view(-1) * s_pred[m, :-1]).sum()
        rhs = rhs / capacity
        rhs = torch.ceil(rhs)

        if cut_pred[m] < 2 * rhs - EPS:
            s = numpy.nonzero(s_pred[m].tolist())
            list_s.append([u + 1 for u in s])
            list_rhs.append(rhs.item())
            list_z.append(cut_pred[m])

    tot_time = perf_counter() - tot_start

    info = [tot_time, load_time, graph_time, forward_time, numpy.mean([len(s) for s in list_s])]

    return list_s, list_rhs, list_z, info

def get_learned_coarsening_RCI(edge, edge_x, demand, capacity):
    config = Box.from_yaml(filename=os.path.join(os.getcwd(), os.pardir, os.pardir, 'config', 'coarsening_config.yaml'))
    model_pt = "../../checkpoints/coarsening_pretrained.pt"  # no pos_weight

    # config = Box.from_yaml(filename=os.path.join(os.getcwd(), os.pardir, os.pardir, 'config', 'oneshot_config.yaml'))
    # model_pt = "../../checkpoints/oneshot_pretrained.pt"

    device = config.train.device if torch.cuda.is_available() else 'cpu'
    set_seed(seed=config.train.seed,
             use_cuda='cuda' in device)

    tot_start = perf_counter()
    load_start = perf_counter()
    model = IndependentBCAgent(**config.model)
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.eval()
    load_time = perf_counter() - load_start

    # print(edge, edge_x, demand, capacity)

    graph_start = perf_counter()
    k = numpy.ceil(sum(demand) / capacity)
    graph = get_graph_based_on_x(edge, edge_x, demand, capacity, False, False)
    gs = get_graph_label(graph, with_label=False)
    graph_time = perf_counter() - graph_start
    m_nids = {}

    model = model.to(device)
    graph, gs = graph.to(device), gs.to(device)

    forward_start = perf_counter()
    with torch.no_grad():
        for _ in range(config.train.n_iterations):
            probs = model(gs)
            # probs[torch.nonzero(gs.ndata['demand'] == 0)] = 0.

            new_size = int(gs.batch_num_nodes()[0] * config.train.contraction_ratio)
            gs, m_nids, mapping = get_contracted_gs(gs, probs, new_size, m_nids)

            if new_size == 3:
                probs = model(gs)
                break

        cur_nids = list(m_nids.keys())
        nid_chunk = torch.split(torch.tensor(cur_nids, device=gs.device), gs.batch_num_nodes().tolist())

        s_pred, cut_pred = [], []
        for b_id, nid in enumerate(nid_chunk):
            sets = torch.zeros(graph.num_nodes(), device=gs.device)

            s_prob = probs[nid].view(-1)
            s_prob[0] = 0.
            set_sn = torch.nonzero(s_prob > 0.5).view(-1).tolist()
            if len(set_sn) == 0:
                set_sn = torch.argmax(s_prob).view(1).tolist()
            for sn in nid.tolist():
                nodes = m_nids[sn]
                if torch.nonzero(nid == sn) in set_sn:
                    sets[nodes - m_nids[nid[0].item()][0]] = 1
            s_pred.append(sets)
            cut_pred.append(get_cut_value(graph, sets.tolist()))

        forward_time = perf_counter() - forward_start
        # print(cut_pred)

        list_s, list_rhs, list_z = [], [], []
        demand = graph.ndata['demand']
        capacity = graph.ndata['capacity'][0]
        for m in range(int(k)):
            rhs = (demand.view(-1) * s_pred[m]).sum()
            rhs = rhs / capacity
            rhs = torch.ceil(rhs)

            # if cut_pred[m] < 2 * (m + 1) - EPS:
            if cut_pred[m] < 2 * rhs - EPS:
                # rhs = 2 * torch.ceil(rhs)
                s = numpy.nonzero(s_pred[m].tolist())
                list_s.append([u + 1 for u in s])
                list_rhs.append(rhs.item())
                list_z.append(cut_pred[m])

        tot_time = perf_counter() - tot_start

        info = [tot_time, load_time, graph_time, forward_time, numpy.mean([len(s) for s in list_s])]

        return list_s, list_rhs, list_z, info


def get_graph_based_on_x(edge, x_bar, demand, capacity, dummy_tf=True, complete=True, depot_complete=False):
    demand = demand[:-1]
    n = len(demand)  # last - dummy depot

    # based on x_bar (index converting)
    cur_e = [(u - 1, v - 1) for u, v in edge]

    src, dest, ef = [], [], []
    for idx, (u, v) in enumerate(cur_e):
        src.extend([u, v])
        dest.extend([v, u])
        ef.extend([x_bar[idx], x_bar[idx]])

    if complete:
        for (u, v) in combinations(range(n), 2):
            if (u, v) not in cur_e and (v, u) not in cur_e:
                src.extend([u, v])
                dest.extend([v, u])
                ef.extend([0.0, 0.0])

    g = dgl.graph((torch.tensor(src).long(), torch.tensor(dest).long()))

    if depot_complete:
        new_edge = [n for n in range(g.num_nodes()) if n not in g.in_edges(0)[0] and n != 0]
        g.add_edges(new_edge, [0 for _ in range(len(new_edge))])
        g.add_edges([0 for _ in range(len(new_edge))], new_edge)

    if g.num_nodes() != len(demand):
        print(g.num_nodes(), n, demand)
        print(edge)

    g.edata['x_val'] = torch.tensor(ef).view(-1, 1)
    g.edata['etypes'] = torch.zeros(len(ef), device=g.device).int()
    g.ndata['demand'] = torch.tensor(demand).view(-1, 1)
    g.ndata['capacity'] = torch.tensor([capacity] * n).view(-1, 1)
    g.ndata['is_depot'] = torch.tensor(demand == 0.0).int().view(-1, 1)

    if dummy_tf:
        dummy = g.num_nodes()
        g.add_edges([i for i in range(dummy)], [dummy for _ in range(dummy)])
        g.ndata['is_dummy'] = torch.zeros(g.num_nodes()).int()
        g.ndata['is_dummy'][dummy] = 1
    else:
        g.ndata['is_dummy'] = torch.zeros(g.num_nodes()).int()

    return g


if __name__ == '__main__':
    import pickle

    with open("./data/pickles/P-n16-k8.pickle", 'rb') as f:
        data = pickle.load(f)

    edge = data['list_e'][1]
    edge_x = data['list_x'][1]
    demand = data['demand']
    capa = data['capacity']

    edge = [(2, 7), (2, 11), (3, 7), (3, 8), (4, 9), (4, 13), (5, 11), (5, 12), (6, 8), (6, 15), (9, 14), (10, 14), (10, 15),
     (12, 16), (13, 16)]
    edge_x = [1. for _ in range(len(edge))]

    edge, edge_x, demand = numpy.array(edge), numpy.array(edge_x), numpy.array(demand)

    S, RHS, z = get_learned_ind_RCI(edge, edge_x, demand, capa)
