import pickle
import torch
from torch_scatter import scatter

from src.utils.dataset import GraphDataset, GraphDataLoader
from src.utils.graph_utils import get_cut_value, get_contracted_gs, get_current_gs


def measure_error_bound(model, data_list, indices, device, max_iter=50, ratio=.75):
    prob_list, s_hat_list = [], []
    inference_num_list = []
    with torch.no_grad():
        for i, graph in enumerate(data_list):
            if i not in indices: continue
            dl = GraphDataLoader(GraphDataset([graph]),
                                 batch_size=128,
                                 shuffle=False,  # config.train.shuffle_dl,
                                 device=device)
            for gs in dl:
                s_hat = gs.ndata['s_hat'].view(gs.batch_size, -1)
                m_nids = {}
                # m_nids = []
                inference_num = 0.
                prev_size = gs.batch_num_nodes()[0]

                for _ in range(max_iter):
                    probs = model(gs)
                    # probs = torch.rand(gs.ndata['s_hat'].shape)
                    inference_num += 1

                    prob_list.append(probs)
                    s_hat_list.append(gs.ndata['s_hat'].float())

                    new_size = int(gs.batch_num_nodes()[0] * ratio)
                    # gs, m_nids, mapping = get_contracted_gs(gs, probs, new_size, m_nids)
                    gs, m_nids, mapping = get_contracted_gs(gs, gs.ndata['s_hat'].float(), new_size, m_nids, eval_flag=False)

                    if new_size == 3 or prev_size == new_size:
                        # probs = model(gs)
                        probs = torch.rand(gs.ndata['s_hat'].shape)
                        inference_num += 1
                        break
                    prev_size = new_size

                inference_num_list.append(inference_num)

        rst_dict = {'probs': prob_list,
                    'target_ys': s_hat_list,
                    'steps': inference_num_list
                    }

        file_name = './data/results/sub_prob_rand_error_{}.pkl'.format(graph.num_nodes()-1)
        with open(file_name, 'wb') as f:
            pickle.dump(rst_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return "done"


def evaluate_coarsening(model, data_list, indices, device, max_iter=50, ratio=.75):
    with torch.no_grad():
        s_pred_list, s_hat_list = [], []
        cut_pred_list, cut_hat_list = [], []
        feasibility_list = []
        inference_num_list = []
        m_list, lhs_list = [], []
        for i, graph in enumerate(data_list):
            if i not in indices: continue
            dl = GraphDataLoader(GraphDataset([graph]),
                                 batch_size=128,
                                 shuffle=False,  # config.train.shuffle_dl,
                                 device=device)
            for gs in dl:
                s_hat = gs.ndata['s_hat'].view(gs.batch_size, -1)
                m_nids = {}
                # m_nids = []
                inference_num = 0.
                prev_size = gs.batch_num_nodes()[0]

                for _ in range(max_iter):
                    probs = model(gs)
                    # probs = torch.rand(gs.ndata['s_hat'].shape)
                    inference_num += 1

                    new_size = int(gs.batch_num_nodes()[0] * ratio)
                    gs, m_nids, mapping = get_contracted_gs(gs, probs, new_size, m_nids)

                    if new_size == 3 or prev_size == new_size:
                        probs = model(gs)
                        # probs = torch.rand(gs.ndata['s_hat'].shape)
                        inference_num += 1
                        break
                    prev_size = new_size

                inference_num_list.append(inference_num)
                cur_nids = list(m_nids.keys())
                nid_chunk = torch.split(torch.tensor(cur_nids, device=gs.device), gs.batch_num_nodes().tolist())

                # start = perf_counter()
                s_pred = []
                for b_id, nid in enumerate(nid_chunk):
                    sets = torch.zeros(graph.num_nodes(), device=gs.device)

                    s_prob = probs[nid].view(-1)
                    s_prob[0] = 0.
                    set_sn = torch.nonzero(s_prob > 0.5).view(-1).tolist()
                    if torch.count_nonzero(s_prob > 0.5) == 0:
                        set_sn = torch.argmax(s_prob).view(1).tolist()

                    for sn in nid.tolist():
                        nodes = m_nids[sn]
                        if torch.nonzero(nid == sn) in set_sn:
                            sets[nodes - m_nids[nid[0].item()][0]] = 1
                    s_pred.append(sets)


                dmd, capacity = graph.ndata['demand'].to(device), graph.ndata['capacity'][0].to(device)
                lhs = (torch.stack(s_pred) * dmd.view(1, -1)).sum(dim=1) / capacity
                feasibility = (lhs > torch.arange(len(s_pred), device=device)).float()
                feasibility_list.append(feasibility)
                lhs_list.append(lhs)
                m_list.append(torch.arange(len(s_pred)))

            s_pred = torch.stack(s_pred)

            s_pred_list.append(s_pred)
            s_hat_list.append(s_hat)

            cut_hat = [get_cut_value(graph, l) for l in s_hat.tolist()]
            cut_pred = [get_cut_value(graph, l) for l in s_pred.tolist()]

            cut_hat_list.append(cut_hat)
            cut_pred_list.append(cut_pred)

        rst_dict = {'pred_ys': s_pred_list,
                    'target_ys': s_hat_list,
                    'pred_cuts': cut_pred_list,
                    'target_cuts': cut_hat_list,
                    'ms': m_list, 
                    'lhs': lhs_list,
                    'feasibility': feasibility_list,
                    'steps': inference_num_list
                    }

        file_name = './data/results/sub_prob_coarsening_{}.pkl'.format(graph.num_nodes()-1) if ratio < 1 else './data/results/sub_prob_oneshot_{}.pkl'.format(graph.num_nodes()-1)
        with open(file_name, 'wb') as f:
            pickle.dump(rst_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return "done"


def evaluate_auto_regressive(model, data_list, indices, device):
    with torch.no_grad():
        s_pred_list, s_hat_list = [], []
        cut_pred_list, cut_hat_list = [], []
        feasibility_list = []
        inference_num_list = []
        m_list, lhs_list = [], []

        for i, graph in enumerate(data_list):
            if i not in indices: continue
            dl = GraphDataLoader(GraphDataset([graph]),
                                 batch_size=128,
                                 shuffle=False,  # config.train.shuffle_dl,
                                 device=device,
                                 add_dummy=True)
            for gs in dl:
                s_hat = gs.ndata['s_hat'].view(-1)
                s_hat_list.append(s_hat)
                batch_idx = torch.arange(gs.batch_size, device=device).repeat_interleave(gs.batch_num_nodes())
                start_idx = scatter(gs.nodes(), batch_idx, reduce='min')
                done = torch.zeros(gs.batch_size, device=device)
                cur_s = []
                inference_nums = (-1) * torch.ones(gs.batch_size, device=gs.device)

                for n_iter in range(gs.batch_num_nodes().max()):
                    gs, _ = get_current_gs(gs=gs, cur_s=cur_s, with_label=False)
                    no_active = 1 - scatter(gs.ndata['active'].view(-1), batch_idx, reduce='max')
                    done = torch.where(done > no_active, done, no_active)
                    if done.min().bool():
                        break
                    actions, _ = model.get_batch_actions(gs)
                    # selected_node = start_idx + actions
                    dummy_selected = gs.ndata['is_dummy'][actions]
                    mask_for_inference_nums = torch.where(inference_nums < 0, dummy_selected.bool(), False)
                    inference_nums[mask_for_inference_nums] = n_iter + 1
                    done = torch.where(done > dummy_selected, done, dummy_selected)
                    # print((1 - done), actions)
                    cur_s.extend(actions[(1 - done).bool()].tolist())

                inference_num_list.append(inference_nums.mean().item())

                s_pred = torch.tensor([int(i in cur_s) for i in range(gs.num_nodes())], device=device)
                s_pred = s_pred.view(gs.batch_size, -1)
                s_pred_list.append(s_pred[:, :-1])
                # print(s_pred_list[-1].shape)

                dmd, capacity = graph.ndata['demand'].to(device), graph.ndata['capacity'][0].to(device)
                lhs = (s_pred.view(gs.batch_size, -1)[:, :-1] * dmd.view(1, -1)).sum(dim=1) / capacity
                feasibility = (lhs > torch.arange(gs.batch_size, device=device)).float()
                feasibility_list.append(feasibility)

                cut_hat = [get_cut_value(graph, s) for s in s_hat.view(gs.batch_size, -1).tolist()]
                cut_pred = [get_cut_value(graph, s) for s in s_pred.view(gs.batch_size, -1).tolist()]

                cut_hat_list.append(cut_hat)
                cut_pred_list.append(cut_pred)

                lhs_list.append(lhs)
                m_list.append(torch.arange(len(s_pred)))

        rst_dict = {'pred_ys': s_pred_list,
                'target_ys': s_hat_list,
                'pred_cuts': cut_pred_list,
                'target_cuts': cut_hat_list,
                # 'ms': m_list, 
                'lhs': lhs_list,
                'feasibility': feasibility_list,
                'steps': inference_num_list
                }

        file_name = './data/results/sub_prob_autoregressive_{}.pkl'.format(graph.num_nodes()-1)
        with open(file_name, 'wb') as f:
            pickle.dump(rst_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return "done"
