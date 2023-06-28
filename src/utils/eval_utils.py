import torch
from torch_scatter import scatter

from src.utils.dataset import GraphDataset, GraphDataLoader
from src.utils.graph_utils import get_cut_value, get_contracted_gs, get_current_gs


def evaluate_coarsening(model, data_list, device, max_iter=50, ratio=.75):
    with torch.no_grad():
        s_pred_list, s_hat_list = [], []
        cut_pred_list, cut_hat_list = [], []
        feasibility_list = []
        inference_num_list = []
        for graph in data_list:
            dl = GraphDataLoader(GraphDataset([graph]),
                                 batch_size=128,
                                 shuffle=False,  # config.train.shuffle_dl,
                                 device=device)
            for gs in dl:
                s_hat = gs.ndata['s_hat'].view(gs.batch_size, -1)
                m_nids = {}
                # m_nids = []
                inference_num = 0.

                for _ in range(max_iter):
                    probs = model(gs)
                    inference_num += 1

                    new_size = int(gs.batch_num_nodes()[0] * ratio)
                    gs, m_nids, mapping = get_contracted_gs(gs, probs, new_size, m_nids)

                    if new_size == 3:
                        probs = model(gs)
                        inference_num += 1
                        break

                inference_num_list.append(inference_num)
                cur_nids = list(m_nids.keys())
                nid_chunk = torch.split(torch.tensor(cur_nids, device=gs.device), gs.batch_num_nodes().tolist())

                # start = perf_counter()
                s_pred = []
                for b_id, nid in enumerate(nid_chunk):
                    sets = torch.zeros(graph.num_nodes(), device=gs.device)

                    # x = gs.edata['x_val'][gs.edge_ids([nid[0] for _ in range(len(nid[1:]))], nid[1:])]
                    # min_nid = nid[x.argmin() + 1].item()
                    #
                    # for sn in nid.tolist():
                    #     nodes = m_nids[sn]
                    #     if sn == min_nid:
                    #         sets[nodes - m_nids[nid[0].item()][0]] = 1

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

                # print("sets calculation time:", perf_counter() - start)

                dmd, capacity = graph.ndata['demand'].to(device), graph.ndata['capacity'][0].to(device)
                lhs = (torch.stack(s_pred) * dmd.view(1, -1)).sum(dim=1) / capacity
                feasibility = (lhs > torch.arange(len(s_pred), device=device)).float()
                feasibility_list.extend(feasibility)

            s_pred = torch.stack(s_pred)

            s_pred_list.extend(s_pred.view(-1).tolist())
            s_hat_list.extend(s_hat.view(-1).tolist())

            cut_hat = [get_cut_value(graph, l) for l in s_hat.tolist()]
            # cut_pred = [get_expected_cut(g, l) for g, l in zip(g_list, s_pred)]
            cut_pred = [get_cut_value(graph, l) for l in s_pred.tolist()]

            cut_hat_list.extend(cut_hat)
            cut_pred_list.extend(cut_pred)

        test_fn = torch.nn.MSELoss()
        pred_ys, target_ys = torch.tensor(s_pred_list), torch.tensor(s_hat_list)
        pred_cuts, target_cuts = torch.tensor(cut_pred_list), torch.tensor(cut_hat_list)
        s_mse = test_fn(pred_ys.float(), target_ys.float())
        diff_abs = (pred_ys - target_ys).abs()
        ratio_diff_exceeding_half = (diff_abs > 0.5).int().sum() / target_ys.shape[0]
        cut_mse = test_fn(pred_cuts, target_cuts)
        cut_gap = ((pred_cuts - target_cuts) / torch.where(target_cuts == 0., torch.ones_like(target_cuts), target_cuts)).mean()
        feasibility_list = torch.tensor(feasibility_list)
        inference_num_list = torch.tensor(inference_num_list)

        rst = {'s_mse': s_mse.item(),
               # 'max_l1': diff_abs.max().item(),
               # 'ratio_diff_exceeding_half': ratio_diff_exceeding_half.item(),
               'cut_mse': cut_mse.item(),
               'cut_gap': cut_gap.item(),
               'feasible_ratio': feasibility_list.mean().item(),
               'avg_inference_num': inference_num_list.mean().item()}

        return rst


def evaluate_constructive(model, data_list, device):
    with torch.no_grad():
        s_pred_list, s_hat_list = [], []
        cut_pred_list, cut_hat_list = [], []
        feasibility_list = []
        inference_num_list = []
        for graph in data_list:
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
                    actions = model.get_batch_actions(gs)
                    # selected_node = start_idx + actions
                    dummy_selected = gs.ndata['is_dummy'][actions]
                    mask_for_inference_nums = torch.where(inference_nums < 0, dummy_selected.bool(), False)
                    inference_nums[mask_for_inference_nums] = n_iter + 1
                    done = torch.where(done > dummy_selected, done, dummy_selected)
                    # print((1 - done), actions)
                    cur_s.extend(actions[(1 - done).bool()].tolist())

                inference_num_list.append(inference_nums.mean().item())

                s_pred = torch.tensor([int(i in cur_s) for i in range(gs.num_nodes())], device=device)
                s_pred_list.append(s_pred)

                dmd, capacity = graph.ndata['demand'].to(device), graph.ndata['capacity'][0].to(device)
                lhs = (s_pred.view(gs.batch_size, -1)[:, :-1] * dmd.view(1, -1)).sum(dim=1) / capacity
                feasibility = (lhs > torch.arange(gs.batch_size, device=device)).float()
                feasibility_list.extend(feasibility)

                cut_hat = [get_cut_value(graph, s) for s in s_hat.view(gs.batch_size, -1).tolist()]
                cut_pred = [get_cut_value(graph, s) for s in s_pred.view(gs.batch_size, -1).tolist()]

                cut_hat_list.extend(cut_hat)
                cut_pred_list.extend(cut_pred)

        test_fn = torch.nn.MSELoss()
        pred_ys, target_ys = torch.cat(s_pred_list), torch.cat(s_hat_list)
        pred_cuts, target_cuts = torch.tensor(cut_pred_list), torch.tensor(cut_hat_list)
        s_mse = test_fn(pred_ys.float(), target_ys.float())
        cut_mse = test_fn(pred_cuts, target_cuts)
        cut_gap = ((pred_cuts - target_cuts) / torch.where(target_cuts == 0., torch.ones_like(target_cuts), target_cuts)).mean()
        feasibility_list = torch.tensor(feasibility_list)
        inference_num_list = torch.tensor(inference_num_list)

        rst = {'s_mse': s_mse.item(),
               # 'max_l1': diff_abs.max().item(),
               # 'ratio_diff_exceeding_half': ratio_diff_exceeding_half.item(),
               'cut_mse': cut_mse.item(),
               'cut_gap': cut_gap.item(),
               'feasible_ratio': feasibility_list.mean().item(),
               'avg_inference_num': inference_num_list.mean().item()}

        return rst
