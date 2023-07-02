import os
from datetime import datetime
from os.path import join
from time import perf_counter

import torch
import torch.nn.functional as F
from box import Box
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_scatter import scatter_sum, scatter_max

import wandb
from src.model.coarsenig_agent import IndependentBCAgent
from src.utils.dataset import GraphDataset, GraphDataLoader
from src.utils.graph_utils import get_cut_value, get_contracted_gs
from src.utils.train_utils import set_seed
from src.utils.eval_utils import evaluate_coarsening


def load_and_evaluate(config, file_name):
    device = config.train.device if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    set_seed(seed=config.train.seed,
             use_cuda='cuda' in device)

    model = IndependentBCAgent(**config.model)
    model.load_state_dict(torch.load("./checkpoints/{}.pt".format(file_name), map_location=device))
    model.to(device)

    instance_sizes = [50, 75, 100, 200]
    test_size = 100

    print(file_name)

    model.eval()
    for size in instance_sizes:
        gs, _ = load_graphs("./data/data_test_{}.bin".format(size))
        g_test_index = torch.load("./data/data_test_index_{}.bin".format(size))

        start = perf_counter()
        rst = evaluate_coarsening(model, gs, g_test_index, device, max_iter=config.train.n_iterations, ratio=config.train.contraction_ratio)

        tot_time = (perf_counter() - start) / test_size
        print(size, tot_time, rst)


def evaluate(model, data_list, device):
    with torch.no_grad():
        s_pred_list, s_hat_list = [], []
        cut_pred_list, cut_hat_list = [], []
        feasibility_list = []
        for graph in data_list:
            dl = GraphDataLoader(GraphDataset([graph]),
                                 batch_size=config.train.batch_size,
                                 shuffle=False,
                                 device=device)
            for gs in dl:
                s_hat = gs.ndata['s_hat'].view(gs.batch_size, -1)
                m_nids = {}

                for n_iter in range(config.train.n_iterations):
                    probs = model(gs)

                    new_size = int(gs.batch_num_nodes()[0] * config.train.contraction_ratio)
                    gs, m_nids, mapping = get_contracted_gs(gs, probs, new_size, m_nids)

                    if new_size == 3:
                        probs = model(gs)
                        break

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

                # print("sets calculation time:", perf_counter() - start)

                dmd, capacity = graph.ndata['demand'].to(device), graph.ndata['capacity'][0].to(device)
                constr = (torch.stack(s_pred) * dmd.view(1, -1)).sum(dim=1) / capacity
                feasibility = (constr > torch.arange(len(s_pred), device=device)).float()
                feasibility_list.extend(feasibility)

            s_pred = torch.stack(s_pred)

            s_pred_list.extend(s_pred.view(-1).tolist())
            s_hat_list.extend(s_hat.view(-1).tolist())

            cut_hat = [get_cut_value(graph, l) for l in s_hat.tolist()]
            cut_pred = [get_cut_value(graph, l) for l in s_pred.tolist()]

            cut_hat_list.extend(cut_hat)
            cut_pred_list.extend(cut_pred)

        test_fn = torch.nn.MSELoss()
        s_mse = test_fn(torch.tensor(s_pred_list).float(), torch.tensor(s_hat_list).float())
        cut_mse = test_fn(torch.tensor(cut_pred_list), torch.tensor(cut_hat_list))
        feasibility_list = torch.tensor(feasibility_list)

        return s_mse, cut_mse, feasibility_list.mean()


def train(config):
    now = datetime.now()
    training_id = now.strftime('%y%m%d%H%M%S')

    device = config.train.device
    if 'cuda' in device:
        if not torch.cuda.is_available():
            device = 'cpu'
            print("Idiot.")

    set_seed(seed=config.train.seed,
             use_cuda='cuda' in device)

    model = IndependentBCAgent(**config.model).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.opt.lr)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=config.opt.T_0)
    loss_fn = getattr(torch.nn, config.train.loss_fn)()

    wandb.init(project='NeuralSEP',
               name=training_id,
               group='Coarsening',
               reinit=True,
               config=config.to_dict())

    config.to_yaml(filename=join(wandb.run.dir, 'exp_config.yaml'))

    gs, _ = load_graphs("./data/{}.bin".format(config.train.data_file))
    g_train, g_test = train_test_split(gs,
                                       test_size=config.train.test_size,
                                       random_state=config.train.seed)

    n_update = 0

    data = GraphDataset(g_train)
    dl = GraphDataLoader(data,
                         batch_size=config.train.batch_size,
                         shuffle=config.train.shuffle_dl,
                         device=device)

    print("Start to train", training_id)
    best_cut_mse = float('inf')
    for e in range(config.train.n_epochs):
        for train_g in dl:
            train_g = train_g.to(device)
            m_nids = {}

            for _ in range(config.train.n_iterations):
                start = perf_counter()
                pred_y = model(train_g)
                if config.train.positive_weight:
                    train_y = train_g.ndata['s_hat']
                    unique_M = train_g.ndata['M'].unique()
                    for m in unique_M:
                        masked_pred = torch.masked_select(pred_y, train_g.ndata['M'] == m)
                        masked_label = torch.masked_select(train_y, train_g.ndata['M'] == m)
                        weight = masked_pred.shape[0] / pred_y.shape[0]
                        pos_weight = torch.count_nonzero(1 - masked_label) / torch.count_nonzero(masked_label)
                        if m == 0:
                            loss = weight * F.binary_cross_entropy_with_logits(masked_pred, masked_label.float(),
                                                                               pos_weight=pos_weight)
                        else:
                            loss += weight * F.binary_cross_entropy_with_logits(masked_pred, masked_label.float(),
                                                                                pos_weight=pos_weight)
                else:
                    loss = loss_fn(pred_y, train_g.ndata['s_hat'].float())

                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()
                fit_time = perf_counter() - start

                n_update += 1

                log_dict = {
                    'loss': loss.item(),
                    'fit_time': fit_time,
                    'epoch': e
                }

                if n_update % config.train.eval_every == 0:
                    model.eval()
                    start = perf_counter()
                    test_s_mse, test_cut_mse, feasibility = evaluate(model, g_test[:10], device)
                    eval_time = perf_counter() - start
                    model.train()
                    log_dict['test_mse'] = test_s_mse
                    log_dict['test_cut_mse'] = test_cut_mse
                    log_dict['eval_time'] = eval_time
                    log_dict['feasibility'] = feasibility

                    if test_cut_mse < best_cut_mse:
                        best_cut_mse = test_cut_mse
                        torch.save(model.state_dict(), "./checkpoints/coarsening_agent_{}_best.pt".format(training_id))

                if n_update % config.train.save_every == 0:
                    torch.save(model.state_dict(), "./checkpoints/coarsening_agent_{}.pt".format(training_id))
                    torch.save(opt.state_dict(), "./checkpoints/opt_coarsening_agent_{}.pt".format(training_id))

                wandb.log(log_dict)

                if train_g.batch_num_nodes().max() <= 3:
                    break

                new_size = int(train_g.batch_num_nodes().min() * config.train.contraction_ratio)
                if new_size >= 3:
                    train_g, m_nids, mapping = get_contracted_gs(train_g, train_g.ndata['s_hat'].float(), new_size,
                                                                 m_nids, eval_flag=False)
                else:
                    break

if __name__ == '__main__':
    config = Box.from_yaml(filename=join(os.getcwd(), 'config', 'coarsening_config.yaml'))  # 'config/bc_config.yaml')
    train(config)
    # load_and_evaluate(config, "coarsening_pretrained")

    # for one-shot model
    # config = Box.from_yaml(filename=join(os.getcwd(), 'config', 'oneshot_config.yaml'))
    # train(config)
    # load_and_evaluate(config, "oneshot_pretrained")
