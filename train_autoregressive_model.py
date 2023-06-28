import os
import numpy
from datetime import datetime
from os.path import join
from time import perf_counter

import torch
from box import Box
from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_scatter import scatter

import wandb
from src.model.autoregressive_agent import ConstructiveBCAgent
from src.utils.dataset import GraphDataset, GraphDataLoader
from src.utils.graph_utils import get_cut_value, get_current_gs
from src.utils.train_utils import set_seed
from src.utils.eval_utils import evaluate_constructive


def load_and_evaluate(config, file_name):
    device = config.train.device if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    set_seed(seed=config.train.seed,
             use_cuda='cuda' in device)

    model = ConstructiveBCAgent(**config.model)
    model.load_state_dict(torch.load("./checkpoints/{}.pt".format(file_name), map_location=device))
    model.to(device)

    instance_sizes = [50, 75, 100, 200]
    test_size = 100

    print(file_name)

    model.eval()
    for size in instance_sizes:
        gs, _ = load_graphs("./data/data_test_{}.bin".format(size))
        # g_test = gs[:test_size]  #
        g_test = numpy.random.choice(gs, test_size)

        start = perf_counter()
        rst = evaluate_constructive(model, g_test, device)

        tot_time = (perf_counter() - start) / test_size
        print(size, tot_time, rst)


def evaluate(model, data_list, device):
    with torch.no_grad():
        s_pred_list, s_hat_list = [], []
        cut_pred_list, cut_hat_list = [], []
        for graph in data_list:
            dl = GraphDataLoader(GraphDataset([graph]),
                                 batch_size=config.train.batch_size,
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

                for _ in range(gs.batch_num_nodes().max()):
                    gs, _ = get_current_gs(gs=gs, cur_s=cur_s, with_label=False)
                    no_active = 1 - scatter(gs.ndata['active'].view(-1), batch_idx, reduce='max')
                    done = torch.where(done > no_active, done, no_active)
                    if done.min().bool():
                        break
                    actions = model.get_batch_actions(gs)
                    # selected_node = start_idx + actions
                    dummy_selected = gs.ndata['is_dummy'][actions]
                    done = torch.where(done > dummy_selected, done, dummy_selected)
                    # print((1 - done), actions)
                    cur_s.extend(actions[(1 - done).bool()].tolist())

                s_pred = torch.tensor([int(i in cur_s) for i in range(gs.num_nodes())], device=device)
                s_pred_list.append(s_pred)

                cut_hat = [get_cut_value(graph, s) for s in s_hat.view(gs.batch_size, -1).tolist()]
                cut_pred = [get_cut_value(graph, s) for s in s_pred.view(gs.batch_size, -1).tolist()]

                cut_hat_list.extend(cut_hat)
                cut_pred_list.extend(cut_pred)

        test_fn = torch.nn.MSELoss()
        s_mse = test_fn(torch.cat(s_pred_list).float(), torch.cat(s_hat_list).float())
        cut_mse = test_fn(torch.tensor(cut_pred_list), torch.tensor(cut_hat_list))
        return s_mse.item(), cut_mse.item()


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

    model = ConstructiveBCAgent(**config.model).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.opt.lr)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=config.opt.T_0)
    loss_fn = getattr(torch.nn, config.train.loss_fn)()

    wandb.init(project='CVRP-Cutting',
               entity='hyeonah_kim',
               name=training_id,
               group='Constructive BC',
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
                         device=device,
                         add_dummy=True)

    print("Start to train", training_id)
    best_cut_mse = float('inf')
    for _ in range(config.train.n_epochs):
        for train_g in dl:
            train_g = train_g.to(device)
            batch_idx = torch.arange(train_g.batch_size, device=device).repeat_interleave(train_g.batch_num_nodes())
            cur_s = []

            for _ in range(train_g.batch_num_nodes().max()):
                train_g, train_y = get_current_gs(gs=train_g, cur_s=cur_s, with_label=True)
                start = perf_counter()
                pred_y, _ = model.get_batch_probs(train_g)

                loss = loss_fn(pred_y, train_y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()
                fit_time = perf_counter() - start

                n_update += 1

                log_dict = {
                    'loss': loss.item(),
                    'fit_time': fit_time
                }

                if n_update % config.train.eval_every == 0:
                    model.eval()
                    start = perf_counter()
                    test_s_mse, test_cut_mse = evaluate(model, g_test[:10], device)
                    eval_time = perf_counter() - start
                    model.train()
                    # log_dict['train_mse'] = train_mse
                    log_dict['test_mse'] = test_s_mse
                    log_dict['test_cut_mse'] = test_cut_mse
                    log_dict['eval_time'] = eval_time
                    # print(log_dict)

                    if test_cut_mse < best_cut_mse:
                        best_cut_mse = test_cut_mse
                        torch.save(model.state_dict(), "./checkpoints/constructive_{}_best.pt".format(training_id))

                if n_update % config.train.save_every == 0:
                    torch.save(model.state_dict(), "./checkpoints/constructive_{}.pt".format(training_id))
                    torch.save(opt.state_dict(), "./checkpoints/opt_constructive_{}.pt".format(training_id))

                wandb.log(log_dict)

                actions = model.get_batch_actions(train_g)
                done = train_g.ndata['is_dummy'][actions]
                cur_s.extend(actions[(1 - done).bool()].tolist())


if __name__ == '__main__':
    config = Box.from_yaml(filename=join(os.getcwd(), 'config', 'autoregressive_config.yaml'))
    # train(config)
    load_and_evaluate(config, "constructive_220603055004_best")
