import os
import pickle
from itertools import combinations
# import pandas as pd
from pathlib import Path

import dgl
import numpy as np
import ray
import torch
import tsplib95
from dgl.data.utils import save_graphs
from julia import Main
from tqdm import tqdm


def get_xml_data():
    # XML CVRP data is generated following Uchoa et al. (2021)
    # Queiroga, Eduardo, et al. "10,000 optimal CVRP solutions for testing machine learning based heuristics."
    # AAAI-22 Workshop on Machine Learning for Operations Research (ML4OR). 2021.
    path = os.path.join(os.pardir, os.pardir, 'data', 'instances')
    file_list = os.listdir(path)
    instances = [file.split('.')[0] for file in file_list if
                 file.startswith('random-5') and not file.startswith('random-5-')]
    Main.include("../jl/call_julia.jl")

    for instance in tqdm(instances):
        file = "../../data/pickles/{}.pickle".format(instance)
        exist = os.path.isfile(file)
        # if not instance.startswith("XML100_112"):
        #     continue

        if exist:
            print("already done.", instance)
            continue

        vrp_path = os.path.join(path, instance + '.vrp')
        # sol_path = os.path.join(path, instance + '.sol')
        sol_path = os.path.join(path, os.pardir, 'solutions', instance + '.sol')
        problem, coords, demands = parse_vrp_file(vrp_path)
        tours, opt, _ = parse_sol_file(sol_path)
        k = len(tours)
        cvrp, lb, list_e, list_x, list_s, list_rhs, list_z = Main.get_cvrp_data(problem, coords, demands, k, opt)

        data = {
            'name': problem.name,
            'k': k,
            'lb': lb,
            'list_e': list_e,
            'list_x': list_x,
            'list_s': list_s,
            'list_rhs': list_rhs,
            'list_z': list_z,
            'demand': cvrp.demand,
            'capacity': cvrp.capacity
        }

        # save
        with open("../../data/pickles/{}.pickle".format(problem.name.split('.')[0]), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def parse_vrp_file(file_path: str):
    if not file_path.endswith('.vrp'):
        # file_path = file_path.split('.')[0]
        file_path += '.vrp'
    problem = tsplib95.loaders.load(file_path)

    coords = np.array([v for v in problem.node_coords.values()])
    coords = np.concatenate([coords, np.expand_dims(coords[problem.depots[0]], axis=0)], axis=0)
    demands = np.array([v for v in problem.demands.values()] + [0])
    return problem, coords, demands


def parse_sol_file(file_path: str):
    if not file_path.endswith('.sol'):
        # file_path = file_path.split('.')[0]
        file_path += '.sol'

    tours = []
    cost = None
    run_time = None

    with open(file_path, 'r') as f:
        while True:
            line = f.readline()
            if line.startswith('Route'):
                # tours.append([int(st) for st in line.split(':')[1].split(' ') if st.isnumeric()])
                tours.append([int(st) for st in line.split(':')[1].split() if st.isnumeric()])
            if line.startswith('Cost'):
                cost = float(line.split(' ')[1])
            if line.startswith('Time'):
                run_time = float(line.split(' ')[1])
            if not line:
                break
    return tours, cost, run_time


def get_julia_data(newly_cal=True, name=None):
    if newly_cal:
        Main.include("../jl/call_julia.jl")
        cvrp, lb, list_e, list_x, list_s, list_rhs, list_z = Main.experiment(name)

        data = {
            'name': name,
            'k': int(name.split("-k")[1]),
            'lb': lb,
            'list_e': list_e,
            'list_x': list_x,
            'list_s': list_s,
            'list_rhs': list_rhs,
            'list_z': list_z,
            'demand': cvrp.demand,
            'capacity': cvrp.capacity
        }

        # print(Main.cvrp)

        # save
        with open("../../data/pickles/{}_m.pickle".format(name), 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    else:
        # load
        with open("../../data/pickles/{}.pickle".format(name), 'rb') as f:
            data = pickle.load(f)
    return data


def save_graphs_sols(complete_g=True, file_name="data"):
    path = os.path.join(os.pardir, os.pardir, 'data', 'pickles')
    instances = os.listdir(path)
    size = 200
    instances = [file for file in instances if file.endswith('220311092127.pickle')]

    print(len(instances))
    gs, labels = [], []  # no need to save rhs
    for instance in tqdm(instances):
        file = "../../data/pickles/{}".format(instance)
        if os.path.isfile(file) and not instance.startswith('E'):
            data = get_julia_data(newly_cal=False, name=instance.split('.')[0])
            k = data['k']
            capacity = data['capacity']
            # cust_n = len(data['demand'][data['demand'] > 0.0])

            n = len(data['demand']) - 1  # dummy
            demand = data['demand'][:n]
            if sum(demand == 0) > 1:
                # print(instance, "has 0 demand.")
                demand[demand == 0] = 1
                demand[0] = 0
                # continue

            for step in range(len(data['list_e'])):
                e = data['list_e'][step]
                x = data['list_x'][step]
                s_hat = data['list_s'][step]

                # based on x_bar
                cur_e = [(u - 1, v - 1) for u, v in e]

                src, dest, ef = [], [], []
                for idx, (u, v) in enumerate(cur_e):
                    src.extend([u, v])
                    dest.extend([v, u])
                    ef.extend([x[idx], x[idx]])

                if complete_g:
                    for (u, v) in combinations(range(n), 2):
                        if (u, v) not in cur_e and (v, u) not in cur_e:
                            src.extend([u, v])
                            dest.extend([v, u])
                            ef.extend([0.0, 0.0])

                g = dgl.graph((torch.tensor(src).long(), torch.tensor(dest).long()))

                g.edata['x_val'] = torch.tensor(ef).view(-1, 1)
                g.ndata['demand'] = torch.tensor(demand).view(-1, 1)
                g.ndata['capacity'] = torch.tensor([capacity] * n).view(-1, 1)
                g.ndata['is_depot'] = torch.tensor(demand == 0.0).int().view(-1, 1)

                # dummy = g.num_nodes()
                # g.add_edges([i for i in range(dummy)], [dummy for _ in range(dummy)])
                # g.ndata['is_dummy'] = torch.zeros(g.num_nodes()).int()
                # g.ndata['is_dummy'][dummy] = 1

                # solution label (S)
                label = []
                for m in range(len(s_hat)):
                    # label[m][dummy] = 0
                    # label.append([int(i + 1 in s_hat[m]) for i in range(n + 1)])
                    label.append([int(i + 1 in s_hat[m]) for i in range(n)])

                g.ndata['s_hat'] = torch.tensor(label).T  # [N X K]
                gs.append(g)  # (steps)
    print(len(gs))
    save_graphs("../../data/{}.bin".format(file_name), gs)


if __name__ == '__main__':
    get_xml_data()
    # save_graphs_sols(False, "data_random_large")
