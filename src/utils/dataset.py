import dgl
import torch
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from src.utils.graph_utils import get_graph_label, get_current_g


class SeqDataset(Dataset):
    def __init__(self, g_list):
        #         assert all(len(g_list) == t.shape[0] for t in tensors), "Size mismatch between inputs"
        #         self.g_list = g_list
        self.g_list, labels = [], []
        num_actions = []
        for g in g_list:
            dummy = g.num_nodes()
            g.add_edges([i for i in range(dummy)], [dummy for _ in range(dummy)])
            g.ndata['is_dummy'] = torch.zeros(g.num_nodes()).int()
            g.ndata['is_dummy'][dummy] = 1

            for m in range(g.ndata['s_hat'].shape[1]):
                s_hat = torch.nonzero(g.ndata['s_hat'][:, m]).view(-1)
                for _ in range(min(len(s_hat) ** 2, 10)):
                    cur_s = []
                    for step in range(len(s_hat) + 1):
                        cur_g = get_current_g(g, cur_s, m)
                        active_nid = torch.nonzero(cur_g.ndata['active']).view(-1)
                        num_actions.append(len(active_nid))
                        label = torch.tensor([float(v in s_hat and v not in cur_s) for v in range(cur_g.num_nodes())])
                        label = label[active_nid]
                        if len(cur_s) < len(s_hat):
                            next_v = np.random.choice(list(set(s_hat.tolist()) - set(cur_s)), 1)[0]
                            cur_s.append(next_v)
                        else:
                            label[-1] = 1.0
                        cur_g.ndata['s_hat'] = torch.zeros((cur_g.ndata['s_hat'].shape[0], 1))
                        self.g_list.append(cur_g)
                        labels.append(label)

        self.labels = labels
        self.num_actions = num_actions
        self.len = len(self.g_list)

    def __getitem__(self, index):
        return self.g_list[index], self.labels[index], self.num_actions[index]

    def __len__(self):
        return self.len


class SeqDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super(SeqDataLoader, self).__init__(*args, **kwargs)

    def collate_fn(self, batch):
        out = tuple(map(list, zip(*batch)))
        gs = dgl.batch(out[0])
        logits, num_actions = [], out[2]
        for l in out[1]:
            logits.extend(l)
        logits = torch.tensor(logits)

        prob_chunks = [l / l.sum() for l in logits.split(num_actions)]
        prob_chunks_padded = pad_sequence(prob_chunks, batch_first=True)

        return gs, prob_chunks_padded


class GraphDataset(Dataset):

    def __init__(self, gs):
        super().__init__()
        self.gs = gs  # not processed
        self.len = len(gs)

    def __getitem__(self, idx):
        return self.gs[idx]

    def __len__(self):
        return self.len


class GraphDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        self.device = kwargs.pop('device', 'cpu')
        self.add_dummy = kwargs.pop('add_dummy', False)

        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        gs = []
        cum_batch_num = 0
        for g in batch:
            batched_g = get_graph_label(g, add_dummy=self.add_dummy)
            batched_g.ndata['batch'] += cum_batch_num
            batched_g.edata['batch'] += cum_batch_num
            gs.append(batched_g)
            cum_batch_num += batched_g.batch_size
        gs = dgl.batch(gs)
        return gs.to(self.device)

