import random
import numpy as np
import torch


def set_seed(seed: int,
             use_cuda: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    # dgl.seed(seed)
