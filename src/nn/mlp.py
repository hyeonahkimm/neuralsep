from typing import List
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden: List[int] = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity',
                 bias: bool = True):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = hidden
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + hidden
        output_dims = hidden + [output_dim]

        self.layers = nn.ModuleList()
        for (in_dim, out_dim) in zip(input_dims, output_dims):
            self.layers.append(nn.Linear(in_dim, out_dim, bias=bias))

    def forward(self, x):
        for linear in self.layers[:-1]:  # hidden layers and hidden_act
            x = linear(x)
            x = self.hidden_act(x)

        # last linear layer and out_act
        x = self.layers[-1](x)
        x = self.out_act(x)

        return x
