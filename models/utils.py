import torch.nn as nn


def get_fc_layers(in_size, hidden_sizes, out_size):
    layers = [nn.Linear(in_size, hidden_sizes[0]), nn.ReLU()]
    for s in hidden_sizes:
        layers.append(nn.Linear(s, s))
        layers.append(nn.ReLU())
    layers.append((nn.Linear(hidden_sizes[-1], out_size)))
    return nn.Sequential(*layers)
