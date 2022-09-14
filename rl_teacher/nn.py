import numpy as np
import torch
import torch.nn as nn


class FullyConnectedMLP(object):
    """Vanilla two hidden layer multi-layer perceptron"""

    def __init__(self, obs_shape, act_shape, h_size=64):
        input_dim = np.prod(obs_shape) + np.prod(act_shape)

        print('*********[FullyConnectedMLP] torch leaky relu default alpha is diff from tf one **************')
        self.model = nn.Sequential(
            nn.Linear(input_dim, h_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_size, h_size),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(h_size, 1)
        )

    def run(self, obs, act):
        obs = torch.from_numpy(obs).float()
        act = torch.from_numpy(act).float()
        flat_obs = torch.flatten(obs, start_dim=1)
        x = torch.cat([flat_obs, act], dim=1)
        return self.model(x)

    @property
    def parameters(self):
        return self.model.parameters()
