import numpy as np
import torch.nn as nn

from agents.parallel_trpo.parallel_trpo import utils


class VF(object):
    coeffs = None

    def __init__(self, input_shape):
        hidden_size = 64
        print('*********[VF] variable sharing across processes? ************')
        # with tf.variable_scope("VF"):
        # h1, _ = utils.make_fully_connected("h1", self.x, hidden_size)
        # h2, _ = utils.make_fully_connected("h2", h1, hidden_size)
        # h3, _ = utils.make_fully_connected("h3", h2, 1, final_op=None)
        # self.net = torch.reshape(h3, (-1,))
        # self.session.run(tf.global_variables_initializer())

        print('*********[VF] initialize vars as in utils.make_fully_connected method ************')
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Flatten(dim=1)
        )
        self.optim = torch.optim.Adam()

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        act = path["action_dists"].astype('float32')
        length = len(path["rewards"])
        al = np.arange(length).reshape(-1, 1) / 10.0
        ret = np.concatenate([o, act, al, np.ones((length, 1))], axis=1)
        return ret

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        if self.net is None:
            self.create_net(featmat.shape[1])
        returns = np.concatenate([path["returns"] for path in paths])
        for _ in range(50):
            self.train(featmat, returns)

    def train(self, x, y):
        ret = self.net(x)
        l2 = torch.nn.MSELoss()(ret, y)
        self.optim.zero_grad()
        l2.backward()
        self.optim.step()

    def predict(self, path):
        if self.net is None:
            return np.zeros(len(path["rewards"]))
        else:
            with torch.no_grad():
                ret = self.net(self._features(path)).numpy()
            return np.reshape(ret, (ret.shape[0],))


class LinearVF(object):
    coeffs = None

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        length = len(path["rewards"])
        al = np.arange(length).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, np.ones((length, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, path):
        if self.coeffs is None:
            return np.zeros(len(path["rewards"]))
        else:
            return self._features(path).dot(self.coeffs)
