import torch.nn as nn


class GenerateNet(nn.Module):
    def __init__(self, n_input, layers, n_output):
        super(GenerateNet, self).__init__()
        self.layers = []
        fc1 = nn.Linear(n_input, layers[0])
        self.layers.append(fc1)
        self.layers.append(nn.Tanh())

        for i in range(1, len(layers)):
            fc_temp = nn.Linear(layers[i - 1], layers[i])
            self.layers.append(fc_temp)
            self.layers.append(nn.Tanh())

        self.layers.append(nn.Linear(layers[-1], n_output))
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)

    def init_weights(self):
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)


