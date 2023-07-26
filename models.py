import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, sizes, nonLinearFunc):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(nonLinearFunc)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



