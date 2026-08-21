import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=32, hidden_layers=2, num_classes=10):
        super(SimpleNN, self).__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        self.input_size = input_size

        layers = []
        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Additional hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, num_classes))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.net(x)

