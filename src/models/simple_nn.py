import torch.nn as nn
import torch.nn.functional as F
    
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
    
    
    def compile(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_model(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()


# replace num_class with config=config 
# config will be a dict 
# replace all the numbers hard coded in into a for loop to come from the config file 
# ask claude to write a yaml importing file 
# in main training scripthave a line that calls the config and parse it that's been imported 
# have all yaml files into 1 