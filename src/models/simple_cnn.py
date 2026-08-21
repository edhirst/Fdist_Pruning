import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, img_size=28):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # two stride-2 pools halve the spatial size twice
        self.flat_size = 64 * (img_size // 4) ** 2
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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