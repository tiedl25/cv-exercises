import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self):
        # START TODO #################
        # see model description in exercise pdf
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        #relu
        #pool

        self.vect = nn.Flatten()

        self.fully1 = nn.Linear(in_features=400, out_features=120)
        #relu

        self.fully2 = nn.Linear(in_features=120, out_features=84)
        #relu

        self.fully3 = nn.Linear(in_features=84, out_features=10)

        self.layers = [
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16*2*2, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
        ]

        # END TODO #################
    
    #def to(self, device):
    #    for layer in self.layers:
    #        layer.to(device)

    def forward(self, x):
        # START TODO #################
        # see model description in exercise pdf
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = self.vect(x)

        x = self.relu(self.fully1(x))

        x = self.relu(self.fully2(x))

        return self.fully3(x)
        # END TODO #################
