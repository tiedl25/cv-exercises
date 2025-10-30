"""Model for CIFAR10."""

import torch as th
from torch import nn


class ConvModel(nn.Module):
    def __init__(self, input_channels: int, num_filters: int, verbose: bool = False):
        """
        Model definition.

        Args:
            input_channels: Number of input channels, this is 3 for the RGB images in CIFAR10
            num_filters: Number of convolutional filters
        """
        super().__init__()
        self.verbose = verbose
        self.num_filters = num_filters
        # first convolutional layer

        # START TODO #################
        # Time to define our network. Hints:
        #   Make each module (i.e. component of the network) an attribute of the class, i.e.
        #   self.conv1 = nn.Conv2d(...)
        #   This is necessary for PyTorch to work correctly.
        # Define the network as follows:
        # 1) Convolution layer with input_channels, output_channels as num_filters,
        #     kernel size 3, stride 2, padding 1, followed by batch norm (nn.BatchNorm2d) and relu (nn.ReLU).
        # 2) Another conv layer with input_channels as num_filters, output_channels as 2 * num_filters,
        #     kernel_size 3, stride 1, padding 1, followed by another batch norm and relu.
        # 3) Averagepooling (nn.AvgPool2d) with kernel size 16, stride 16.
        # 4) Linear layer (nn.Linear) with input_features=2 * num_filters, output_features=10.

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_filters, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=2*num_filters, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(2*num_filters)

        self.avgPool = nn.AvgPool2d(kernel_size=16, stride=16)

        self.lin = nn.Linear(in_features=2*num_filters, out_features=10)

        # END TODO ###################

    def forward(self, x: th.Tensor):
        """
        Model forward pass.

        Args:
            x: Model input, shape [batch_size, in_c, in_h, in_w]

        Returns:
            Model output, shape [batch_size, num_classes]
        """
        if self.verbose:
            print(f"Input shape: {x.shape}")
        # START TODO #################
        # Apply first convolutional layer, batch norm and relu.

        x = self.relu(self.norm1(self.conv1(x)))

        # END TODO ###################
        if self.verbose:
            print(f"Shape after first layer: {x.shape}")
        # START TODO #################
        # Apply second convolutional layer, batch norm and relu

        x = self.relu(self.norm2(self.conv2(x)))

        # END TODO ###################
        if self.verbose:
            print(f"Shape after second layer: {x.shape}")
        # START TODO #################
        # Apply averagepool

        x = self.avgPool(x)

        # END TODO ###################
        if self.verbose:
            print(f"Shape after averagepool: {x.shape}")

        # Here we reshape the input to 2D shape (batch_size, 2 * num_filters)
        # so we can apply a linear layer.
        x = th.reshape(x, (-1, 2 * self.num_filters))
        if self.verbose:
            print(f"Shape after reshape: {x.shape}")

        # START TODO #################
        # Apply the linear.

        x = self.lin(x)

        # END TODO ###################
        if self.verbose:
            print(f"Model output shape: {x.shape}")
        return x
