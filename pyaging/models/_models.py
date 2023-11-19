import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize a simple linear model.

        Args:
        - input_dim: Dimension of input features.
        """
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Define a linear layer

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input data.

        Returns:
        - Output of the linear layer.
        """
        x = self.linear(x)
        return x


import torch.nn as nn
import torch.nn.functional as F


class AltumAge(nn.Module):
    def __init__(self):
        """
        Initialize the AltumAge model with multiple linear layers and batch normalization.
        """
        super(AltumAge, self).__init__()

        # Define the linear layers
        self.linear1 = nn.Linear(20318, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 32)
        self.linear5 = nn.Linear(32, 32)
        self.linear6 = nn.Linear(32, 1)

        # Define batch normalization layers
        self.bn1 = nn.BatchNorm1d(20318, eps=0.001, momentum=0.99)
        self.bn2 = nn.BatchNorm1d(32, eps=0.001, momentum=0.99)
        self.bn3 = nn.BatchNorm1d(32, eps=0.001, momentum=0.99)
        self.bn4 = nn.BatchNorm1d(32, eps=0.001, momentum=0.99)
        self.bn5 = nn.BatchNorm1d(32, eps=0.001, momentum=0.99)
        self.bn6 = nn.BatchNorm1d(32, eps=0.001, momentum=0.99)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input data.

        Returns:
        - Output after passing through layers and activation functions.
        """
        x = self.bn1(x)
        x = self.linear1(x)
        x = F.selu(x)

        x = self.bn2(x)
        x = self.linear2(x)
        x = F.selu(x)

        x = self.bn3(x)
        x = self.linear3(x)
        x = F.selu(x)

        x = self.bn4(x)
        x = self.linear4(x)
        x = F.selu(x)

        x = self.bn5(x)
        x = self.linear5(x)
        x = F.selu(x)

        x = self.bn6(x)
        x = self.linear6(x)

        return x


import torch
import torch.nn as nn


class PCARDModel(nn.Module):
    def __init__(self, input_dim, pc_dim):
        """
        Initialize a PCA Rotation and Linear Discriminant (PCARD) model.

        Args:
        - input_dim: Dimension of input features.
        - pc_dim: Dimension of PCA rotation.
        """
        super(PCARDModel, self).__init__()
        self.rotation = nn.Parameter(
            torch.empty((input_dim, pc_dim)), requires_grad=False
        )
        self.linear = nn.Linear(pc_dim, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input data.

        Returns:
        - Output after PCA rotation and linear layer.
        """
        x = torch.mm(x, self.rotation)  # Apply PCA rotation
        x = self.linear(x)
        return x


import torch
import torch.nn as nn


class PCLinearModel(nn.Module):
    def __init__(self, input_dim, pc_dim):
        """
        Initialize a Principal Component Linear Model.

        Args:
        - input_dim: Dimension of input features.
        - pc_dim: Dimension of PCA rotation.
        """
        super(PCLinearModel, self).__init__()
        self.center = nn.Parameter(torch.empty(input_dim), requires_grad=False)
        self.rotation = nn.Parameter(
            torch.empty((input_dim, pc_dim)), requires_grad=False
        )
        self.linear = nn.Linear(pc_dim, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input data.

        Returns:
        - Output after centering, PCA rotation, and linear layer.
        """
        x = x - self.center  # Apply centering
        x = torch.mm(x, self.rotation)  # Apply PCA rotation
        x = self.linear(x)
        return x


import torch
import torch.nn as nn


class PCGrimAge(nn.Module):
    def __init__(self, input_dim, pc_dim, comp_dims=[]):
        """
        Initialize a PC Grim Age model.

        Args:
        - input_dim: Dimension of input features.
        - pc_dim: Dimension of PCA rotation.
        - comp_dims: List of component dimensions for linear layers.
        """
        super(PCGrimAge, self).__init__()

        self.center = nn.Parameter(torch.empty(input_dim), requires_grad=False)
        self.rotation = nn.Parameter(
            torch.empty((input_dim, pc_dim)), requires_grad=False
        )

        # Define linear layers for each component dimension
        step1_layers = [nn.Linear(comp_dim, 1) for comp_dim in comp_dims]
        self.step1_layers = nn.ModuleList(step1_layers)

        # Create and store parameters for features
        self.step1_features = []
        for idx, comp_dim in enumerate(comp_dims):
            feature_param = nn.Parameter(torch.empty(comp_dim), requires_grad=False)
            self.step1_features.append(feature_param)
            setattr(self, f"step1_feature{idx}", feature_param)

        self.step2 = nn.Linear(len(comp_dims) + 2, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        - x: Input data.

        Returns:
        - Output after various transformations and linear layers.
        """
        # Extract and separate age and gender features
        age = x[:, -1]
        female = x[:, -2]
        x = x[:, :-2]

        # Apply centering and PCA rotation
        x = x - self.center
        x = torch.mm(x, self.rotation)

        # Concatenate transformed features, gender, and age
        x = torch.cat((x, female.unsqueeze(1), age.unsqueeze(1)), dim=1)

        # Process through linear layers
        xs = [
            layer(x[:, features.long()])[:, 0]
            for features, layer in zip(self.step1_features, self.step1_layers)
        ]
        x = torch.stack(xs, dim=1)

        # Add gender and age again for final linear layer
        x = torch.cat((x, age.unsqueeze(1), female.unsqueeze(1)), dim=1)
        x = self.step2(x)

        return x
