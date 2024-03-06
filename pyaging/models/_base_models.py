from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class pyagingModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.metadata = {
            "clock_name": None,
            "data_type": None,
            "species": None,
            "year": None,
            "approved_by_author": None,
            "citation": None,
            "doi": None,
            "notes": None,
            "research_only": None,
            "version": None,
        }

        self.reference_values = None

        self.preprocess_name = None
        self.preprocess_dependencies = None

        self.postprocess_name = None
        self.postprocess_dependencies = None

        self.features = None
        self.base_model_features = self.features

        self.base_model = None

    def forward(self, x):
        x = self.preprocess(x)
        x = self.base_model(x)
        x = self.postprocess(x)
        return x

    @abstractmethod
    def preprocess(self, x):
        """
        Preprocess the input data. This method should be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def postprocess(self, x):
        """
        Postprocess the model output. This method should be implemented by all subclasses.
        """
        pass


class LinearModel(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize a simple linear model.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Define a linear layer

    def forward(self, x):
        x = self.linear(x)
        return x


class PCLinearModel(nn.Module):
    def __init__(self, input_dim, pc_dim):
        """
        Initialize a Principal Component Linear Model.
        """
        super().__init__()
        self.center = nn.Parameter(torch.empty(input_dim), requires_grad=False)
        self.rotation = nn.Parameter(torch.empty((input_dim, pc_dim)), requires_grad=False)
        self.linear = nn.Linear(pc_dim, 1)

    def forward(self, x):
        x = x - self.center  # Apply centering
        x = torch.mm(x, self.rotation)  # Apply PCA rotation
        x = self.linear(x)
        return x


class AltumAgeNeuralNetwork(nn.Module):
    def __init__(self):
        """
        Initialize the AltumAge model with multiple linear layers and batch normalization.
        """
        super().__init__()

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
