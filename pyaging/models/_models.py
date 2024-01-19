import torch
import torch.nn as nn
import torch.nn.functional as F
from ._base_models import *

class AltumAge(pyagingModel):
    def __init__(self):
        super().__init__()

    def preprocess(self, x):
        x = (x - self.preprocess_dependencies[0])/self.preprocess_dependencies[1]
        return x

    def postprocess(self, x):
        return x
