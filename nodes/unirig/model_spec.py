from abc import ABC, abstractmethod
import torch.nn as nn


class ModelSpec(nn.Module, ABC):

    @abstractmethod
    def __init__(self):
        super().__init__()
