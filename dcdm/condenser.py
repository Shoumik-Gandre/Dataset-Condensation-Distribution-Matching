from typing import Callable, Tuple
import torch
from torch import nn


BlackBoxFn = Callable[[torch.Tensor], torch.Tensor]


class DistributionMatchingCondenser(nn.Module):
    
    def __init__(self, blackbox_fn: BlackBoxFn, num_classes: int, data_per_class: int) -> None:
        super().__init__()
        self._blackbox_fn = blackbox_fn
        self.dataset = nn.Parameter(torch.randn(data_per_class*num_classes))
        self.data_per_class = data_per_class
        self.num_classes = num_classes
    
    def forward(self, input: torch.Tensor, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._blackbox_fn(input), self._blackbox_fn(self.labelwise_index_dataset(label))

    def labelwise_index_dataset(self, label: int):
        return self.dataset[label * self.data_per_class : (label + 1) * self.data_per_class]
    
    @property
    def blackbox_fn(self):
        return self._blackbox_fn
    
    @blackbox_fn.setter
    def blackbox_fn(self, value: BlackBoxFn):
        self._blackbox_fn = value