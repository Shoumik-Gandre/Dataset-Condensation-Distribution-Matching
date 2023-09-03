from typing import Tuple
import torch
from torch.utils.data import Dataset, Subset


ClassificationItem = Tuple[torch.Tensor, torch.Tensor]


def subset_labelwise(dataset: Dataset, num_labels: int) -> Tuple[Subset[ClassificationItem]]:
    """Classwise subset is a tuple that can be indexed by a label 
    to obtain the subset of the features with those labels."""
    
    # record the indexes by class in the dataset
    indexes_by_class = [[] for _ in range(num_labels)]
    for index, (_, label) in enumerate(dataset): # type: ignore
        indexes_by_class[int(label)].append(index)
    
    # generate classwise subsets with the above indexes
    labelwise_subsets = tuple(
        Subset(dataset, indexes_by_class[label])
        for label in range(num_labels)
    )

    return labelwise_subsets