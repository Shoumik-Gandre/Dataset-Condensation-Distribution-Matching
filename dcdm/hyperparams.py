from dataclasses import dataclass


@dataclass
class DistributionMatchingHyperparameters:
    iterations: int
    batch_size: int
    lr_dataset: float
    momentum_dataset: float
    batchnorm_batchsize_perclass: int
    ipc: int