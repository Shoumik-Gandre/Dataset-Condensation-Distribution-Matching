from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn, optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, SubsetRandomSampler, Subset
from tqdm import tqdm
from dcdm.dataset_init import DatasetInitStrategy
from dcdm.model_init import ModelInitStrategy
from dcdm.hyperparams import DistributionMatchingHyperparameters


@dataclass
class DistributionMatchingSynthesizer:
    dimensions: Tuple[int, ...]
    num_labels: int
    dataset: Dataset[Tuple[torch.Tensor, torch.Tensor]]
    device: torch.device
    dataset_init_strategy: DatasetInitStrategy
    model_init_strategy: ModelInitStrategy
    hyperparams: DistributionMatchingHyperparameters

    def synthesize(self) -> TensorDataset:
        # Generate Synthetic Dataset
        syn_dataset = self.dataset_init_strategy.init()

        # Subset the real dataset labelwise
        real_dataset_labelwise = self._get_labelwise_subsets(dataset=self.dataset, num_labels=self.num_labels)

        # Optimizer used on the synthetic dataset's features
        optimizer_dataset = optim.SGD(params=(syn_dataset.tensors[0],), 
                                            lr=self.hyperparams.lr_dataset, 
                                            momentum=self.hyperparams.momentum_dataset)
        optimizer_dataset.zero_grad()

        # Loop for training synthetic data on multiple model initializations
        for iteration in tqdm(range(self.hyperparams.iterations), desc="iterations", position=0):
            
            # Initialize Model
            model: nn.Module = self.model_init_strategy.init().to(self.device)

            loss = torch.tensor(0.0, device=self.device)
            for label in range(self.num_labels):

                input_real, _ = self.sample_batch_real(real_dataset_labelwise, label)
                input_syn, _ = self.sample_batch_syn(syn_dataset, label)

                output_real = model(input_real.to(self.device)).detach()
                output_syn  = model(input_syn.to(self.device))

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)
                
            optimizer_dataset.zero_grad()
            loss.backward()
            optimizer_dataset.step()
                
        return syn_dataset
    
    def sample_batch_real(self, real_dataset_labelwise, label):
        dataloader_real = DataLoader(
            real_dataset_labelwise[label], 
            batch_size=self.hyperparams.batch_size
        )
        return next(iter(dataloader_real))
    
    def sample_batch_syn(self, syn_dataset, label):
        dataloader_syn = DataLoader(
            syn_dataset, 
            batch_size=self.hyperparams.ipc, 
            sampler=SubsetRandomSampler(
                indices=range(
                    self.hyperparams.ipc * (label), 
                    self.hyperparams.ipc * (label + 1)
                )
            )
        )
        return next(iter(dataloader_syn))

    def _get_labelwise_subsets(self, dataset: Dataset, num_labels: int) -> Tuple[Subset[Tuple[torch.Tensor, torch.Tensor]]]:
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
    