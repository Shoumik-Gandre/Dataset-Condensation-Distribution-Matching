import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from dcdm.condenser import DistributionMatchingCondenser
from dcdm.hyperparams import DistributionMatchingHyperparameters


class CondenserTrainer:

    def __init__(self, dm_condenser: DistributionMatchingCondenser, train_dataset: Dataset, eval_dataset: Dataset, hyerparams: DistributionMatchingHyperparameters, device: torch.device) -> None:
        self.dm_condenser = dm_condenser
        self.optimizer_dataset = optim.SGD(dm_condenser.parameters(), lr=hyerparams.lr_dataset, momentum=0.5) # optimizer_img for synthetic data
        self.hyperparams = hyerparams
        self.device = device
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def get_train_dataloader(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.hyperparams.batch_size, shuffle=True)
    
    def train(self):
        for self.epoch in range(1, 1+self.hyperparams.iterations):
            self.training_step()
            self.evaluation_step()
    
    def training_step(self) -> None:
        loss = torch.tensor(0.0).to(self.device)
        for c in range(self.num_classes):
            input_real = self.get_input(c)
            out_real, out_syn = self.dm_condenser(input_real, c)
            out_real = out_real.detach()

            loss += torch.sum((torch.mean(out_real, dim=0) - torch.mean(out_syn, dim=0))**2)
        
        self.optimizer_dataset.zero_grad()
        loss.backward()
        self.optimizer_dataset.step()
    
    def evaluation_step(self) -> None:
        # TODO: Train a copy of the network
        ...