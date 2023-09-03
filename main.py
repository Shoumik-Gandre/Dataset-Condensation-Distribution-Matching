from dataclasses import dataclass
import logging
import time
from argparse import Namespace
from pathlib import Path
from typing import Mapping, Union

import hydra
import torch
from omegaconf import DictConfig
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models, transforms

from dcdm.dataset_init import RandomStratifiedInitStrategy
from dcdm.hyperparams import DistributionMatchingHyperparameters
from dcdm.model_init import HomogenousModelInitStrategy, ModelInitStrategy
from dcdm.synthesizer import DistributionMatchingSynthesizer
from networks import ConvNetEmbedder, ConvNet
from utils import catchtime, augment
from torchmetrics import Accuracy, MeanMetric

log = logging.getLogger(__name__)


def evaluate_synset(net: nn.Module, images_train: torch.Tensor, labels_train: torch.Tensor, 
                    testloader: DataLoader, 
                    device: torch.device, 
                    lr: float, num_epochs: int, batch_size: int,
                    dc_aug_param: Mapping[str, Union[str, float, int]],
                    num_classes: int):
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=num_epochs//2 + 1, gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = DataLoader(dst_train, batch_size=batch_size, shuffle=True, num_workers=0)


    with catchtime() as t:
        average_loss_train = MeanMetric()
        metric_train = Accuracy(task='multiclass', num_classes=num_classes)

        for ep in range(1, 1+num_epochs):
            # Train Step
            net.train()
            metric_train = Accuracy(task='multiclass', num_classes=num_classes)
            average_loss_train = MeanMetric()
            for inputs, labels in trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = augment(inputs, dc_aug_param, device=device)
                outputs: torch.Tensor = net(inputs)
                loss: torch.Tensor = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                average_loss_train(loss.item())
                metric_train(outputs.argmax(1).cpu(), labels.cpu())

            lr_scheduler.step()

    # Eval Step
    net.eval()
    with torch.no_grad():
        metric_eval = Accuracy(task='multiclass', num_classes=num_classes)
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            
            metric_eval(outputs.argmax(1).cpu(), labels.cpu())

    lr_scheduler.step()
    
    return average_loss_train.compute(), metric_train.compute(), metric_eval.compute()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    log.info("Started")
    # Create dataset root if it doesn't exist.
    Path(config.files.dataset_root).mkdir(parents=True, exist_ok=True)

    # Config
    device = torch.device(config.device)
    dimensions = config.dcdm.dimensions
    hyperparams = DistributionMatchingHyperparameters(**config.dcdm.hyperparams)

    # Datasets:
    log.info("Started Configuring Datasets ...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(config.files.dataset_root, train=True, download=True, transform=transform)
    real_images = torch.stack([train_data[0] for train_data in train_dataset])
    real_labels = torch.tensor([train_data[1] for train_data in train_dataset], dtype=torch.long)
    train_dataset = TensorDataset(real_images, real_labels) # type: ignore

    eval_dataset = datasets.MNIST(config.files.dataset_root, train=False, download=True, transform=transform)
    log.info("Ended Configuring Datasets")


    dataset_init_strategy = RandomStratifiedInitStrategy(dimensions=dimensions, num_classes=10, ipc=hyperparams.ipc, device=device)

    synthesizer = DistributionMatchingSynthesizer(
        dimensions=dimensions,
        num_labels=10,
        dataset=train_dataset,
        device=device,
        dataset_init_strategy=dataset_init_strategy,
        model_init_strategy=HomogenousModelInitStrategy(
            model_class=ConvNetEmbedder, 
            model_args={
                'channel': 1, 
                'num_classes': 10, 
                'net_width': 128, 
                'net_depth': 3, 
                'net_act': 'relu', 
                'net_norm': 'instancenorm', 
                'net_pooling': 'avgpooling', 
                'im_size': dimensions[1:]
            }
        ),
        hyperparams=hyperparams
    )

    log.info("Synthesizing Synthetic Dataset")
    dataset = synthesizer.synthesize()
    log.info("Synthesized Synthetic Dataset")

    dataset = TensorDataset(dataset.tensors[0].detach().clone(), dataset.tensors[1].detach().clone())

    train_dataloader = DataLoader(dataset, 256, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, 256)

    for i in range(1):
        model = ConvNet(
            **{
                'channel': 1, 
                'num_classes': 10, 
                'net_width': 128, 
                'net_depth': 3, 
                'net_act': 'relu', 
                'net_norm': 'instancenorm', 
                'net_pooling': 'avgpooling', 
                'im_size': dimensions[1:]
            }
        )
        model = nn.DataParallel(model)
        model = model.to(device)

        avg_loss_train, acc_train, acc_test = evaluate_synset(
            net=model, 
            images_train=dataset.tensors[0], 
            labels_train=dataset.tensors[1], 
            testloader=eval_dataloader, 
            device=torch.device(config.device),
            lr=config.dcdm.eval.lr,
            num_epochs=config.dcdm.eval.num_epochs,
            batch_size=config.dcdm.eval.batch_size,
            dc_aug_param = {
                'crop': 4,
                'scale': 0.2,
                'rotate': 45,
                'noise': 0.001,
                'strategy': 'crop_scale_rotate',
            },
            num_classes=10
        )
        log.info(f'Evaluate: train loss = {avg_loss_train:.6f} train acc = {acc_train:.4f}, test acc = {acc_test:.4f}')



# def evaluate_synset(net, images_train, labels_train, testloader, args: Namespace):
#     net = net.to(args.device)
#     images_train = images_train.to(args.device)
#     labels_train = labels_train.to(args.device)
#     lr = float(args.lr_net)
#     Epoch = int(args.epoch_eval_train)
#     lr_schedule = [Epoch//2+1]
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
#     criterion = nn.CrossEntropyLoss().to(args.device)

#     dst_train = TensorDataset(images_train, labels_train)
#     trainloader = DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

#     start = time.time()
#     for ep in range(Epoch+1):
#         loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug = True)
#         if ep in lr_schedule:
#             lr *= 0.1
#             optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

#     time_train = time.time() - start
#     loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug = False)
#     log.info(f'Evaluate: epoch = {Epoch} train time = {int(time_train)}s train loss = {loss_train :.6f} train acc = {acc_train:.4f}, test acc = {acc_test:.4f}')

#     return net, acc_train, acc_test




if __name__ == '__main__':
    main()

