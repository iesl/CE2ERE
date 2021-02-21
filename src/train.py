import torch
from typing import Dict

from torch.nn import Module
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model: Module, epochs: int, learning_rate: float, train_dataloader: DataLoader,
                 valid_dataloader_dict: Dict[str, DataLoader], test_dataloader_dict: Dict[str, DataLoader],
                 opt: torch.optim.Optimizer, roberta_size_type="roberta-base"):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_dataloader = train_dataloader
        self.valid_dataloader_dict = valid_dataloader_dict
        self.test_dataloader_dict = test_dataloader_dict

        self.opt = opt

        self.roberta_size_type = roberta_size_type
        if self.roberta_size_type == "roberta-base":
            self.roberta_dim = 768
        else:
            self.roberta_dim = 1024

    def train(self):
        for epoch in range(1, self.epochs+1):
            print(epoch)
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                print(type(batch[3]), type(batch[6]), type(batch[12]), type(batch[15]))
                print("batch[3]:", batch[3])
                print("batch[6]:", batch[6])
                print("batch[12]:", batch[12])
                print("batch[15]:", batch[15])

