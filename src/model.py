import torch

from transformers import RobertaModel
from typing import Dict
from torch.nn import Module, CrossEntropyLoss, Linear, LeakyReLU


class MLP(Module):
    def __init__(self, hidden_size: int, mlp_size: int, num_classes: int):
        super().__init__()
        self.fc1 = Linear(hidden_size * 4, mlp_size * 2)
        self.fc2 = Linear(mlp_size * 2, num_classes)
        self.leaky_relu = LeakyReLU(0.2, True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        return x


class RoBERTa_MLP(Module):
    def __init__(self, num_classes: int, data_type: str, lambda_dict: Dict[str, float], mlp_size: int, hidden_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.data_type = data_type
        self.lambdas = lambda_dict
        self.hidden_size = hidden_size
        self.model = RobertaModel.from_pretrained('roberta-large')

        hier_weights, temp_weights = self._get_weights()
        self.hier_class_weights = torch.FloatTensor(hier_weights)
        self.hier_class_weights = torch.FloatTensor(hier_weights)

        self.MLP = MLP(hidden_size, mlp_size, num_classes)

        # self.hieve_anno_loss = CrossEntropyLoss(weight=self.hier_class_weights)
        # self.matres_anno_loss = CrossEntropyLoss(weight=self.temp_class_weights)
        # self.hier_transitivity_loss = _hier_transitivity_loss()
        # self.temp_transitivity_loss = _temp_transitivity_loss()
        # self.cross_category_loss = _cross_category_loss()

    def _get_weights(self):
        HierPC = 1802.0
        HierCP = 1846.0
        HierCo = 758.0
        HierNo = 63755.0
        HierTo = HierPC + HierCP + HierCo + HierNo  # total number of event pairs
        hier_weights = [0.25 * HierTo / HierPC, 0.25 * HierTo / HierCP, 0.25 * HierTo / HierCo, 0.25 * HierTo / HierNo]
        temp_weights = [0.25 * 818.0 / 412.0, 0.25 * 818.0 / 263.0, 0.25 * 818.0 / 30.0, 0.25 * 818.0 / 113.0]
        return hier_weights, temp_weights


    def forward(self):
        print("check")