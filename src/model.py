import torch
from torch import Tensor
from transformers import RobertaModel
from typing import Dict, Tuple

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
    def __init__(self, num_classes: int, data_type: str, mlp_size: int, hidden_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.data_type = data_type
        self.hidden_size = hidden_size
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.MLP = MLP(hidden_size, mlp_size, num_classes)

    def _get_embeddings_from_position(self, roberta_embd: Tensor, position: Tensor):
        batch_size = position.shape[0]
        return torch.cat([roberta_embd[i, position[i], :].unsqueeze(0) for i in range(0, batch_size)], 0)

    def _get_relation_representation(self, tensor1: Tensor, tensor2: Tensor):
        sub = torch.sub(tensor1, tensor2)
        mul = torch.mul(tensor1, tensor2)
        return torch.cat((tensor1, tensor2, sub, mul), 1)

    def forward(self, batch: Tuple[torch.Tensor]):
        x_sntc, y_sntc, z_sntc = batch[3], batch[4], batch[5]
        x_position, y_position, z_position = batch[6], batch[7], batch[8]

        # Produce contextualized embeddings using RobertaModel for all tokens of the entire document
        # get embeddings corresponding to x_position number
        output_x = self._get_embeddings_from_position(self.roberta_model(x_sntc)[0], x_position) # shape: [16, 120, 1024]
        output_y = self._get_embeddings_from_position(self.roberta_model(y_sntc)[0], y_position)
        output_z = self._get_embeddings_from_position(self.roberta_model(z_sntc)[0], z_position)

        # For each event pair (e1; e2), the contextualized features are obtained as the concatenation of h_e1 and h_e2,
        # along with their element-wise Hadamard product and subtraction.
        alpha_representation = self._get_relation_representation(output_x, output_y)
        beta_representation = self._get_relation_representation(output_y, output_z)
        gamma_representation = self._get_relation_representation(output_x, output_z)

        # alpha_logits: [16, 8]
        alpha_logits = self.MLP(alpha_representation)
        beta_logits = self.MLP(beta_representation)
        gamma_logits = self.MLP(gamma_representation)

        return alpha_logits.float(), beta_logits.float(), gamma_logits.float()