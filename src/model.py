import torch
from torch import Tensor
from transformers import RobertaModel
from typing import Dict, Tuple

from torch.nn import Module, CrossEntropyLoss, Linear, LeakyReLU, LSTM, LogSoftmax


class MLP(Module):
    def __init__(self, hidden_size: int, mlp_size: int, num_classes: int):
        super().__init__()
        self.fc1 = Linear(hidden_size, mlp_size)
        self.fc2 = Linear(mlp_size, num_classes)
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
        if hidden_size == 1024:
            self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        elif hidden_size == 768:
            self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        else:
            raise ValueError(f"roberta_hidden_size={hidden_size} is not supported at this time!")
        self.MLP = MLP(8 * hidden_size, 2 * mlp_size, num_classes)

    def _get_embeddings_from_position(self, roberta_embd: Tensor, position: Tensor):
        batch_size = position.shape[0]
        return torch.cat([roberta_embd[i, position[i], :].unsqueeze(0) for i in range(0, batch_size)], 0)

    def _get_relation_representation(self, tensor1: Tensor, tensor2: Tensor):
        sub = torch.sub(tensor1, tensor2)
        mul = torch.mul(tensor1, tensor2)
        return torch.cat((tensor1, tensor2, sub, mul), 1)

    def forward(self, batch: Tuple[torch.Tensor], device: torch.device):
        x_sntc, y_sntc, z_sntc = batch[3].to(device), batch[4].to(device), batch[5].to(device)
        x_position, y_position, z_position = batch[6].to(device), batch[7].to(device), batch[8].to(device)

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

        return alpha_logits, beta_logits, gamma_logits


class BiLSTM_MLP(Module):
    def __init__(self, num_classes: int, data_type: str, hidden_size: int, num_layers: int, mlp_size: int,
                 lstm_input_size: int, roberta_size_type="roberta-base"):
        super().__init__()
        self.num_classes = num_classes
        self.data_type = data_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp_size = mlp_size
        self.lstm_input_size = lstm_input_size
        self.bilstm = LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.MLP = MLP(8 * hidden_size, 2 * mlp_size, num_classes)

        self.roberta_size_type = roberta_size_type
        self.RoBERTa_layer = RobertaModel.from_pretrained(roberta_size_type)
        if roberta_size_type == "roberta-base":
            self.roberta_dim = 768
        elif roberta_size_type == "roberta-large":
            self.roberta_dim = 1024
        else:
            raise ValueError(f"{roberta_size_type} doesn't exist!")

    def _get_embeddings_from_position(self, lstm_embd: Tensor, position: Tensor):
        batch_size = position.shape[0]
        return torch.cat([lstm_embd[i, position[i], :].unsqueeze(0) for i in range(0, batch_size)], 0)

    def _get_roberta_embedding(self, sntc):
        with torch.no_grad():
            roberta_list = []
            for s in sntc: # s: [120]; [padded_length], sntc: [batch_size, padded_length]
                roberta_embd = self.RoBERTa_layer(s.unsqueeze(0))[0] # [1, 120, 768]
                roberta_list.append(roberta_embd.view(-1, self.roberta_dim)) # [120, 768]
            return torch.stack(roberta_list)

    def _get_relation_representation(self, tensor1: Tensor, tensor2: Tensor):
        sub = torch.sub(tensor1, tensor2)
        mul = torch.mul(tensor1, tensor2)
        return torch.cat((tensor1, tensor2, sub, mul), 1)

    def forward(self, batch: Tuple[torch.Tensor], device: torch.device):
        # x_sntc: [64, 120]; [batch_size, padding_length]; word id information
        x_sntc, y_sntc, z_sntc = batch[3].to(device), batch[4].to(device), batch[5].to(device)
        x_position, y_position, z_position = batch[6].to(device), batch[7].to(device), batch[8].to(device)

        # get RoBERTa embedding
        roberta_x_sntc = self._get_roberta_embedding(x_sntc) #[64, 120, 768];[batch_size, padded_len, roberta_dim]
        roberta_y_sntc = self._get_roberta_embedding(y_sntc)
        roberta_z_sntc = self._get_roberta_embedding(z_sntc)

        # BiLSTM layer
        bilstm_output_A, _ = self.bilstm(roberta_x_sntc) #[batch_size, padded_len, lstm_hidden_dim * 2]; [64, 120, 512]
        bilstm_output_B, _ = self.bilstm(roberta_y_sntc)
        bilstm_output_C, _ = self.bilstm(roberta_z_sntc)

        output_A = self._get_embeddings_from_position(bilstm_output_A, x_position) #[batch_size, lstm_hidden_dim * 2]; [64, 512]
        output_B = self._get_embeddings_from_position(bilstm_output_B, y_position)
        output_C = self._get_embeddings_from_position(bilstm_output_C, z_position)

        # Subtraction + Hadamard
        alpha_repr = self._get_relation_representation(output_A, output_B) # [batch_size, lstm_hidden_dim * 2 * 4][64, 2048]
        beta_repr = self._get_relation_representation(output_B, output_C)
        gamma_repr = self._get_relation_representation(output_A, output_C)
        alpha_reverse_repr = self._get_relation_representation(output_B, output_A)

        # MLP layer
        alpha_logits = self.MLP(alpha_repr) # [batch_size, num_classes]; [64, 8]
        beta_logits = self.MLP(beta_repr)
        gamma_logits = self.MLP(gamma_repr)
        alpha_reverse_logits = self.MLP(alpha_reverse_repr)

        return alpha_logits, beta_logits, gamma_logits, alpha_reverse_logits
