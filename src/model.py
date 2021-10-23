import torch
from torch import Tensor
from transformers import RobertaModel
from typing import Dict, Tuple, List
import torch.nn.functional as F
from torch.nn import Module, Linear, LeakyReLU, LSTM

from embeddings import BoxEmbedding


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


class GumbelIntersection(Module):
    def __init__(self, intersection_temp: float):
        super().__init__()
        self.intersection_temp = intersection_temp

    def forward(self, box1: Tensor, box2: Tensor):
        """
        returns gumbel intersection box of box1 and box2
        box1: [batch_size, # of datasets, min/max, dim]
        box2: [batch_size, # of datasets, min/max, dim]
        return: [batch_size, # of datasets, min/max, dim]
        """
        assert box1.shape[-2] == 2
        assert box2.shape[-2] == 2
        input_boxes = torch.stack([box1, box2], dim=1) # [batch_size, box1/box2, # of datasets, min/max, dim]
        input_boxes_min = input_boxes[..., 0, :]       # [batch_size, box1's min/box2's min, # of datasets, dim]
        input_boxes_max = input_boxes[..., 1, :]       # [batch_size, box1's max/box2's max, # of datasets, dim]

        boxes_min = self.intersection_temp * torch.logsumexp(input_boxes_min / self.intersection_temp, dim=1)   # [batch_size, # of datasets, dim]
        boxes_min = torch.max(boxes_min, torch.max(input_boxes_min, dim=1).values)                  # [batch_size, # of datasets, dim]
        boxes_max = -self.intersection_temp * torch.logsumexp(-input_boxes_max / self.intersection_temp, dim=1)
        boxes_max = torch.min(boxes_max, torch.min(input_boxes_max, dim=1).values)                  # [batch_size, # of datasets, dim]
        return torch.stack([boxes_min, boxes_max], dim=-2) # [batch_size, # of datasets, min/max, dim]


class SoftVolume(Module):
    def __init__(self, volume_temp: float, intersection_temp: float):
        super().__init__()
        self.euler_gamma = 0.57721566490153286060
        self.eps = 1e-23
        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp

    def forward(self, box: Tensor):
        """
        box: [batch_size, # of datasets, min/max, dim]
        return: volume [batch_size, # of datasets]
        """
        assert box.shape[-2] == 2
        box_min, box_max = box[..., 0, :], box[..., 1, :]           # [batch_size, # of datasets, dim]
        inside = box_max - box_min - 2 * self.euler_gamma * self.intersection_temp
        soft_plus = F.softplus(inside, beta=1 / self.volume_temp)
        volume = torch.sum(torch.log(soft_plus + self.eps), dim=-1) # [batch_size, # of datasets]
        return volume


class BoxToBoxVolume(Module):
    def __init__(self, volume_temp: float, intersection_temp: float):
        super().__init__()
        self.gumbel_box = GumbelIntersection(intersection_temp)
        self.volume = SoftVolume(volume_temp, intersection_temp)

    def forward(self, box1: Tensor, box2: Tensor) -> Tensor:
        """
        :param box1: Box for relation1. Tensor of shape (batch_size, # of datasets, min/max, embed_dim)
        :param box2: Box for relation2. Tensor of shape (batch_size, # of datasets, min/max, embed_dim)
        return: log probabilities of P(box1 | box2)
        """
        assert box1.shape[-2] == 2
        assert box2.shape[-2] == 2
        intersection_box = self.gumbel_box(box1, box2)
        intersection_volume = self.volume(intersection_box) # logP(A, B); [batch_size, # of datasets]
        box2_volume = self.volume(box2)                     # logP(B); [batch_size, # of datasets]
        conditional_prob = intersection_volume - box2_volume # logP(A,B)-logP(B)=logP(A|B); [batch_size, # of datasets]
        assert (torch.le(conditional_prob, 0)).all()        # all probability values should be less than or equal to 0
        return intersection_box, conditional_prob


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
        sub = torch.sub(tensor1, tensor2) # [64, 512]
        mul = torch.mul(tensor1, tensor2) # [64, 512]
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

        # MLP layer
        alpha_logits = self.MLP(alpha_repr) # [batch_size, num_classes]; [64, 8]
        beta_logits = self.MLP(beta_repr)
        gamma_logits = self.MLP(gamma_repr)

        return alpha_logits, beta_logits, gamma_logits

class Vector_BiLSTM_MLP(Module):
    def __init__(self, num_classes: int, data_type: str, hidden_size: int, num_layers: int, mlp_size: int,
                 lstm_input_size: int, mlp_output_dim: int, hieve_mlp_size: int, matres_mlp_size: int,
                 roberta_size_type="roberta-base"):
        super().__init__()
        self.eps = 1e-10
        self.num_classes = num_classes
        self.data_type = data_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp_size = mlp_size

        self.MLP = MLP(2 * hidden_size, 2 * mlp_size, mlp_output_dim)

        self.FF1_MLP_hieve = MLP(mlp_output_dim, 2 * mlp_size, hieve_mlp_size)
        self.FF1_MLP_matres = MLP(mlp_output_dim, 2 * mlp_size, matres_mlp_size)

        self.FF2_MLP_hieve = MLP(mlp_output_dim, 2 * mlp_size, hieve_mlp_size)
        self.FF2_MLP_matres = MLP(mlp_output_dim, 2 * mlp_size, matres_mlp_size)

        self.lstm_input_size = lstm_input_size
        self.bilstm = LSTM(self.lstm_input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

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

    def _get_dot_product(self, output_A1, output_B1, output_C1, output_A2, output_B2, output_C2):
        dot_AB = torch.sigmoid(torch.sum(torch.mul(output_A1, output_B1), dim=-1, keepdim=True)).clamp_min(self.eps)  # [batch_size, 1]
        dot_BC = torch.sigmoid(torch.sum(torch.mul(output_B1, output_C1), dim=-1, keepdim=True)).clamp_min(self.eps)
        dot_AC = torch.sigmoid(torch.sum(torch.mul(output_A1, output_C1), dim=-1, keepdim=True)).clamp_min(self.eps)

        dot_BA = torch.sigmoid(torch.sum(torch.mul(output_B2, output_A2), dim=-1, keepdim=True)).clamp_min(self.eps)
        dot_CB = torch.sigmoid(torch.sum(torch.mul(output_C2, output_B2), dim=-1, keepdim=True)).clamp_min(self.eps)
        dot_CA = torch.sigmoid(torch.sum(torch.mul(output_C2, output_A2), dim=-1, keepdim=True)).clamp_min(self.eps)
        return dot_AB, dot_BA, dot_BC, dot_CB, dot_AC, dot_CA


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

        output_A = self.MLP(output_A)  # [batch_size, mlp_output_dim]; [64, 44]
        output_B = self.MLP(output_B)
        output_C = self.MLP(output_C)

        if self.data_type == "hieve":
            output_A1 = self.FF1_MLP_hieve(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B1 = self.FF1_MLP_hieve(output_B)
            output_C1 = self.FF1_MLP_hieve(output_C)

            output_A2 = self.FF2_MLP_hieve(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B2 = self.FF2_MLP_hieve(output_B)
            output_C2 = self.FF2_MLP_hieve(output_C)
            dot_AB, dot_BA, dot_BC, dot_CB, dot_AC, dot_CA = self._get_dot_product(output_A1, output_B1, output_C1, output_A2, output_B2, output_C2)
        elif self.data_type == "matres":
            output_A1 = self.FF1_MLP_matres(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B1 = self.FF1_MLP_matres(output_B)
            output_C1 = self.FF1_MLP_matres(output_C)

            output_A2 = self.FF2_MLP_matres(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B2 = self.FF2_MLP_matres(output_B)
            output_C2 = self.FF2_MLP_matres(output_C)
            dot_AB, dot_BA, dot_BC, dot_CB, dot_AC, dot_CA = self._get_dot_product(output_A1, output_B1, output_C1, output_A2, output_B2, output_C2)
        elif self.data_type == "joint":
            output_A1_hieve = self.FF1_MLP_hieve(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B1_hieve = self.FF1_MLP_hieve(output_B)
            output_C1_hieve = self.FF1_MLP_hieve(output_C)

            output_A2_hieve = self.FF2_MLP_hieve(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B2_hieve = self.FF2_MLP_hieve(output_B)
            output_C2_hieve = self.FF2_MLP_hieve(output_C)

            output_A1_matres = self.FF1_MLP_matres(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B1_matres = self.FF1_MLP_matres(output_B)
            output_C1_matres = self.FF1_MLP_matres(output_C)

            output_A2_matres = self.FF2_MLP_matres(output_A) #[batch_size, mlp_output_dim]; [64, 32]
            output_B2_matres = self.FF2_MLP_matres(output_B)
            output_C2_matres = self.FF2_MLP_matres(output_C)

            dot_AB_hieve, dot_BA_hieve, dot_BC_hieve, dot_CB_hieve, dot_AC_hieve, dot_CA_hieve \
                = self._get_dot_product(output_A1_hieve, output_B1_hieve, output_C1_hieve, output_A2_hieve, output_B2_hieve, output_C2_hieve)
            dot_AB_matres, dot_BA_matres, dot_BC_matres, dot_CB_matres, dot_AC_matres, dot_CA_matres \
                = self._get_dot_product(output_A1_matres, output_B1_matres, output_C1_matres, output_A2_matres, output_B2_matres, output_C2_matres)
            # [batch_size, 1]
            dot_AB = torch.cat([dot_AB_hieve, dot_AB_matres], dim=-1)
            dot_BA = torch.cat([dot_BA_hieve, dot_BA_matres], dim=-1)
            dot_BC = torch.cat([dot_BC_hieve, dot_BC_matres], dim=-1)
            dot_CB = torch.cat([dot_CB_hieve, dot_CB_matres], dim=-1)
            dot_AC = torch.cat([dot_AC_hieve, dot_AC_matres], dim=-1)
            dot_CA = torch.cat([dot_CA_hieve, dot_CA_matres], dim=-1)
        return dot_AB, dot_BA, dot_BC, dot_CB, dot_AC, dot_CA


class Box_BiLSTM_MLP(Module):
    def __init__(self, num_classes: int, data_type: str, hidden_size: int, num_layers: int, mlp_size: int,
                 lstm_input_size: int, volume_temp: int, intersection_temp: int, mlp_output_dim: int,
                 proj_output_dim: int, loss_type: int, n_tags, use_vec_mlp=False, roberta_size_type="roberta-base"):
        super().__init__()
        self.num_classes = num_classes
        self.data_type = data_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp_size = mlp_size
        self.lstm_input_size = lstm_input_size
        self.n_tags = n_tags
        self.bilstm = LSTM(self.lstm_input_size + self.n_tags, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        if use_vec_mlp:
            self.MLP = MLP(2 * hidden_size, 2 * mlp_size, mlp_output_dim)
        self.MLP_hieve = MLP(2 * hidden_size, 2 * mlp_size, 2 * proj_output_dim)
        self.MLP_matres = MLP(2 * hidden_size, 2 * mlp_size, 2 * proj_output_dim)
        self.volume = BoxToBoxVolume(volume_temp=volume_temp, intersection_temp=intersection_temp)

        self.loss_type = loss_type
        if self.loss_type == 1:
            self.MLP_pair = MLP(2 * 3 * hidden_size, 2 * mlp_size, 2 * proj_output_dim)
        elif self.loss_type == 4 or self.loss_type == 2 or self.loss_type == 3:
            self.MLP_h_pair = MLP(2 * 3 * hidden_size, 2 * mlp_size, 2 * proj_output_dim)
            self.MLP_m_pair = MLP(2 * 3 * hidden_size, 2 * mlp_size, 2 * proj_output_dim)

        self.roberta_size_type = roberta_size_type
        self.RoBERTa_layer = RobertaModel.from_pretrained(roberta_size_type)
        print("roberta:", roberta_size_type)
        if roberta_size_type == "roberta-base":
            self.roberta_dim = 768
        elif roberta_size_type == "roberta-large":
            self.roberta_dim = 1024
        else:
            self.roberta_dim = 768

        self.box_embedding = BoxEmbedding(volume_temp=volume_temp, threshold=20)

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

    def _get_pairwise_representation(self, tensor1: Tensor, tensor2: Tensor):
        mul = torch.mul(tensor1, tensor2)  # [64, 512]
        return torch.cat((tensor1, tensor2, mul), 1)

    def _get_pairwise_representation2(self, tensor1: Tensor, tensor2: Tensor):
        return torch.cat((tensor1, tensor2), 1)

    def _get_minimum_enclosing_box(self, box1: Tensor, box2: Tensor):
        # shape: [batch_size, 1, box_min/box_max, hidden_dim]
        box_min = torch.min(torch.cat([box1[..., 0, :], box2[..., 0, :]], dim=-2), dim=-2).values
        box_max = torch.max(torch.cat([box1[..., 1, :], box2[..., 1, :]], dim=-2), dim=-2).values
        return torch.stack((box_min, box_max), dim=-2).unsqueeze(1)

    def forward(self, batch: Tuple[torch.Tensor], device: torch.device, data_type: str):
        # x_sntc: [64, 120]; [batch_size, padding_length]; word id information
        x_sntc, y_sntc, z_sntc = batch[3].to(device), batch[4].to(device), batch[5].to(device)
        x_position, y_position, z_position = batch[6].to(device), batch[7].to(device), batch[8].to(device)
        if self.n_tags > 0:
            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag = batch[9].to(device), batch[10].to(device), batch[11].to(device)

        # get RoBERTa embedding
        roberta_x_sntc = self._get_roberta_embedding(x_sntc) #[64, 120, 768];[batch_size, padded_len, roberta_dim]
        roberta_y_sntc = self._get_roberta_embedding(y_sntc)
        roberta_z_sntc = self._get_roberta_embedding(z_sntc)

        # BiLSTM layer
        if self.n_tags > 0:
            # [batch_size, padded_len, lstm_hidden_dim * 2]; [64, 120, 512]
            bilstm_output_A, _ = self.bilstm(torch.cat([roberta_x_sntc, x_sntc_pos_tag], dim=-1))
            bilstm_output_B, _ = self.bilstm(torch.cat([roberta_y_sntc, y_sntc_pos_tag], dim=-1))
            bilstm_output_C, _ = self.bilstm(torch.cat([roberta_z_sntc, z_sntc_pos_tag], dim=-1))
        else:
            bilstm_output_A, _ = self.bilstm(roberta_x_sntc)
            bilstm_output_B, _ = self.bilstm(roberta_y_sntc)
            bilstm_output_C, _ = self.bilstm(roberta_z_sntc)

        output_A = self._get_embeddings_from_position(bilstm_output_A, x_position) #[batch_size, lstm_hidden_dim * 2]; [64, 512]
        output_B = self._get_embeddings_from_position(bilstm_output_B, y_position)
        output_C = self._get_embeddings_from_position(bilstm_output_C, z_position)

        pairAB = self._get_pairwise_representation(output_A, output_B)
        pairBC = self._get_pairwise_representation(output_B, output_C)
        pairAC = self._get_pairwise_representation(output_A, output_C)

        # projection layers
        if data_type == "hieve":
            # event word
            output_A = self.MLP_hieve(output_A).unsqueeze(1)  # [batch_size, 1, 2 * proj_output_dim]
            output_B = self.MLP_hieve(output_B).unsqueeze(1)
            output_C = self.MLP_hieve(output_C).unsqueeze(1)
            if self.loss_type == 1 or self.loss_type == 3:
                pairAB = self.MLP_pair(pairAB).unsqueeze(1)

        elif data_type == "matres":
            # event word
            output_A = self.MLP_matres(output_A).unsqueeze(1)  # [batch_size, 1, 2 * proj_output_dim]
            output_B = self.MLP_matres(output_B).unsqueeze(1)
            output_C = self.MLP_matres(output_C).unsqueeze(1)
            if self.loss_type == 1 or self.loss_type == 3:
                pairAB = self.MLP_pair(pairAB).unsqueeze(1)

        elif data_type == "joint":
            output_A_hieve = self.MLP_hieve(output_A) # [output_dim, 2*proj_output_dim]
            output_B_hieve = self.MLP_hieve(output_B)
            output_C_hieve = self.MLP_hieve(output_C)
            output_A_matres = self.MLP_matres(output_A) # [output_dim, 2*proj_output_dim]
            output_B_matres = self.MLP_matres(output_B)
            output_C_matres = self.MLP_matres(output_C)
            output_A = torch.stack([output_A_hieve, output_A_matres], dim=1) # [output_dim, 2, 2*proj_output_dim]
            output_B = torch.stack([output_B_hieve, output_B_matres], dim=1)
            output_C = torch.stack([output_C_hieve, output_C_matres], dim=1)

            if self.loss_type == 1:
                pairAB_hieve = self.MLP_pair(pairAB)
                pairAB_matres = self.MLP_pair(pairAB)
                pairAB = torch.stack([pairAB_hieve, pairAB_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]
            elif self.loss_type == 4 or self.loss_type == 2 or self.loss_type == 3:
                pairAB_hieve = self.MLP_h_pair(pairAB)
                pairAB_matres = self.MLP_m_pair(pairAB)
                pairAB = torch.stack([pairAB_hieve, pairAB_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]
                pairBC_hieve = self.MLP_h_pair(pairBC)
                pairBC_matres = self.MLP_m_pair(pairBC)
                pairBC = torch.stack([pairBC_hieve, pairBC_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]
                pairAC_hieve = self.MLP_h_pair(pairAC)
                pairAC_matres = self.MLP_m_pair(pairAC)
                pairAC = torch.stack([pairAC_hieve, pairAC_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]

        dataset_num = output_A.shape[1]
        boxes_A, boxes_B, boxes_C = [], [], []
        if self.loss_type == 1 or self.loss_type == 3 or self.loss_type == 4 or self.loss_type == 2:
            pboxes_AB = []
            pboxes_BC = []
            pboxes_AC = []
        for i in range(dataset_num):
            # box embedding layer
            # [batch_size, 1, min/max, 2*proj_output_dim/2]; [64, 1, 2, 128]
            box_A_tmp = self.box_embedding.get_box_embeddings(output_A[..., i, :]).unsqueeze(dim=1)
            box_B_tmp = self.box_embedding.get_box_embeddings(output_B[..., i, :]).unsqueeze(dim=1)
            box_C_tmp = self.box_embedding.get_box_embeddings(output_C[..., i, :]).unsqueeze(dim=1)

            boxes_A.append(box_A_tmp)
            boxes_B.append(box_B_tmp)
            boxes_C.append(box_C_tmp)
            if self.loss_type == 1 or self.loss_type == 3 or self.loss_type == 4 or self.loss_type == 2:
                pbox_AB_tmp = self.box_embedding.get_box_embeddings(pairAB[..., i, :]).unsqueeze(dim=1)
                pboxes_AB.append(pbox_AB_tmp)
                pbox_BC_tmp = self.box_embedding.get_box_embeddings(pairBC[..., i, :]).unsqueeze(dim=1)
                pboxes_BC.append(pbox_BC_tmp)
                pbox_AC_tmp = self.box_embedding.get_box_embeddings(pairAC[..., i, :]).unsqueeze(dim=1)
                pboxes_AC.append(pbox_AC_tmp)

        box_A = torch.cat(boxes_A, dim=1) # [batch_size, # of boxes, min/max, 2*proj_output_dim/2]
        box_B = torch.cat(boxes_B, dim=1)
        box_C = torch.cat(boxes_C, dim=1)
        if self.loss_type == 1 or self.loss_type == 3 or self.loss_type == 4 or self.loss_type == 2:
            pbox_AB = torch.cat(pboxes_AB, dim=1)
            pbox_BC = torch.cat(pboxes_BC, dim=1)
            pbox_AC = torch.cat(pboxes_AC, dim=1)

        # conditional probabilities
        inter_AB, vol_AB = self.volume(box_A, box_B) # [batch_size, # of datasets]; [64, 2] (joint case) [64, 1] (single case)
        _, vol_BA = self.volume(box_B, box_A)
        inter_BC, vol_BC = self.volume(box_B, box_C)
        _, vol_CB = self.volume(box_C, box_B)
        inter_AC, vol_AC = self.volume(box_A, box_C)
        _, vol_CA = self.volume(box_C, box_A)

        if self.loss_type == 1:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, None, None, None
        elif self.loss_type == 2 and len(boxes_A) == 2:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, None, None, None
        elif self.loss_type == 3 and len(boxes_A) == 2:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            _, pvol_BC = self.volume(inter_BC, pbox_BC)
            _, pvol_AC = self.volume(inter_AC, pbox_AC)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, pvol_BC, pvol_AC, None
        elif self.loss_type == 4:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            _, pvol_BC = self.volume(inter_BC, pbox_BC)
            _, pvol_AC = self.volume(inter_AC, pbox_AC)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, pvol_BC, pvol_AC, None
        else:
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, None, None, None, None


class Box_RoBERTa_MLP(Module):
    def __init__(self, num_classes: int, data_type: str, hidden_size: int, num_layers: int, mlp_size: int,
                 lstm_input_size: int, volume_temp: int, intersection_temp: int, mlp_output_dim: int,
                 proj_output_dim: int, loss_type: int, n_tags, use_vec_mlp=False, roberta_size_type="roberta-base"):
        super().__init__()
        self.num_classes = num_classes
        self.data_type = data_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mlp_size = mlp_size
        self.lstm_input_size = lstm_input_size
        self.n_tags = n_tags

        self.roberta_size_type = roberta_size_type
        self.RoBERTa_layer = RobertaModel.from_pretrained(roberta_size_type)
        print("roberta:", roberta_size_type)
        if roberta_size_type == "roberta-base":
            self.roberta_dim = 768
        elif roberta_size_type == "roberta-large":
            self.roberta_dim = 1024
        else:
            self.roberta_dim = 768

        self.MLP_hieve = MLP(self.roberta_dim, 2 * mlp_size, 2 * proj_output_dim)
        self.MLP_matres = MLP(self.roberta_dim, 2 * mlp_size, 2 * proj_output_dim)
        self.volume = BoxToBoxVolume(volume_temp=volume_temp, intersection_temp=intersection_temp)

        self.loss_type = loss_type
        if self.loss_type == 1:
            self.MLP_pair = MLP(3 * self.roberta_dim, 2 * mlp_size, 2 * proj_output_dim)
        elif self.loss_type == 4 or self.loss_type == 2 or self.loss_type == 3:
            self.MLP_h_pair = MLP(3 * self.roberta_dim, 2 * mlp_size, 2 * proj_output_dim)
            self.MLP_m_pair = MLP(3 * self.roberta_dim, 2 * mlp_size, 2 * proj_output_dim)

        self.box_embedding = BoxEmbedding(volume_temp=volume_temp, threshold=20)

    def _get_embeddings_from_position(self, roberta_embd: Tensor, position: Tensor):
        batch_size = position.shape[0]
        return torch.cat([roberta_embd[i, position[i], :].unsqueeze(0) for i in range(0, batch_size)], 0)

    def _get_pairwise_representation(self, tensor1: Tensor, tensor2: Tensor):
        mul = torch.mul(tensor1, tensor2)  # [64, 512]
        return torch.cat((tensor1, tensor2, mul), 1)

    def forward(self, batch: Tuple[torch.Tensor], device: torch.device, data_type: str):
        # x_sntc: [64, 120]; [batch_size, padding_length]; word id information
        x_sntc, y_sntc, z_sntc = batch[3].to(device), batch[4].to(device), batch[5].to(device)
        x_position, y_position, z_position = batch[6].to(device), batch[7].to(device), batch[8].to(device)
        if self.n_tags > 0:
            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag = batch[9].to(device), batch[10].to(device), batch[11].to(
                device)

        output_A = self._get_embeddings_from_position(self.RoBERTa_layer(x_sntc)[0], x_position)  # [128, 768]
        output_B = self._get_embeddings_from_position(self.RoBERTa_layer(y_sntc)[0], y_position)
        output_C = self._get_embeddings_from_position(self.RoBERTa_layer(z_sntc)[0], z_position)

        pairAB = self._get_pairwise_representation(output_A, output_B)
        pairBC = self._get_pairwise_representation(output_B, output_C)
        pairAC = self._get_pairwise_representation(output_A, output_C)

        # projection layers
        if data_type == "hieve":
            # event word
            output_A = self.MLP_hieve(output_A).unsqueeze(1)  # [batch_size, 1, 2 * proj_output_dim]
            output_B = self.MLP_hieve(output_B).unsqueeze(1)
            output_C = self.MLP_hieve(output_C).unsqueeze(1)
            if self.loss_type == 1 or self.loss_type == 3:
                pairAB = self.MLP_pair(pairAB).unsqueeze(1)

        elif data_type == "matres":
            # event word
            output_A = self.MLP_matres(output_A).unsqueeze(1)  # [batch_size, 1, 2 * proj_output_dim]
            output_B = self.MLP_matres(output_B).unsqueeze(1)
            output_C = self.MLP_matres(output_C).unsqueeze(1)
            if self.loss_type == 1 or self.loss_type == 3:
                pairAB = self.MLP_pair(pairAB).unsqueeze(1)

        elif data_type == "joint":
            output_A_hieve = self.MLP_hieve(output_A)  # [output_dim, 2*proj_output_dim]
            output_B_hieve = self.MLP_hieve(output_B)
            output_C_hieve = self.MLP_hieve(output_C)
            output_A_matres = self.MLP_matres(output_A)  # [output_dim, 2*proj_output_dim]
            output_B_matres = self.MLP_matres(output_B)
            output_C_matres = self.MLP_matres(output_C)
            output_A = torch.stack([output_A_hieve, output_A_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]
            output_B = torch.stack([output_B_hieve, output_B_matres], dim=1)
            output_C = torch.stack([output_C_hieve, output_C_matres], dim=1)

            if self.loss_type == 1:
                pairAB_hieve = self.MLP_pair(pairAB)
                pairAB_matres = self.MLP_pair(pairAB)
                pairAB = torch.stack([pairAB_hieve, pairAB_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]
            elif self.loss_type == 4 or self.loss_type == 2 or self.loss_type == 3:
                pairAB_hieve = self.MLP_h_pair(pairAB)
                pairAB_matres = self.MLP_m_pair(pairAB)
                pairAB = torch.stack([pairAB_hieve, pairAB_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]
                pairBC_hieve = self.MLP_h_pair(pairBC)
                pairBC_matres = self.MLP_m_pair(pairBC)
                pairBC = torch.stack([pairBC_hieve, pairBC_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]
                pairAC_hieve = self.MLP_h_pair(pairAC)
                pairAC_matres = self.MLP_m_pair(pairAC)
                pairAC = torch.stack([pairAC_hieve, pairAC_matres], dim=1)  # [output_dim, 2, 2*proj_output_dim]

        dataset_num = output_A.shape[1]
        boxes_A, boxes_B, boxes_C = [], [], []
        if self.loss_type == 1 or self.loss_type == 3 or self.loss_type == 4 or self.loss_type == 2:
            pboxes_AB = []
            pboxes_BC = []
            pboxes_AC = []
        for i in range(dataset_num):
            # box embedding layer
            # [batch_size, 1, min/max, 2*proj_output_dim/2]; [64, 1, 2, 128]
            box_A_tmp = self.box_embedding.get_box_embeddings(output_A[..., i, :]).unsqueeze(dim=1)
            box_B_tmp = self.box_embedding.get_box_embeddings(output_B[..., i, :]).unsqueeze(dim=1)
            box_C_tmp = self.box_embedding.get_box_embeddings(output_C[..., i, :]).unsqueeze(dim=1)

            boxes_A.append(box_A_tmp)
            boxes_B.append(box_B_tmp)
            boxes_C.append(box_C_tmp)
            if self.loss_type == 1 or self.loss_type == 3 or self.loss_type == 4 or self.loss_type == 2:
                pbox_AB_tmp = self.box_embedding.get_box_embeddings(pairAB[..., i, :]).unsqueeze(dim=1)
                pboxes_AB.append(pbox_AB_tmp)
                pbox_BC_tmp = self.box_embedding.get_box_embeddings(pairBC[..., i, :]).unsqueeze(dim=1)
                pboxes_BC.append(pbox_BC_tmp)
                pbox_AC_tmp = self.box_embedding.get_box_embeddings(pairAC[..., i, :]).unsqueeze(dim=1)
                pboxes_AC.append(pbox_AC_tmp)

        box_A = torch.cat(boxes_A, dim=1)  # [batch_size, # of boxes, min/max, 2*proj_output_dim/2]
        box_B = torch.cat(boxes_B, dim=1)
        box_C = torch.cat(boxes_C, dim=1)
        if self.loss_type == 1 or self.loss_type == 3 or self.loss_type == 4 or self.loss_type == 2:
            pbox_AB = torch.cat(pboxes_AB, dim=1)
            pbox_BC = torch.cat(pboxes_BC, dim=1)
            pbox_AC = torch.cat(pboxes_AC, dim=1)

        # conditional probabilities
        inter_AB, vol_AB = self.volume(box_A, box_B)  # [batch_size, # of datasets]; [64, 2] (joint case) [64, 1] (single case)
        _, vol_BA = self.volume(box_B, box_A)
        inter_BC, vol_BC = self.volume(box_B, box_C)
        _, vol_CB = self.volume(box_C, box_B)
        inter_AC, vol_AC = self.volume(box_A, box_C)
        _, vol_CA = self.volume(box_C, box_A)

        if self.loss_type == 1:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, None, None, None
        elif self.loss_type == 2 and len(boxes_A) == 2:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, None, None, None
        elif self.loss_type == 3 and len(boxes_A) == 2:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            _, pvol_BC = self.volume(inter_BC, pbox_BC)
            _, pvol_AC = self.volume(inter_AC, pbox_AC)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, pvol_BC, pvol_AC, None
        elif self.loss_type == 4:
            _, pvol_AB = self.volume(inter_AB, pbox_AB)
            _, pvol_BC = self.volume(inter_BC, pbox_BC)
            _, pvol_AC = self.volume(inter_AC, pbox_AC)
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, pvol_AB, pvol_BC, pvol_AC, None
        else:
            return vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, None, None, None, None
