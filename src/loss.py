import torch
from torch import Tensor, autograd
from torch.nn import Module, LogSoftmax , BCEWithLogitsLoss
from collections import defaultdict

from utils import log1mexp


class TransitionLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_y_alpha, log_y_beta, log_y_gamma, alpha_index, beta_index, gamma_index):
        zero = torch.zeros(1).to(log_y_alpha.device)
        loss = torch.max(zero, log_y_alpha[:, alpha_index] + log_y_beta[:, beta_index] - log_y_gamma[:, gamma_index])
        return loss


class TransitionNotLoss(Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, log_y_alpha, log_y_beta, log_y_gamma, alpha_index, beta_index, gamma_index):
        zero = torch.zeros(1).to(log_y_alpha.device)
        log_not_y_gamma = (1 - log_y_gamma.exp()).clamp(self.eps).log()
        loss = torch.max(zero, log_y_alpha[:, alpha_index] + log_y_beta[:, beta_index] - log_not_y_gamma[:, gamma_index])
        return loss


class TransitivityLoss(Module):
    def __init__(self):
        super().__init__()
        self.softmax = LogSoftmax(dim=1)
        self.transition_loss = TransitionLoss()
        self.transition_not_loss = TransitionNotLoss()

    def forward(self, alpha_logits: Tensor, beta_logits: Tensor, gamma_logits: Tensor):
        log_y_alpha = self.softmax(alpha_logits)
        log_y_beta = self.softmax(beta_logits)
        log_y_gamma = self.softmax(gamma_logits)

        loss = self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 0, 0)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 2, 0)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 1, 1)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 2, 1)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 0, 0)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 1, 1)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 2, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 3, 3)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 3, 2, 3)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 3, 1)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 3, 2)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 3, 0)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 3, 2)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 3, 0, 1)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 3, 0, 2)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 3, 1, 0)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 3, 1, 2)

        return loss

class CrossCategoryLoss(Module):
    def __init__(self):
        super().__init__()
        self.softmax = LogSoftmax(dim=1)
        self.transition_loss = TransitionLoss()
        self.transition_not_loss = TransitionNotLoss()

    def forward(self, alpha_logits: Tensor, beta_logits: Tensor, gamma_logits: Tensor):
        """
        The induction table for conjunctive constraints on temporal and subevent relations.
        Refer Table1 in the paper: "Joint Constrained Learning for Event-Event Relation Extraction"
        0 - PC (Parent-Child), 1 - CP (Child-Parent), 2 - CR (CoRef), 3 - NR (NoRel)
        4 - BF (Before), 5 - AF (After), 6 - EQ (Equal), 7 - VG (Vague)
        """
        log_y_alpha = self.softmax(alpha_logits)
        log_y_beta = self.softmax(beta_logits)
        log_y_gamma = self.softmax(gamma_logits)

        loss = self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 4, 4)      # (PC, BF) -> BF
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 4, 1) # (PC, BF) -> -CP
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 4, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 6, 4)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 6, 1)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 0, 6, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 5, 5)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 5, 0)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 5, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 6, 5)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 6, 0)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 1, 6, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 4, 4)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 4, 1)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 4, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 5, 5)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 5, 0)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 5, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 6, 6)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 7, 7)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 2, 7, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 0, 4)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 0, 1)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 0, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 2, 4)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 2, 1)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 4, 2, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 1, 5)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 1, 0)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 1, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 2, 5)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 2, 0)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 5, 2, 2)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 6, 2, 6)
        loss += self.transition_loss(log_y_alpha, log_y_beta, log_y_gamma, 7, 2, 7)
        loss += self.transition_not_loss(log_y_alpha, log_y_beta, log_y_gamma, 7, 2, 2)
        return loss


class BCELossWithLog(Module):
    """
    binary cross entropy loss with log probabilities
    """
    def __init__(self, data_type, hier_weights, temp_weights):
        super().__init__()
        if data_type == "hieve":
            self.weights = autograd.Variable(hier_weights)
        elif data_type == "matres":
            self.weights = autograd.Variable(temp_weights)
        elif data_type == "joint":
            self.hier_weights = autograd.Variable(hier_weights)
            self.temp_weights = autograd.Variable(temp_weights)

    def loss_calculation(self, volume1, volume2, label1, label2):
        # loss = -(label1 * volume1 + (1 - label1) * log1mexp(volume1) + label2 * volume2 + (1 - label2) * log1mexp(volume2)).sum()
        vol1_pos_loss = (label1 * volume1).sum()
        vol1_neg_loss = ((1 - label1) * log1mexp(volume1)).sum()
        vol1_loss = vol1_pos_loss+vol1_neg_loss

        vol2_pos_loss = (label2 * volume2).sum()
        vol2_neg_loss = ((1 - label2) * log1mexp(volume2)).sum()
        vol2_loss = vol2_pos_loss + vol2_neg_loss

        loss = vol1_loss + vol2_loss
        return -loss

    def loss_calculation_with_weights(self, volume1, volume2, label1, label2, type):
        # loss = -(label1 * volume1 + (1 - label1) * log1mexp(volume1) + label2 * volume2 + (1 - label2) * log1mexp(volume2)).sum()
        vol1_pos_loss = (label1 * volume1)
        vol1_neg_loss = ((1 - label1) * log1mexp(volume1))
        vol1_loss = vol1_pos_loss+vol1_neg_loss

        vol2_pos_loss = (label2 * volume2)
        vol2_neg_loss = ((1 - label2) * log1mexp(volume2))
        vol2_loss = vol2_pos_loss + vol2_neg_loss

        if type == "hieve":
            weighted_loss = (vol1_loss[label1 == 1] * self.hier_weights[0]).sum()
            weighted_loss += (vol1_loss[label1 == 0] * self.hier_weights[1]).sum()
            weighted_loss += (vol2_loss[label2 == 1] * self.hier_weights[0]).sum()
            weighted_loss += (vol2_loss[label2 == 0] * self.hier_weights[1]).sum()
        if type == "matres":
            weighted_loss = (vol1_loss[label1 == 1] * self.temp_weights[0]).sum()
            weighted_loss += (vol1_loss[label1 == 0] * self.temp_weights[1]).sum()
            weighted_loss += (vol2_loss[label2 == 1] * self.temp_weights[0]).sum()
            weighted_loss += (vol2_loss[label2 == 0] * self.temp_weights[1]).sum()
        return -weighted_loss

    def forward(self, volume1, volume2, labels, flag, lambda_dict):
        """
        volume1: P(A|B); [batch_size, # of datasets]
        volume2: P(B|A); [batch_size, # of datasets]
        labels: [batch_size, 2]; PC: (1,0), CP: (0,1), CR: (1,1), VG: (0,0)
        flag:   [batch_size]; 0: HiEve, 1: MATRES
        -(labels[:, 0] * log volume1 + (1 - labels[:, 0]) * log(1 - volume1) + labels[:, 1] * log volume2 + (1 - labels[:, 1]) * log(1 - volume2)).sum()
        """
        if volume1.shape[-1] == 1:
            label1 = labels[:, 0].unsqueeze(-1)
            label2 = labels[:, 1].unsqueeze(-1)
            assert volume1.shape == label1.shape and volume2.shape == label2.shape
            loss = self.loss_calculation(volume1, volume2, label1, label2)
        else:
            hieve_mask = (flag == 0).nonzero()
            hieve_loss = self.loss_calculation(volume1[:, 0][hieve_mask], volume2[:, 0][hieve_mask], labels[:, 0][hieve_mask], labels[:, 1][hieve_mask])
            matres_mask = (flag == 1).nonzero()
            matres_loss = self.loss_calculation(volume1[:, 1][matres_mask], volume2[:, 1][matres_mask], labels[:, 0][matres_mask], labels[:, 1][matres_mask])
            loss = lambda_dict["lambda_condi_h"] * hieve_loss + lambda_dict["lambda_condi_m"] * matres_loss
        return loss


class BCELossWithLogP(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pvol, label, flag, lambda_dict):
        """
        volume: P((A n B n AB) | AB)
        label: 1 or 0
        PC, CP, CR: P(A,B|AB) -> 1 and NR: P(A,B|AB) -> 0
        """
        not_nr_indices = [i for i, lbl in enumerate(label.tolist()) if lbl[0] != 0 or lbl[1] != 0]
        nr_indices = [i for i, lbl in enumerate(label.tolist()) if lbl[0] == 0 and lbl[1] == 0]
        assert len(not_nr_indices) + len(nr_indices) == pvol.shape[0]

        if pvol.shape[-1] == 1:
            not_nr_vol = pvol[not_nr_indices]
            nr_vol = pvol[nr_indices]
            loss = -(not_nr_vol.sum() + log1mexp(nr_vol).sum())
        else:
            not_nr_pvol = pvol[not_nr_indices]
            nr_pvol = pvol[nr_indices]
            not_nr_flag = flag[not_nr_indices]
            nr_flag = flag[nr_indices]
            hieve_loss = -((not_nr_pvol[:,0][not_nr_flag == 0]).sum() + log1mexp(nr_pvol[:,0][nr_flag == 0]).sum())
            matres_loss = -((not_nr_pvol[:,1][not_nr_flag == 1]).sum() + log1mexp(nr_pvol[:,1][nr_flag == 1]).sum())
            loss = lambda_dict["lambda_pair_h"] * hieve_loss + lambda_dict["lambda_pair_m"] * matres_loss
        return loss


class BCELogitLoss(Module):
    def __init__(self):
        super().__init__()
        self.bll = BCEWithLogitsLoss()

    def forward(self, logit1, logit2, labels, flag):
        """
        logit1: P(A|B); [batch_size, # of datasets]
        logit2: P(B|A); [batch_size, # of datasets]
        labels: [batch_size, 2]; PC: (1,0), CP: (0,1), CR: (1,1), VG: (0,0)
        flag:   [batch_size]; 0: HiEve, 1: MATRES
        """
        labels = labels.to(device=logit1.device, dtype=torch.float)
        if logit1.shape[-1] == 1:
            loss = self.bll(logit1, labels[:,0].unsqueeze(-1)) + self.bll(logit2, labels[:,1].unsqueeze(-1))
        else:
            # loss between P(A|B) and labels[:,0] for HiEve Data +
            # loss between P(B|A) and labels[:,1] for HiEve Data
            hieve_mask = (flag == 0).nonzero()
            hieve_loss = self.bll(logit1[:,0][hieve_mask],labels[:,0][hieve_mask]) + self.bll(logit2[:,0][hieve_mask], labels[:,1][hieve_mask])

            # loss between P(A|B) and labels[:,0] for MATRES Data +
            # loss between P(B|A) and labels[:,1] for MATRES Data
            matres_mask = (flag == 1).nonzero()
            matres_loss = self.bll(logit1[:,1][matres_mask],labels[:,0][matres_mask]) + self.bll(logit2[:,1][matres_mask], labels[:,1][matres_mask])
            loss = hieve_loss + matres_loss
        return loss


class BoxCrossCategoryLoss(Module):
    def __init__(
            self, xy_rel_id: torch.Tensor, yz_rel_id: torch.Tensor, xz_rel_id: torch.Tensor,
            flag: torch.Tensor
        ):
        super().__init__()
        self.eps = 1e-8
        self.batch_size = xy_rel_id.shape[0]
        self.hieve_mask = (flag == 0)
        self.matres_mask = (flag == 1)
        self.xy_rel_map = self.get_rel_map(xy_rel_id)
        self.yz_rel_map = self.get_rel_map(yz_rel_id)
        self.xz_rel_map = self.get_rel_map(xz_rel_id)
        self.dataset_map = {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1
        }
    
    def get_rel_map(self, rel_id: torch.Tensor) -> dict:
        pc = ((rel_id[..., 0] == 1) & (rel_id[..., 1] == 0))
        cp = ((rel_id[..., 0] == 0) & (rel_id[..., 1] == 1))
        cr = ((rel_id[..., 0] == 1) & (rel_id[..., 1] == 1))
        vg = ((rel_id[..., 0] == 0) & (rel_id[..., 1] == 0))
        rel_map = {
            0: (pc & self.hieve_mask),
            1: (cp & self.hieve_mask),
            2: (cr & self.hieve_mask),
            3: (vg & self.hieve_mask),
            4: (pc & self.matres_mask),
            5: (cp & self.matres_mask),
            6: (cr & self.matres_mask),
            7: (vg & self.matres_mask)
        }
        return rel_map
    
    @staticmethod
    def loss_calculation(volume1, volume2, volume3, idx, flag1, flag2, flag3):
        loss = volume1[idx, flag1] + volume2[idx, flag2] - volume3[idx, flag3]
        return -loss.sum()
    
    @staticmethod
    def neg_loss_calculation(volume1, volume2, volume3, idx, flag1, flag2, flag3):
        neg_volume3 = log1mexp(volume3[idx][flag3])
        loss = volume1[idx][flag1] + volume2[idx][flag2] - neg_volume3
        return -loss.sum()
    
    def forward(self, volume1, volume2, volume3):
        loss_recipe = [
            (0,4,4), (0,6,4), 
            (1,5,5), (1,6,5), 
            (2,4,4), (2,5,5), 
            (2,6,6), (2,7,7), 
            (4,0,4), (4,2,4), 
            (5,1,5), (5,2,5), 
            (6,2,6), (7,2,7)
        ]
        neg_loss_recipe = [
            (0,4,1), (0,4,2),
            (0,6,1), (0,6,2),
            (1,5,0), (1,5,2),
            (1,6,0), (1,6,2),
            (2,4,1), (2,4,2),
            (2,5,0), (2,5,2),
            (4,0,1), (4,0,2),
            (4,2,1), (4,2,2),
            (5,1,0), (5,1,2),
            (5,2,0), (5,2,2),
            (2,7,2), (7,2,2)
        ]
        loss = 0
        for xy, yz, xz in loss_recipe:
            idx = (self.xy_rel_map[xy] & self.yz_rel_map[yz] & self.xz_rel_map[xz]).nonzero().flatten()
            if idx.shape[0] > 0:
                flag1 = self.dataset_map[xy]
                flag2 = self.dataset_map[yz]
                flag3 = self.dataset_map[xz]
                loss += self.loss_calculation(volume1, volume2, volume3, idx, flag1, flag2, flag3)
        for xy, yz, xz in neg_loss_recipe:
            idx = (self.xy_rel_map[xy] & self.yz_rel_map[yz] & self.xz_rel_map[xz]).nonzero().flatten()
            if idx.shape[0] > 0:
                flag1 = self.dataset_map[xy]
                flag2 = self.dataset_map[yz]
                flag3 = self.dataset_map[xz]
                loss += self.neg_loss_calculation(volume1, volume2, volume3, idx, flag1, flag2, flag3)
        return loss

