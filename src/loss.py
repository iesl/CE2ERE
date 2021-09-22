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
            self.weights = hier_weights
        elif data_type == "matres":
            self.weights = temp_weights
        elif data_type == "joint":
            self.hier_weights = hier_weights
            self.temp_weights = temp_weights

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

        vol_loss = vol1_loss + vol2_loss
        pc = ((label1 == 1) & (label2 == 0)).squeeze(-1)
        cp = ((label1 == 0) & (label2 == 1)).squeeze(-1)
        cr = ((label1 == 1) & (label2 == 1)).squeeze(-1)
        nr = ((label1 == 0) & (label2 == 0)).squeeze(-1)
        if type == "hieve":
            weighted_loss = (vol_loss[pc] * self.hier_weights[0]).sum()
            weighted_loss += (vol_loss[cp] * self.hier_weights[1]).sum()
            weighted_loss += (vol_loss[cr] * self.hier_weights[2]).sum()
            weighted_loss += (vol_loss[nr] * self.hier_weights[3]).sum()
        if type == "matres":
            weighted_loss = (vol_loss[pc] * self.temp_weights[0]).sum()
            weighted_loss += (vol_loss[cp] * self.temp_weights[1]).sum()
            weighted_loss += (vol_loss[cr] * self.temp_weights[2]).sum()
            weighted_loss += (vol_loss[nr] * self.temp_weights[3]).sum()
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
            # hieve_mask = (flag == 0).nonzero()
            # hieve_loss = self.loss_calculation_with_weights(volume1[:, 0][hieve_mask], volume2[:, 0][hieve_mask], labels[:, 0][hieve_mask], labels[:, 1][hieve_mask], "hieve")
            # matres_mask = (flag == 1).nonzero()
            # matres_loss = self.loss_calculation_with_weights(volume1[:, 1][matres_mask], volume2[:, 1][matres_mask], labels[:, 0][matres_mask], labels[:, 1][matres_mask], "matres")
            hieve_mask = (flag == 0).nonzero()
            hieve_loss = self.loss_calculation(volume1[:, 0][hieve_mask], volume2[:, 0][hieve_mask], labels[:, 0][hieve_mask], labels[:, 1][hieve_mask])
            matres_mask = (flag == 1).nonzero()
            matres_loss = self.loss_calculation(volume1[:, 1][matres_mask], volume2[:, 1][matres_mask], labels[:, 0][matres_mask], labels[:, 1][matres_mask])
            loss = lambda_dict["lambda_condi_h"] * hieve_loss + lambda_dict["lambda_condi_m"] * matres_loss
        return loss


class BCELossWithLogP(Module):
    def __init__(self, data_type, hier_weights, temp_weights):
        super().__init__()
        if data_type == "hieve":
            self.weights = hier_weights
        elif data_type == "matres":
            self.weights = temp_weights
        elif data_type == "joint":
            self.hier_weights = hier_weights
            self.temp_weights = temp_weights

    def forward(self, pvol, label, flag, lambda_dict):
        """
        volume: P((A n B n AB) | AB)
        label: 1 or 0
        PC, CP, CR: P(A,B|AB) -> 1 and NR: P(A,B|AB) -> 0
        """
        # not_nr = ((label[:, 0] != 0) | (label[:, 1] != 0)).squeeze(-1)
        pc = ((label[:, 0] == 1) & (label[:, 1] == 0)).squeeze(-1)
        cp = ((label[:, 0] == 0) & (label[:, 1] == 1)).squeeze(-1)
        cr = ((label[:, 0] == 1) & (label[:, 1] == 1)).squeeze(-1)
        nr = ((label[:, 0] == 0) & (label[:, 1] == 0)).squeeze(-1)

        if pvol.shape[-1] == 1:
            pc_vol = pvol[pc]
            cp_vol = pvol[cp]
            cr_vol = pvol[cr]
            nr_vol = pvol[nr]
            loss = -(pc_vol.sum() + cp_vol.sum() + cr_vol.sum() + log1mexp(nr_vol).sum())
        else:
            # hieve_loss = (pvol[:, 0][(flag == 0) & pc] * self.hier_weights[0]).sum()
            # hieve_loss += (pvol[:, 0][(flag == 0) & cp] * self.hier_weights[1]).sum()
            # hieve_loss += (pvol[:, 0][(flag == 0) & cr] * self.hier_weights[2]).sum()
            # hieve_loss += (log1mexp(pvol[:, 0][(flag == 0) & nr]) * self.hier_weights[3]).sum()
            #
            # matres_loss = (pvol[:, 1][(flag == 1) & pc] * self.temp_weights[0]).sum()
            # matres_loss += (pvol[:, 1][(flag == 1) & cp] * self.temp_weights[1]).sum()
            # matres_loss += (pvol[:, 1][(flag == 1) & cr] * self.temp_weights[2]).sum()
            # matres_loss += (log1mexp(pvol[:, 1][(flag == 1) & nr]) * self.temp_weights[3]).sum()

            hieve_loss = pvol[:, 0][(flag == 0) & pc].sum()
            hieve_loss += pvol[:, 0][(flag == 0) & cp].sum()
            hieve_loss += pvol[:, 0][(flag == 0) & cr].sum()
            hieve_loss += log1mexp(pvol[:, 0][(flag == 0) & nr]).sum()

            matres_loss = pvol[:, 1][(flag == 1) & pc].sum()
            matres_loss += pvol[:, 1][(flag == 1) & cp].sum()
            matres_loss += pvol[:, 1][(flag == 1) & cr].sum()
            matres_loss += log1mexp(pvol[:, 1][(flag == 1) & nr]).sum()
            loss = -(lambda_dict["lambda_pair_h"] * hieve_loss + lambda_dict["lambda_pair_m"] * matres_loss)
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
        logit1 = logit1.clamp_min(1e-10)
        logit2 = logit2.clamp_min(1e-10)
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
    def __init__(self):
        super().__init__()
        self.eps = 1e-8
        self.dataset_map = {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1
        }
        self.loss_recipe = [
            (0, 4, 4), (0, 6, 4),
            (1, 5, 5), (1, 6, 5),
            (2, 4, 4), (2, 5, 5),
            (2, 6, 6), (2, 7, 7),
            (4, 0, 4), (4, 2, 4),
            (5, 1, 5), (5, 2, 5),
            (6, 2, 6), (7, 2, 7)
        ]
        self.neg_loss_recipe = [
            (0, 4, 1), (0, 4, 2),
            (0, 6, 1), (0, 6, 2),
            (1, 5, 0), (1, 5, 2),
            (1, 6, 0), (1, 6, 2),
            (2, 4, 1), (2, 4, 2),
            (2, 5, 0), (2, 5, 2),
            (4, 0, 1), (4, 0, 2),
            (4, 2, 1), (4, 2, 2),
            (5, 1, 0), (5, 1, 2),
            (5, 2, 0), (5, 2, 2),
            (2, 7, 2), (7, 2, 2)
        ]

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
    def loss_calculation(volume1, volume2, volume3, flag1, flag2, flag3):
        loss = torch.max(torch.zeros(1).to(volume1.device), volume1[:, flag1] + volume2[:, flag2] - volume3[:, flag3])
        return loss.sum()

    @staticmethod
    def neg_loss_calculation(volume1, volume2, volume3, flag1, flag2, flag3):
        neg_volume3 = log1mexp(volume3[:, flag3])
        loss = torch.max(torch.zeros(1).to(volume1.device), volume1[:, flag1] + volume2[:, flag2] - neg_volume3)
        return loss.sum()

    @staticmethod
    def create_probabilities(volume1, volume2):
        vol_PC = volume1 + log1mexp(volume2)
        vol_CP = log1mexp(volume1) + volume2
        vol_CR = volume1 + volume2
        vol_NR = log1mexp(volume1) + log1mexp(volume2)
        return [vol_PC, vol_CP, vol_CR, vol_NR]

    def forward(self, vol_AB, vol_BA, vol_BC, vol_CB, vol_AC, vol_CA, xy_rel_id, yz_rel_id, xz_rel_id):
        pAB_list, pBC_list, pAC_list = [], [], []
        pAB_list.extend(self.create_probabilities(vol_AB, vol_BA))
        pBC_list.extend(self.create_probabilities(vol_BC, vol_CB))
        pAC_list.extend(self.create_probabilities(vol_AC, vol_CA))

        loss = 0
        for xy, yz, xz in self.loss_recipe:
            flag1 = self.dataset_map[xy]
            flag2 = self.dataset_map[yz]
            flag3 = self.dataset_map[xz]
            loss += self.loss_calculation(pAB_list[xy % 4], pBC_list[yz % 4], pAC_list[xz % 4], flag1, flag2, flag3)
        for xy, yz, xz in self.neg_loss_recipe:
            flag1 = self.dataset_map[xy]
            flag2 = self.dataset_map[yz]
            flag3 = self.dataset_map[xz]
            loss += self.neg_loss_calculation(pAB_list[xy % 4], pBC_list[yz % 4], pAC_list[xz % 4], flag1, flag2, flag3)
        return loss