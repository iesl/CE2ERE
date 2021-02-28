import torch
from torch import Tensor
from torch.nn import Module, LogSoftmax


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
        log_y_alpha = self.softmax(alpha_logits[:, 0:4])
        log_y_beta = self.softmax(beta_logits[:, 0:4])
        log_y_gamma = self.softmax(gamma_logits[:, 0:4])

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

        return loss.sum()


class SymmetryLoss(Module):
    def __init__(self):
        super().__init__()
        self.softmax = LogSoftmax(dim=1)

    def symmetry_loss(self, log_y_alpha, log_y_alpha_reverse, alpha_index, alpha_reverse_index):
        zero = torch.zeros(1).to(log_y_alpha.device)
        loss = torch.max(zero, log_y_alpha[:, alpha_index] - log_y_alpha_reverse[:, alpha_reverse_index])
        return loss

    def forward(self, alpha_logits: Tensor, alpha_reverse_logits: Tensor):
        log_y_alpha = self.softmax(alpha_logits[:, 0:4])
        log_y_alpha_reverse = self.softmax(alpha_reverse_logits[:, 0:4])

        loss = self.symmetry_loss(log_y_alpha, log_y_alpha_reverse, 0, 1)
        loss += self.symmetry_loss(log_y_alpha, log_y_alpha_reverse, 1, 0)
        loss += self.symmetry_loss(log_y_alpha, log_y_alpha_reverse, 2, 3)
        loss += self.symmetry_loss(log_y_alpha, log_y_alpha_reverse, 3, 2)
        return loss.sum()


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
        return loss.sum()
