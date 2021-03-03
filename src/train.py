import time

import torch
import wandb
from torch import Tensor, FloatTensor
from tqdm import tqdm
from typing import Dict, Union

from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader

from metrics import metric


class Trainer:
    def __init__(self, data_type: str, model: Module, device: torch.device, epochs: int, learning_rate: float, train_dataloader: DataLoader, evaluator: Module,
                 opt: torch.optim.Optimizer, loss_type: int, loss_anno_dict: Dict[str, Module], loss_symmetry: Module,
                 loss_transitivity: Module, loss_cross_category: Module, lambda_dict: Dict[str, float], no_valid: bool):
        self.data_type = data_type
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_dict = lambda_dict
        self.train_dataloader = train_dataloader

        self.evaluator = evaluator
        self.opt = opt

        self.loss_type = loss_type
        self.loss_anno_dict = loss_anno_dict
        self.loss_func_symm = loss_symmetry
        self.loss_func_trans = loss_transitivity
        self.loss_func_cross = loss_cross_category

        self.no_valid = no_valid
        self.best_f1_score = 0.0

    def _get_anno_loss(self, batch_size: int, flag: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor,
                       xy_rel_id: Tensor, yz_rel_id: Tensor, xz_rel_id: Tensor):
        anno_loss = 0
        for i in range(0, batch_size):
            if flag[i] == 0:    # HiEve
                alpha_loss = self.loss_anno_dict["hieve"](alpha[i][0:4].unsqueeze(0), xy_rel_id[i].unsqueeze(0))
                beta_loss = self.loss_anno_dict["hieve"](beta[i][0:4].unsqueeze(0), yz_rel_id[i].unsqueeze(0))
                gamma_loss = self.loss_anno_dict["hieve"](gamma[i][0:4].unsqueeze(0), xz_rel_id[i].unsqueeze(0))
            elif flag[i] == 1:  # MATRES
                alpha_loss = self.loss_anno_dict["matres"](alpha[i][4:].unsqueeze(0), xy_rel_id[i].unsqueeze(0))
                beta_loss = self.loss_anno_dict["matres"](beta[i][4:].unsqueeze(0), yz_rel_id[i].unsqueeze(0))
                gamma_loss = self.loss_anno_dict["matres"](gamma[i][4:].unsqueeze(0), xz_rel_id[i].unsqueeze(0))
            anno_loss += (alpha_loss + beta_loss + gamma_loss)
        return anno_loss.sum()

    def _get_trans_loss(self, alpha: Tensor, beta: Tensor, gamma: Tensor, flag: Tensor):
        batch_size = alpha.shape[0]
        # hier_loss = self.loss_func_trans(alpha[:, 0:4], beta[:, 0:4], gamma[:, 0:4])
        # temp_loss = self.loss_func_trans(alpha[:, 4:], beta[:, 4:], gamma[:, 4:])
        # trans_loss = self.lambda_dict["lambda_trans"] * (hier_loss + temp_loss)
        # return trans_loss.sum()
        trans_loss = 0
        for i in range(0, batch_size):
            if flag[i] == 0:  # HiEve
                trans_loss += self.loss_func_trans(alpha[i][0:4].unsqueeze(0), beta[i][0:4].unsqueeze(0), gamma[i][0:4].unsqueeze(0))
            elif flag[i] == 1:  # MATRES
                trans_loss += self.loss_func_trans(alpha[i][4:].unsqueeze(0), beta[i][4:].unsqueeze(0), gamma[i][4:].unsqueeze(0))
        return trans_loss.sum()

    def _get_symm_loss(self, alpha: Tensor, alpha_reverse: Tensor):
        hier_loss = self.loss_func_symm(alpha[:, 0:4], alpha_reverse[:, 0:4])
        temp_loss = self.loss_func_symm(alpha[:, 4:], alpha_reverse[:, 4:])
        symm_loss = self.lambda_dict["lambda_symm"] * (hier_loss + temp_loss)
        return symm_loss.sum()

    def _get_loss(self, batch_size: int, flag: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor, alpha_reverse: Tensor,
                       xy_rel_id: Tensor, yz_rel_id: Tensor, xz_rel_id: Tensor):
        anno_loss = 0
        trans_loss = 0
        symm_loss = 0
        for i in range(0, batch_size):
            if flag[i] == 0:    # HiEve
                alpha_loss = self.loss_anno_dict["hieve"](alpha[i][0:4].unsqueeze(0), xy_rel_id[i].unsqueeze(0))
                beta_loss = self.loss_anno_dict["hieve"](beta[i][0:4].unsqueeze(0), yz_rel_id[i].unsqueeze(0))
                gamma_loss = self.loss_anno_dict["hieve"](gamma[i][0:4].unsqueeze(0), xz_rel_id[i].unsqueeze(0))
                anno_loss += (alpha_loss + beta_loss + gamma_loss)
                symm_loss += self.loss_func_symm(alpha[i][0:4].unsqueeze(0), alpha_reverse[i][0:4].unsqueeze(0))
                trans_loss += self.loss_func_trans(alpha[i][0:4].unsqueeze(0), beta[i][0:4].unsqueeze(0), gamma[i][0:4].unsqueeze(0))
            elif flag[i] == 1:  # MATRES
                alpha_loss = self.loss_anno_dict["matres"](alpha[i][4:].unsqueeze(0), xy_rel_id[i].unsqueeze(0))
                beta_loss = self.loss_anno_dict["matres"](beta[i][4:].unsqueeze(0), yz_rel_id[i].unsqueeze(0))
                gamma_loss = self.loss_anno_dict["matres"](gamma[i][4:].unsqueeze(0), xz_rel_id[i].unsqueeze(0))
                anno_loss += (alpha_loss + beta_loss + gamma_loss)
                symm_loss += self.loss_func_symm(alpha[i][0:4].unsqueeze(0), alpha_reverse[i][0:4].unsqueeze(0))
                trans_loss += self.loss_func_trans(alpha[i][4:].unsqueeze(0), beta[i][4:].unsqueeze(0), gamma[i][4:].unsqueeze(0))
        return anno_loss.sum() + self.lambda_dict["lambda_symm"] * symm_loss.sum() + self.lambda_dict["lambda_trans"] * trans_loss.sum()


    def train(self):
        # if self.no_valid is False:
        #     self.evaluation()
        #     wandb.log({})
        full_start_time = time.time()
        self.model.zero_grad()
        for epoch in range(1, self.epochs+1):
            epoch_start_time = time.time()
            print()
            print('======== Epoch {:} / {:} ========'.format(epoch, self.epochs))
            print("Training start...")
            self.model.train()
            loss_vals = []
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                device = self.device
                xy_rel_id, yz_rel_id, xz_rel_id = batch[12].to(device), batch[13].to(device), batch[14].to(device)
                flag = batch[15]  # 0: HiEve, 1: MATRES
                batch_size = xy_rel_id.size(0)

                alpha, beta, gamma, alpha_reverse = self.model(batch, device)

                if self.data_type.lower() == "hieve":
                    loss = self.loss_anno_dict["hieve"](alpha, xy_rel_id) + self.loss_anno_dict["hieve"](beta, yz_rel_id) + self.loss_anno_dict["hieve"](gamma, xz_rel_id)
                    loss += self.loss_func_trans(alpha, beta, gamma).sum()
                elif self.data_type.lower() == "matres":
                    loss = self.loss_anno_dict["matres"](alpha, xy_rel_id) + self.loss_anno_dict["matres"](beta, yz_rel_id) + self.loss_anno_dict["matres"](gamma, xz_rel_id)
                    loss += self.loss_func_trans(alpha, beta, gamma).sum()
                elif self.data_type.lower() == "joint":
                    loss = self._get_loss(batch_size, flag, alpha, beta, gamma, alpha_reverse, xy_rel_id, yz_rel_id, xz_rel_id)
                    # loss = self._get_anno_loss(batch_size, flag, alpha, beta, gamma, xy_rel_id, yz_rel_id, xz_rel_id)
                    # loss += self._get_symm_loss(alpha, alpha_reverse)
                    # loss += self._get_trans_loss(alpha, beta, gamma, flag)
                    # loss += self.loss_func_cross(alpha, beta, gamma).sum()
                loss_vals.append(loss.item())
                loss.backward()
                self.opt.step()
            loss = sum(loss_vals) / len(loss_vals)
            print("loss:", loss)
            wandb.log(
                {
                    "[Train] Epoch": epoch,
                    "[Train] Loss": loss,
                    "[Train] Elapsed Time:": (time.time() - epoch_start_time)
                },
                commit=False,
            )

            # evaluate
            if self.no_valid is False:
                self.evaluation()

            wandb.log({})
        wandb.log({"Full Elapsed Time": (time.time() - full_start_time)})
        print("Training done!")

    def evaluation(self):

        if self.data_type.lower() == "hieve":
            hieve_metrics = self.evaluator.evaluate("hieve")
            wandb.log(hieve_metrics, commit=False)
            print("hieve_metrics:", hieve_metrics)

            f1_score = hieve_metrics["[HiEve] F1-PC-CP-AVG"]
            if self.best_f1_score < f1_score:
                self.best_f1_score = f1_score
            wandb.log({"[HiEve] Best F1 Score": self.best_f1_score}, commit=False)
        elif self.data_type.lower() == "matres":
            matres_metrics = self.evaluator.evaluate("matres")
            wandb.log(matres_metrics, commit=False)
            print("matres_metrics:", matres_metrics)

            f1_score = matres_metrics["[MATRES] F1 Score"]
            if self.best_f1_score < f1_score:
                self.best_f1_score = f1_score
            wandb.log({"[MATRES] Best F1 Score": self.best_f1_score}, commit=False)
        elif self.data_type.lower() == "joint":
            hieve_metrics = self.evaluator.evaluate("hieve")
            wandb.log(hieve_metrics, commit=False)
            print("hieve_metrics:", hieve_metrics)

            matres_metrics = self.evaluator.evaluate("matres")
            wandb.log(matres_metrics, commit=False)
            print("matres_metrics:", matres_metrics)

            f1_score = hieve_metrics["[HiEve] F1-PC-CP-AVG"] + matres_metrics["[MATRES] F1 Score"]
            if self.best_f1_score < f1_score:
                self.best_f1_score = f1_score
            wandb.log({"[Both] Best F1 Score": self.best_f1_score}, commit=False)

class Evaluator:
    def __init__(self, model: Module, device: torch.device, valid_dataloader_dict: Dict[str, DataLoader], test_dataloader_dict: Dict[str, DataLoader]):
        self.model = model
        self.device = device
        self.valid_dataloader_dict = valid_dataloader_dict
        self.test_dataloader_dict = test_dataloader_dict
        self.best_hieve_score = 0.0
        self.best_matres_score = 0.0

    def evaluate(self, type: str):
        dataloader = self.valid_dataloader_dict[type]
        self.model.eval()
        pred_vals, rel_ids = [], []
        eval_start_time = time.time()
        print(f"Validation-[{type}] start... ", end="")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                device = self.device
                xy_rel_id = batch[12].to(device)
                alpha, beta, gamma, alpha_reverse = self.model(batch, device)  # alpha: [16, 8]

                xy_rel_ids = xy_rel_id.to("cpu").numpy() # xy_rel_id: [16]
                pred = torch.max(alpha, 1).indices.cpu().numpy()

                # if type == "matres":
                #     pred = pred - 4
                #     pred[pred < 0] = 9
                print("pred:", pred, "true:", xy_rel_ids)
                pred_vals.extend(pred)
                rel_ids.extend(xy_rel_ids)

        if type == "hieve":
            metrics, result_table = metric(type, y_true=rel_ids, y_pred=pred_vals)
            assert metrics is not None
            print("result_table:", result_table)
            if self.best_hieve_score < metrics["[HiEve] F1-PC-CP-AVG"]:
                self.best_hieve_score = metrics["[HiEve] F1-PC-CP-AVG"]
            metrics["[HiEve] Best F1-PC-CP-AVG"] = self.best_hieve_score

        if type == "matres":
            metrics, CM = metric(type, y_true=rel_ids, y_pred=pred_vals)
            assert metrics is not None
            print("CM:", CM)
            if self.best_matres_score < metrics["[MATRES] F1 Score"]:
                self.best_matres_score = metrics["[MATRES] F1 Score"]
            metrics["[MATRES] Best F1 Score"] = self.best_matres_score

        print("done!")
        metrics["[Valid] Elapsed Time"] = (time.time() - eval_start_time)
        return metrics