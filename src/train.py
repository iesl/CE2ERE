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
    def __init__(self, model: Module, device: torch.device, epochs: int, learning_rate: float, train_dataloader: DataLoader, evaluator: Module,
                 opt: torch.optim.Optimizer, loss_anno_dict: Dict[str, Module], loss_transitivity: Module,
                 loss_cross_category: Module, lambda_dict: Dict[str, float], roberta_size_type="roberta-base"):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lambda_dict = lambda_dict
        self.train_dataloader = train_dataloader

        self.evaluator = evaluator
        self.opt = opt

        self.loss_anno_dict = loss_anno_dict
        self.loss_func_trans = loss_transitivity
        self.loss_func_cross = loss_cross_category

        self.roberta_size_type = roberta_size_type
        if self.roberta_size_type == "roberta-base":
            self.roberta_dim = 768
        else:
            self.roberta_dim = 1024

    def _get_anno_loss(self, batch_size: int, flag: Tensor, alpha: Tensor, beta: Tensor, gamma: Tensor,
                       xy_rel_id: Tensor, yz_rel_id: Tensor, xz_rel_id: Tensor):
        anno_loss = 0
        for i in range(0, batch_size):
            if flag[i] == 0:    # HiEve
                alpha_loss = self.loss_anno_dict["hieve"](alpha[i][0:4].unsqueeze(0), xy_rel_id[i].unsqueeze(0))
                beta_loss = self.loss_anno_dict["hieve"](beta[i][0:4].unsqueeze(0), yz_rel_id[i].unsqueeze(0))
                gamma_loss = self.loss_anno_dict["hieve"](gamma[i][0:4].unsqueeze(0), xz_rel_id[i].unsqueeze(0))
                anno_loss += self.lambda_dict["lambda_annoH"] * (alpha_loss + beta_loss + gamma_loss)
            elif flag[i] == 1:  # MATRES
                alpha_loss = self.loss_anno_dict["matres"](alpha[i][4:].unsqueeze(0), xy_rel_id[i].unsqueeze(0))
                beta_loss = self.loss_anno_dict["matres"](beta[i][4:].unsqueeze(0), yz_rel_id[i].unsqueeze(0))
                gamma_loss = self.loss_anno_dict["matres"](gamma[i][4:].unsqueeze(0), xz_rel_id[i].unsqueeze(0))
                anno_loss += self.lambda_dict["lambda_annoT"] * (alpha_loss + beta_loss + gamma_loss)
        return anno_loss

    def _get_trans_loss(self, alpha: Tensor, beta: Tensor, gamma: Tensor):
        hier_loss = self.lambda_dict["lambda_transH"] * self.loss_func_trans(alpha[0:4], beta[0:4], gamma[0:4])
        temp_loss = self.lambda_dict["lambda_transT"] * self.loss_func_trans(alpha[4:], beta[4:], gamma[4:])
        trans_loss = hier_loss + temp_loss
        return trans_loss

    def train(self):
        self.evaluation()
        for epoch in range(1, self.epochs+1):
            epoch_start_time = time.time()

            self.model.train()
            loss_vals = []
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                # print("batch:", batch)
                device = self.device
                xy_rel_id, yz_rel_id, xz_rel_id = batch[12].to(device), batch[13].to(device), batch[14].to(device)
                flag = batch[15]  # 0: HiEve, 1: MATRES
                batch_size = xy_rel_id.size(0)

                alpha, beta, gamma = self.model(batch, device)

                anno_loss = self._get_anno_loss(batch_size, flag, alpha, beta, gamma, xy_rel_id, yz_rel_id, xz_rel_id)
                trans_loss = self._get_trans_loss(alpha, beta, gamma)
                cross_loss = self.loss_func_cross(alpha, beta, gamma)
                loss = anno_loss + trans_loss + cross_loss
                loss_vals.append(loss.item())

                assert not torch.isnan(loss).any()
                loss.backward()
                for param in self.model.parameters():
                    if param.grad is not None:
                        assert not torch.isnan(param.grad).any()
                self.opt.step()
            loss = sum(loss_vals) / len(self.train_dataloader)
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
            self.evaluation()

    def evaluation(self):
        hieve_metrics = self.evaluator.evaluate("hieve")
        wandb.log(hieve_metrics, commit=False)
        print("hieve_metrics:", hieve_metrics)

        matres_metrics = self.evaluator.evaluate("matres")
        wandb.log(matres_metrics, commit=False)
        print("matres_metrics:", matres_metrics)

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
                alpha, beta, gamma = self.model(batch, device)  # alpha: [16, 8]

                xy_rel_ids = xy_rel_id.to("cpu").numpy() # xy_rel_id: [16]
                pred = torch.max(alpha, 1).indices.cpu().numpy()
                pred_vals.extend(pred)
                rel_ids.extend(xy_rel_ids)

        if type == "hieve":
            metrics, result_table = metric(type, y_true=rel_ids, y_pred=pred_vals)
            assert metrics is not None and result_table is not None
            print("result_table:", result_table)
            if self.best_hieve_score < metrics["[HiEve] F1-PC-CP-AVG"]:
                self.best_hieve_score = metrics["[HiEve] F1-PC-CP-AVG"]
            metrics["[HiEve] Best F1-PC-CP-AVG"] = self.best_hieve_score

        if type == "matres":
            metrics, CM = metric(type, y_true=rel_ids, y_pred=pred_vals)
            assert metrics is not None and CM is not None
            print("CM:", CM)
            if self.best_matres_score < metrics["[MATRES] F1 Score"]:
                self.best_matres_score = metrics["[MATRES] F1 Score"]
            metrics["[MATRES] Best F1 Score"] = self.best_matres_score

        print("done!")
        metrics["[Valid] Elapsed Time"] = (time.time() - eval_start_time)
        return metrics