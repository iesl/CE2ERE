import logging
import os
import time
import datetime
from pathlib import Path

import torch
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import Tensor, optim
from tqdm import tqdm
from typing import Dict, Union, Optional
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from evalulation import threshold_evalution, two_threshold_evalution
from loss import BCELossWithLog, BCELossWithLogP, BCELogitLoss
from metrics import metric, ConstraintViolation, CrossCategoryConstraintViolation

logger = logging.getLogger()

class Trainer:
    def __init__(self, data_type: str, model_type: str, model: Module, device: torch.device, epochs: int, learning_rate: float,
                 train_dataloader: DataLoader, evaluator: Module, opt: torch.optim.Optimizer, loss_type: int, loss_anno_dict: Dict[str, Module],
                 loss_transitivity_h: Module, loss_transitivity_t: Module, loss_cross_category: Module,
                 lambda_dict: Dict[str, float], no_valid: bool, debug: bool, cv_valid: int, model_save: int,
                 wandb_id: Optional[str] = "", eval_step: Optional[int] = 1, patience: Optional[int] = 8, max_grad_norm: Optional[float] = 5):
        self.data_type = data_type
        self.model_type = model_type
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
        self._get_trans_loss_h = loss_transitivity_h
        self._get_trans_loss_t = loss_transitivity_t
        self.loss_func_cross = loss_cross_category

        self.cross_entropy_loss = CrossEntropyLoss()
        self.bce_loss = BCELossWithLog()

        self.pbce_loss = BCELossWithLogP()
        self.bce_logit_loss = BCELogitLoss()
        self.no_valid = no_valid
        self.best_f1_score = 0.0
        self.best_epoch = -1
        self.eval_step = eval_step
        self.debug = debug
        self.patience = patience
        self.max_grad_norm = max_grad_norm

        self.cv_valid = cv_valid      # contraint evaluation flag. 0: false, 1: true
        self.model_save = model_save

        if self.model_save:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
            self.model_save_dir = "./model/"
            Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
            self.model_save_path = self.model_save_dir + f"{data_type}_{timestamp}_{wandb_id}.pt"

    def _update_save_best_score(self, f1_score: float, epoch: int):
        if self.best_f1_score < f1_score:
            self.best_f1_score = f1_score
            self.best_epoch = epoch
            if self.model_save:
                torch.save(self.model.state_dict(), self.model_save_path)
                logger.info("model is saved here: %s, best epoch: %s, best f1 score: %f"
                            % (os.path.abspath(self.model_save_path), self.best_epoch, self.best_f1_score))
            else:
                logger.info("best epoch: %s, best f1 score: %f" % (self.best_epoch, self.best_f1_score))

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

    def _get_trans_loss(self, alpha: Tensor, beta: Tensor, gamma: Tensor):
        hier_loss = self.loss_func_trans(alpha[:, 0:4], beta[:, 0:4], gamma[:, 0:4])
        temp_loss = self.loss_func_trans(alpha[:, 4:], beta[:, 4:], gamma[:, 4:])
        trans_loss = self.lambda_dict["lambda_trans"] * (hier_loss + temp_loss)
        return trans_loss.sum()

    def train(self, saved_model=None):
        full_start_time = time.time()
        self.model.zero_grad()
        if saved_model:
            # evaluate
            self.evaluation()
            wandb.log({})
        else:
            for epoch in range(1, self.epochs+1):
                epoch_start_time = time.time()
                self.model.train()
                logger.info("Training start...")
                logger.info("======== Epoch {:} / {:} ========".format(epoch, self.epochs))
                loss_vals = []
                for step, batch in enumerate(tqdm(self.train_dataloader)):
                    device = self.device
                    if self.model_type == "box":
                        xy_rel_id = torch.stack(batch[12], dim=-1).to(device) # [batch_size, 2]
                        flag = batch[15]  # 0: HiEve, 1: MATRES
                        vol_A_B, vol_B_A, _, _, _, _, pvol_AB, vol_mh = self.model(batch, device, self.data_type) # [batch_size, # of datasets]
                        loss = self.lambda_dict["lambda_condi"] * self.bce_loss(vol_A_B, vol_B_A, xy_rel_id, flag)
                        if self.loss_type == 1 or self.loss_type == 5:
                            loss += (1 - self.lambda_dict["lambda_condi"]) * self.pbce_loss(pvol_AB, xy_rel_id, flag)
                        if self.loss_type == 4:
                            loss += self.lambda_dict["lambda_pair"] * self.pbce_loss(pvol_AB, xy_rel_id, flag)
                        if self.loss_type == 2:
                            loss += self.lambda_dict["lambda_cross"] * -vol_mh.sum()
                        if self.loss_type == 3:
                            loss += self.lambda_dict["lambda_pair"] * self.pbce_loss(pvol_AB, xy_rel_id, flag)
                            loss += self.lambda_dict["lambda_cross"] * -vol_mh.sum()
                        assert not torch.isnan(loss)
                    elif self.model_type == "vector":
                        xy_rel_id = torch.stack(batch[12], dim=-1).to(device) # [batch_size, 2]
                        flag = batch[15]  # 0: HiEve, 1: MATRES
                        logits_A_B, logits_B_A, _, _, _, _ = self.model(batch, device, self.data_type) # [batch_size, # of datasets]
                        loss = self.bce_logit_loss(logits_A_B,logits_B_A, xy_rel_id, flag)
                        assert not torch.isnan(loss)
                    else:
                        xy_rel_id, yz_rel_id, xz_rel_id = batch[12].to(device), batch[13].to(device), batch[14].to(device)
                        flag = batch[15]  # 0: HiEve, 1: MATRES
                        batch_size = xy_rel_id.size(0)
                        alpha, beta, gamma = self.model(batch, device) # [batch_size, 8]

                        if self.data_type == "hieve":
                            loss = self.lambda_dict["lambda_anno"] * (self.loss_anno_dict["hieve"](alpha, xy_rel_id) + self.loss_anno_dict["hieve"](beta, yz_rel_id) + self.loss_anno_dict["hieve"](gamma, xz_rel_id))
                        elif self.data_type == "matres":
                            loss = self.lambda_dict["lambda_anno"] * (self.loss_anno_dict["matres"](alpha, xy_rel_id) + self.loss_anno_dict["matres"](beta, yz_rel_id) + self.loss_anno_dict["matres"](gamma, xz_rel_id))
                        elif self.data_type == "joint":
                            loss = self.lambda_dict["lambda_anno"] * self._get_anno_loss(batch_size, flag, alpha, beta, gamma, xy_rel_id, yz_rel_id, xz_rel_id)
                            if self.loss_type:
                                loss += self.lambda_dict["lambda_trans"] * self._get_trans_loss_h(alpha[:, 0:4], beta[:, 0:4], gamma[:, 0:4]).sum()
                                loss += self.lambda_dict["lambda_trans"] * self._get_trans_loss_t(alpha[:, 4:8], beta[:, 4:8], gamma[:, 4:8]).sum()
                                if self.loss_type == 2:
                                    loss += (self.lambda_dict["lambda_cross"] * self.loss_func_cross(alpha, beta, gamma)).sum()

                    loss_vals.append(loss.item())
                    loss.backward()
                    if self.model_type == "box" or self.model_type == "vector":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step()

                loss = sum(loss_vals) / len(loss_vals)
                logger.info("epoch: %d, loss: %f" % (epoch, loss))
                wandb.log(
                    {
                        "[Train] Epoch": epoch,
                        "[Train] Loss": loss,
                        "[Train] Elapsed Time:": (time.time() - epoch_start_time)
                    },
                    commit=False,
                )

                # evaluate
                if epoch % self.eval_step == 0 and self.no_valid is False:
                    self.evaluation(epoch)
                    if (epoch - self.best_epoch) >= self.patience:
                        print(f"\nAccuracy has not changed in {self.patience} steps! Stopping the run after final evaluation...")
                        break
                wandb.log({})

            self.evaluation(epoch)
            wandb.log({})
            wandb.log({"Full Elapsed Time": (time.time() - full_start_time)})
            logger.info("Training done!")

    def evaluation(self, epoch: Optional[int] = -1):
        valid_metrics, test_metrics = {}, {}
        cv_valid_metrics, cv_test_metrics = {}, {} # constraint violation dict
        if self.data_type == "hieve" or self.data_type == "joint":
            valid_metric = self.evaluator.evaluate("hieve", "valid")
            valid_metrics.update(valid_metric)

            if not self.debug:
                test_metric = self.evaluator.evaluate("hieve", "test")
                test_metrics.update(test_metric)
                # cv_test_metric = self.evaluator.evaluate("hieve", "cv-test")
                # cv_test_metrics.update(cv_test_metric)

        if self.data_type == "matres" or self.data_type == "joint":
            valid_metric = self.evaluator.evaluate("matres", "valid")
            valid_metrics.update(valid_metric)

            if not self.debug:
                test_metric = self.evaluator.evaluate("matres", "test")
                test_metrics.update(test_metric)
                # cv_test_metric = self.evaluator.evaluate("matres", "cv-test")
                # cv_test_metrics.update(cv_test_metric)

        logger.info("valid_metrics: {0}".format(valid_metrics))
        wandb.log(valid_metrics, commit=False)

        if not self.debug:
            logger.info("test_metrics: {0}".format(test_metrics))
            logger.info("cv_test_metrics: {0}".format(cv_test_metrics))
            wandb.log(test_metrics, commit=False)

        if self.data_type == "hieve":   # single task
            f1_score = valid_metrics[f"[valid-{self.data_type}] F1 Score (PC-CP)"]
        elif self.data_type == "matres":
            f1_score = valid_metrics[f"[valid-{self.data_type}] F1 Score"]
        else:                           # joint task
            f1_score = valid_metrics[f"[valid-hieve] F1 Score (PC-CP)"] + valid_metrics[f"[valid-matres] F1 Score"]
            # cross category constraint violation evaluation
            # logger.info("Cross Category Constraint Violation Evalution starts...")
            # cross_cv_eval = CrossCategoryConstraintViolation(self.model_type)
            # h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list = self.evaluator.cross_evaluate("hieve", "cv-test")
            # assert len(h_cv_xy_list) == len(h_cv_yz_list) == len(h_cv_xz_list) \
            #        == len(m_cv_xy_list) == len(m_cv_yz_list) == len(m_cv_xz_list)
            # if self.model_type == "box" or self.model_type == "vector":
            #     cross_cv_eval.update_violation_count_box(h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list)
            # else:
            #     cross_cv_eval.update_violation_count_vector(h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list)
            #
            # h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list = self.evaluator.cross_evaluate("matres", "cv-test")
            # assert len(h_cv_xy_list) == len(h_cv_yz_list) == len(h_cv_xz_list) \
            #        == len(m_cv_xy_list) == len(m_cv_yz_list) == len(m_cv_xz_list)
            # if self.model_type == "box" or self.model_type == "vector":
            #     cross_cv_eval.update_violation_count_box(h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list)
            # else:
            #     cross_cv_eval.update_violation_count_vector(h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list)
            #
            # logger.info(f"cross constraint-violation: %s" % cross_cv_eval.violation_dict)
            # logger.info(f"cross cv all_cases: %s, total count: %s" % (cross_cv_eval.all_case_count, sum(cross_cv_eval.all_case_count.values())))
            # logger.info("done!")
        self._update_save_best_score(f1_score, epoch)
        wandb.log({f"[{self.data_type}] Best F1 Score": self.best_f1_score}, commit=False)


class ThresholdEvaluator:
    def __init__(self, train_type: str, model_type: str, model: Module, device: torch.device,
                 valid_dataloader_dict: Dict[str, DataLoader], test_dataloader_dict: Dict[str, DataLoader],
                 valid_cv_dataloader_dict: Dict[str, DataLoader], test_cv_dataloader_dict: Dict[str, DataLoader],
                 eval_type: str, save_plot: int, threshold1: float, threshold2: Optional[float]=-0.5,
                 threshold3: Optional[float]=0, threshold4: Optional[float]=0, wandb_id: Optional[str]=""):
        self.train_type = train_type
        self.model_type = model_type
        self.model = model
        self.device = device
        self.valid_dataloader_dict = valid_dataloader_dict
        self.test_dataloader_dict = test_dataloader_dict
        self.valid_cv_dataloader_dict = valid_cv_dataloader_dict
        self.test_cv_dataloader_dict = test_cv_dataloader_dict
        self.best_hieve_score = 0.0
        self.best_matres_score = 0.0

        self.evaluator = eval_type
        if self.evaluator == "one":
            self.hieve_threshold = threshold1
            self.matres_threshold = threshold1
        elif self.evaluator == "two":
            self.hieve_threshold1 = threshold1
            self.hieve_threshold2 = threshold2
            self.matres_threshold1 = threshold1
            self.matres_threshold2 = threshold2
        elif self.evaluator == "four":
            self.hieve_threshold1 = threshold1
            self.hieve_threshold2 = threshold2
            self.matres_threshold1 = threshold3
            self.matres_threshold2 = threshold4

        self.save_plot = save_plot
        if self.save_plot:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d%H%M%S")
            self.fig_save_dir = "./figures/" + f"{self.train_type}_{timestamp}_{wandb_id}/"
            Path(self.fig_save_dir).mkdir(parents=True, exist_ok=True)

    def create_disttribution_plot(self, eval_type, prob1, prob2, p1_name, p2_name, rids, target):
        rids = np.array(rids)
        rids_index = (rids == target).nonzero()[0]
        prob1 = np.array(prob1)[rids_index]
        prob2 = np.array(prob2)[rids_index]

        # print("========================================================================")
        # print(target)
        # print(rids)
        # print("rids_index1", rids_index1)
        # print("rids_index2", rids_index2)
        # print(p1_name, prob1)
        # print(p2_name, prob2)
        # print("========================================================================")

        # df = pd.DataFrame()
        # df['index'] = rids_index
        # df[p1_name] = prob1
        # df[p2_name] = prob2

        # df = pd.melt(df, id_vars="index", var_name="type", value_name="prob")
        # sns.catplot(x='index', y='prob', hue='type', data=df, kind='bar')
        # plt.savefig(self.fig_save_dir + f"{p1_name}_{target}_distribution.png")
        # plt.clf()

        fig, axs = plt.subplots(2)
        counts1, bins1 = np.histogram(prob1)
        counts2, bins2 = np.histogram(prob2)
        axs[0].hist(bins1[:-1], bins1, weights=counts1)
        axs[0].set_ylabel("count")

        axs[1].hist(bins2[:-1], bins2, weights=counts2)
        axs[1].set_xlabel("probability")
        axs[1].set_ylabel("count")
        axs[0].set_title(f"{p1_name}[top] and {p2_name}[bottom]-a:{target[0]},b:{target[1]}")

        plt.savefig(self.fig_save_dir + f"{eval_type}_{p1_name}_{p2_name}_{target}_frequency.png")
        plt.clf()

    def evaluate(self, data_type: str, eval_type: str):
        if eval_type == "valid":
            dataloader = self.valid_dataloader_dict[data_type]
            constraint_violation = None
        elif eval_type == "test":
            dataloader = self.test_dataloader_dict[data_type]
            constraint_violation = None
        elif eval_type == "cv-valid":
            dataloader = self.valid_cv_dataloader_dict[data_type]
            constraint_violation = ConstraintViolation(self.model_type)
        elif eval_type == "cv-test":
            dataloader = self.test_cv_dataloader_dict[data_type]
            constraint_violation = ConstraintViolation(self.model_type)
        self.model.eval()
        preds, targets = [], []
        vol_ab, vol_bc, vol_ac = [], [], []
        vol_ba, vol_cb, vol_ca = [], [], []
        rids = []
        eval_start_time = time.time()
        logger.info(f"[{eval_type}-{data_type}] start... ")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                device = self.device
                xy_rel_id = torch.stack(batch[12], dim=-1).to(device) # [batch_size, 2]
                yz_rel_id = torch.stack(batch[13], dim=-1).to(device)
                xz_rel_id = torch.stack(batch[14], dim=-1).to(device)
                flag = batch[15]  # 0: HiEve, 1: MATRES
                vol_A_B, vol_B_A, vol_B_C, vol_C_B, vol_A_C, vol_C_A, _, _ = self.model(batch, device, self.train_type) # [batch_size, 2]

                if vol_A_B.shape[-1] == 2:
                    if data_type == "hieve":
                        vol_A_B, vol_B_A = vol_A_B[:, 0][flag == 0], vol_B_A[:, 0][flag == 0]  # [batch_size]
                        vol_B_C, vol_C_B = vol_B_C[:, 0][flag == 0], vol_C_B[:, 0][flag == 0]
                        vol_A_C, vol_C_A = vol_A_C[:, 0][flag == 0], vol_C_A[:, 0][flag == 0]
                        xy_rel_id = xy_rel_id[flag == 0]
                        yz_rel_id = yz_rel_id[flag == 0]
                        xz_rel_id = xz_rel_id[flag == 0]
                    elif data_type == "matres":
                        vol_A_B, vol_B_A = vol_A_B[:, 1][flag == 1], vol_B_A[:, 1][flag == 1]
                        vol_B_C, vol_C_B = vol_B_C[:, 1][flag == 1], vol_C_B[:, 1][flag == 1]
                        vol_A_C, vol_C_A = vol_A_C[:, 1][flag == 1], vol_C_A[:, 1][flag == 1]
                        xy_rel_id = xy_rel_id[flag == 1]
                        yz_rel_id = yz_rel_id[flag == 1]
                        xz_rel_id = xz_rel_id[flag == 1]
                else:
                    vol_A_B, vol_B_A = vol_A_B.squeeze(1), vol_B_A.squeeze(1)  # [batch_size]
                    vol_B_C, vol_C_B = vol_B_C.squeeze(1), vol_C_B.squeeze(1)
                    vol_A_C, vol_C_A = vol_A_C.squeeze(1), vol_C_A.squeeze(1)

                if self.evaluator == "one":
                    if data_type == "hieve":
                        threshold = self.hieve_threshold
                    if data_type == "matres":
                        threshold = self.matres_threshold
                    xy_preds, xy_targets, xy_constraint_dict = threshold_evalution(vol_A_B, vol_B_A, xy_rel_id, threshold)
                    yz_preds, yz_targets, yz_constraint_dict = threshold_evalution(vol_B_C, vol_C_B, yz_rel_id, threshold)
                    xz_preds, xz_targets, xz_constraint_dict = threshold_evalution(vol_A_C, vol_C_A, xz_rel_id, threshold)
                elif self.evaluator == "two" or self.evaluator == "four":
                    if data_type == "hieve":
                        threshold1 = self.hieve_threshold1
                        threshold2 = self.hieve_threshold2
                    if data_type == "matres":
                        threshold1 = self.matres_threshold1
                        threshold2 = self.matres_threshold2
                    xy_preds, xy_targets, xy_constraint_dict = two_threshold_evalution(vol_A_B, vol_B_A, xy_rel_id, threshold1, threshold2)
                    yz_preds, yz_targets, yz_constraint_dict = two_threshold_evalution(vol_B_C, vol_C_B, yz_rel_id, threshold1, threshold2)
                    xz_preds, xz_targets, xz_constraint_dict = two_threshold_evalution(vol_A_C, vol_C_A, xz_rel_id, threshold1, threshold2)

                assert len(xy_preds) == len(xy_targets)
                preds.extend(xy_preds)
                targets.extend(xy_targets)
                vol_ab.extend(torch.exp(vol_A_B).tolist())
                vol_ba.extend(torch.exp(vol_B_A).tolist())
                rids.extend([''.join(map(str, item)) for item in xy_rel_id.tolist()])

                if constraint_violation:
                    constraint_violation.update_violation_count_box(xy_constraint_dict, yz_constraint_dict, xz_constraint_dict)

            if constraint_violation:
                logger.info(f"[{eval_type}-{data_type}] constraint-violation: %s" % constraint_violation.violation_dict)
                logger.info(f"[{eval_type}-{data_type}] all_cases: %s" % constraint_violation.all_case_count)

        metrics = metric(data_type, eval_type, self.model_type, y_true=targets, y_pred=preds)
        logger.info("done!")
        metrics[f"[{eval_type}] Elapsed Time"] = (time.time() - eval_start_time)

        ####### plot for conditional probabilities #######
        if (eval_type == "valid" or eval_type == "test") and self.save_plot:
            for label in ["10", "01", "11", "00"]:
                self.create_disttribution_plot(eval_type, vol_ab, vol_ba, "vol_ab", "vol_ba", rids, label)
                logger.info("# of {0} labels: {1}".format(label, len((np.array(rids)==label).nonzero()[0])))
        return metrics

    def cross_evaluate(self, data_type: str, eval_type: str):
        if eval_type == "cv-test":
            dataloader = self.test_cv_dataloader_dict[data_type]
        else:
            raise ValueError("Invalid evaluation type")

        self.model.eval()
        h_cv_xy_list, h_cv_yz_list, h_cv_xz_list = [], [], []
        m_cv_xy_list, m_cv_yz_list, m_cv_xz_list = [], [], []
        eval_start_time = time.time()
        logger.info(f"[{eval_type}-{data_type}] start... ")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                device = self.device
                xy_rel_id = torch.stack(batch[12], dim=-1).to(device) # [batch_size, 2]
                yz_rel_id = torch.stack(batch[13], dim=-1).to(device)
                xz_rel_id = torch.stack(batch[14], dim=-1).to(device)
                flag = batch[15]  # 0: HiEve, 1: MATRES
                vol_A_B, vol_B_A, vol_B_C, vol_C_B, vol_A_C, vol_C_A, _, _ = self.model(batch, device, self.train_type) # [batch_size, 2]

                if vol_A_B.shape[-1] == 2:
                    if data_type == "hieve":
                        h_vol_A_B, h_vol_B_A = vol_A_B[:, 0], vol_B_A[:, 0] # [batch_size]
                        h_vol_B_C, h_vol_C_B = vol_B_C[:, 0], vol_C_B[:, 0]
                        h_vol_A_C, h_vol_C_A = vol_A_C[:, 0], vol_C_A[:, 0]
                        if self.evaluator == "one":
                            h_xy_preds, h_xy_targets, h_xy_constraint_dict = threshold_evalution(h_vol_A_B, h_vol_B_A, xy_rel_id, self.hieve_threshold)
                            h_yz_preds, h_yz_targets, h_yz_constraint_dict = threshold_evalution(h_vol_B_C, h_vol_C_B, yz_rel_id, self.hieve_threshold)
                            h_xz_preds, h_xz_targets, h_xz_constraint_dict = threshold_evalution(h_vol_A_C, h_vol_C_A, xz_rel_id, self.hieve_threshold)
                        elif self.evaluator == "two" or self.evaluator == "four":
                            h_xy_preds, h_xy_targets, h_xy_constraint_dict = two_threshold_evalution(h_vol_A_B, h_vol_B_A, xy_rel_id, self.hieve_threshold1, self.hieve_threshold2)
                            h_yz_preds, h_yz_targets, h_yz_constraint_dict = two_threshold_evalution(h_vol_B_C, h_vol_C_B, yz_rel_id, self.hieve_threshold1, self.hieve_threshold2)
                            h_xz_preds, h_xz_targets, h_xz_constraint_dict = two_threshold_evalution(h_vol_A_C, h_vol_C_A, xz_rel_id, self.hieve_threshold1, self.hieve_threshold2)
                        m_vol_A_B, m_vol_B_A = vol_A_B[:, 1], vol_B_A[:, 1]
                        m_vol_B_C, m_vol_C_B = vol_B_C[:, 1], vol_C_B[:, 1]
                        m_vol_A_C, m_vol_C_A = vol_A_C[:, 1], vol_C_A[:, 1]
                        if self.evaluator == "one":
                            m_xy_preds, m_xy_targets, m_xy_constraint_dict = threshold_evalution(m_vol_A_B, m_vol_B_A, xy_rel_id, self.matres_threshold)
                            m_yz_preds, m_yz_targets, m_yz_constraint_dict = threshold_evalution(m_vol_B_C, m_vol_C_B, yz_rel_id, self.matres_threshold)
                            m_xz_preds, m_xz_targets, m_xz_constraint_dict = threshold_evalution(m_vol_A_C, m_vol_C_A, xz_rel_id, self.matres_threshold)
                        elif self.evaluator == "two" or self.evaluator == "four":
                            m_xy_preds, m_xy_targets, m_xy_constraint_dict = two_threshold_evalution(m_vol_A_B, m_vol_B_A, xy_rel_id, self.matres_threshold1, self.matres_threshold2)
                            m_yz_preds, m_yz_targets, m_yz_constraint_dict = two_threshold_evalution(m_vol_B_C, m_vol_C_B, yz_rel_id, self.matres_threshold1, self.matres_threshold2)
                            m_xz_preds, m_xz_targets, m_xz_constraint_dict = two_threshold_evalution(m_vol_A_C, m_vol_C_A, xz_rel_id, self.matres_threshold1, self.matres_threshold2)
                    elif data_type == "matres":
                        h_vol_A_B, h_vol_B_A = vol_A_B[:, 0], vol_B_A[:, 0]
                        h_vol_B_C, h_vol_C_B = vol_B_C[:, 0], vol_C_B[:, 0]
                        h_vol_A_C, h_vol_C_A = vol_A_C[:, 0], vol_C_A[:, 0]
                        if self.evaluator == "one":
                            h_xy_preds, h_xy_targets, h_xy_constraint_dict = threshold_evalution(h_vol_A_B, h_vol_B_A, xy_rel_id, self.hieve_threshold)
                            h_yz_preds, h_yz_targets, h_yz_constraint_dict = threshold_evalution(h_vol_B_C, h_vol_C_B, yz_rel_id, self.hieve_threshold)
                            h_xz_preds, h_xz_targets, h_xz_constraint_dict = threshold_evalution(h_vol_A_C, h_vol_C_A, xz_rel_id, self.hieve_threshold)
                        elif self.evaluator == "two" or self.evaluator == "four":
                            h_xy_preds, h_xy_targets, h_xy_constraint_dict = two_threshold_evalution(h_vol_A_B, h_vol_B_A, xy_rel_id, self.hieve_threshold1, self.hieve_threshold2)
                            h_yz_preds, h_yz_targets, h_yz_constraint_dict = two_threshold_evalution(h_vol_B_C, h_vol_C_B, yz_rel_id, self.hieve_threshold1, self.hieve_threshold2)
                            h_xz_preds, h_xz_targets, h_xz_constraint_dict = two_threshold_evalution(h_vol_A_C, h_vol_C_A, xz_rel_id, self.hieve_threshold1, self.hieve_threshold2)

                        m_vol_A_B, m_vol_B_A = vol_A_B[:, 1], vol_B_A[:, 1]
                        m_vol_B_C, m_vol_C_B = vol_B_C[:, 1], vol_C_B[:, 1]
                        m_vol_A_C, m_vol_C_A = vol_A_C[:, 1], vol_C_A[:, 1]
                        if self.evaluator == "one":
                            m_xy_preds, m_xy_targets, m_xy_constraint_dict = threshold_evalution(m_vol_A_B, m_vol_B_A, xy_rel_id, self.matres_threshold)
                            m_yz_preds, m_yz_targets, m_yz_constraint_dict = threshold_evalution(m_vol_B_C, m_vol_C_B, yz_rel_id, self.matres_threshold)
                            m_xz_preds, m_xz_targets, m_xz_constraint_dict = threshold_evalution(m_vol_A_C, m_vol_C_A, xz_rel_id, self.matres_threshold)
                        elif self.evaluator == "two" or self.evaluator == "four":
                            m_xy_preds, m_xy_targets, m_xy_constraint_dict = two_threshold_evalution(m_vol_A_B, m_vol_B_A, xy_rel_id, self.matres_threshold1, self.matres_threshold2)
                            m_yz_preds, m_yz_targets, m_yz_constraint_dict = two_threshold_evalution(m_vol_B_C, m_vol_C_B, yz_rel_id, self.matres_threshold1, self.matres_threshold2)
                            m_xz_preds, m_xz_targets, m_xz_constraint_dict = two_threshold_evalution(m_vol_A_C, m_vol_C_A, xz_rel_id, self.matres_threshold1, self.matres_threshold2)

                    h_cv_xy_list.append(h_xy_constraint_dict)
                    h_cv_yz_list.append(h_yz_constraint_dict)
                    h_cv_xz_list.append(h_xz_constraint_dict)

                    m_cv_xy_list.append(m_xy_constraint_dict)
                    m_cv_yz_list.append(m_yz_constraint_dict)
                    m_cv_xz_list.append(m_xz_constraint_dict)
        logger.info("Cross Evaluation Elapsed Time: %s" % (time.time() - eval_start_time))
        return h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list


class VectorBiLSTMEvaluator:
    def __init__(self, train_type: str, model_type: str, model: Module, device: torch.device,
                 valid_dataloader_dict: Dict[str, DataLoader], test_dataloader_dict: Dict[str, DataLoader],
                 valid_cv_dataloader_dict: Dict[str, DataLoader], test_cv_dataloader_dict: Dict[str, DataLoader]):
        self.train_type = train_type
        self.model_type = model_type
        self.model = model
        self.device = device
        self.valid_dataloader_dict = valid_dataloader_dict
        self.test_dataloader_dict = test_dataloader_dict
        self.valid_cv_dataloader_dict = valid_cv_dataloader_dict
        self.test_cv_dataloader_dict = test_cv_dataloader_dict
        self.best_hieve_score = 0.0
        self.best_matres_score = 0.0

    def evaluate(self, data_type: str, eval_type: str):
        if eval_type == "valid":
            dataloader = self.valid_dataloader_dict[data_type]
            constraint_violation = None
        elif eval_type == "test":
            dataloader = self.test_dataloader_dict[data_type]
            constraint_violation = None
        elif eval_type == "cv-valid":
            dataloader = self.valid_cv_dataloader_dict[data_type]
            constraint_violation = ConstraintViolation(self.model_type)
        elif eval_type == "cv-test":
            dataloader = self.test_cv_dataloader_dict[data_type]
            constraint_violation = ConstraintViolation(self.model_type)
        self.model.eval()
        pred_vals, rel_ids = [], []
        rids = []
        eval_start_time = time.time()
        logger.info(f"[{eval_type}-{data_type}] start... ")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                device = self.device

                xy_rel_id = batch[12].to(device)
                alpha, beta, gamma = self.model(batch, device)  # alpha: [16, 8]
                xy_rel_ids = xy_rel_id.to("cpu").numpy() # xy_rel_id: [16]

                if self.train_type == "hieve" or self.train_type == "matres":
                    pred = torch.max(alpha, 1).indices.cpu().numpy()  # alpha: [16, 4]
                    alpha_indices = torch.max(alpha, 1).indices
                    beta_indices = torch.max(beta, 1).indices
                    gamma_indices = torch.max(gamma, 1).indices
                else:
                    if data_type == "hieve":
                        pred = torch.max(alpha[:, 0:4], 1).indices.cpu().numpy() # [16, 4]
                        alpha_indices = torch.max(alpha[:, 0:4], 1).indices
                        beta_indices = torch.max(beta[:, 0:4], 1).indices
                        gamma_indices = torch.max(gamma[:, 0:4], 1).indices
                    elif data_type == "matres":
                        pred = torch.max(alpha[:, 4:8], 1).indices.cpu().numpy() # [16, 4]
                        alpha_indices = torch.max(alpha[:, 4:8], 1).indices
                        beta_indices = torch.max(beta[:, 4:8], 1).indices
                        gamma_indices = torch.max(gamma[:, 4:8], 1).indices

                alpha_indices = [val.item() for val in alpha_indices]
                beta_indices = [val.item() for val in beta_indices]
                gamma_indices = [val.item() for val in gamma_indices]

                pred_vals.extend(pred)
                rel_ids.extend(xy_rel_ids)
                rids.extend(xy_rel_ids.tolist())
                if constraint_violation:
                    constraint_violation.update_violation_count_vector(alpha_indices, beta_indices, gamma_indices)

            if constraint_violation:
                logger.info(f"[{eval_type}-{data_type}] constraint-violation: %s" % constraint_violation.violation_dict)
                logger.info(f"[{eval_type}-{data_type}] all_cases: %s" % constraint_violation.all_case_count)

        metrics = metric(data_type, eval_type, self.model_type, y_true=rel_ids, y_pred=pred_vals)
        logger.info("done!")
        metrics[f"[{eval_type}] Elapsed Time"] = (time.time() - eval_start_time)
        ####### plot for conditional probabilities #######
        if (eval_type == "valid" or eval_type == "test"):
            for label in [0, 1, 2, 3]:
                logger.info("# of {0} labels: {1}".format(label, len((np.array(rids) == label).nonzero()[0])))
        return metrics

    def cross_evaluate(self, data_type: str, eval_type: str):
        if eval_type == "cv-test":
            dataloader = self.test_cv_dataloader_dict[data_type]
        else:
            raise ValueError("Invalid evaluation type")

        self.model.eval()
        h_cv_xy_list, h_cv_yz_list, h_cv_xz_list = [], [], []
        m_cv_xy_list, m_cv_yz_list, m_cv_xz_list = [], [], []
        eval_start_time = time.time()
        logger.info(f"[{eval_type}-{data_type}] start... ")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                device = self.device
                alpha, beta, gamma = self.model(batch, device)

                if self.train_type == "joint":
                    h_alpha_indices = torch.max(alpha[:, 0:4], 1).indices
                    h_beta_indices = torch.max(beta[:, 0:4], 1).indices
                    h_gamma_indices = torch.max(gamma[:, 0:4], 1).indices

                    h_alpha_indices = [val.item() for val in h_alpha_indices]
                    h_beta_indices = [val.item() for val in h_beta_indices]
                    h_gamma_indices = [val.item() for val in h_gamma_indices]

                    m_alpha_indices = torch.max(alpha[:, 4:8], 1).indices
                    m_beta_indices = torch.max(beta[:, 4:8], 1).indices
                    m_gamma_indices = torch.max(gamma[:, 4:8], 1).indices

                    m_alpha_indices = [val.item() for val in m_alpha_indices]
                    m_beta_indices = [val.item() for val in m_beta_indices]
                    m_gamma_indices = [val.item() for val in m_gamma_indices]

                    h_cv_xy_list.extend(h_alpha_indices)
                    h_cv_yz_list.extend(h_beta_indices)
                    h_cv_xz_list.extend(h_gamma_indices)

                    m_cv_xy_list.extend(m_alpha_indices)
                    m_cv_yz_list.extend(m_beta_indices)
                    m_cv_xz_list.extend(m_gamma_indices)
        logger.info("Cross Evaluation Elapsed Time: %s" % (time.time() - eval_start_time))
        return h_cv_xy_list, h_cv_yz_list, h_cv_xz_list, m_cv_xy_list, m_cv_yz_list, m_cv_xz_list