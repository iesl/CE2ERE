import logging
import time
import datetime
from pathlib import Path

import torch
import wandb
from torch import Tensor
from tqdm import tqdm
from typing import Dict, Union, Optional
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader

from evalulation import threshold_evalution, two_threshold_evalution
from loss import BCELossWithLog, BCELogitLoss
from metrics import metric, ConstraintViolation

logger = logging.getLogger()

class Trainer:
    def __init__(self, data_type: str, model_type: str, model: Module, device: torch.device, epochs: int, learning_rate: float,
                 train_dataloader: DataLoader, evaluator: Module, opt: torch.optim.Optimizer, loss_type: int, loss_anno_dict: Dict[str, Module],
                 loss_transitivity_h: Module, loss_transitivity_t: Module, loss_cross_category: Module, lambda_dict: Dict[str, float], no_valid: bool,
                 debug: bool, wandb_id: Optional[str] = "", eval_step: Optional[int] = 1, patience: Optional[int] = 8):
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
        self.bce_logit_loss = BCELogitLoss()
        self.no_valid = no_valid
        self.best_f1_score = 0.0
        self.best_epoch = -1
        self.eval_step = eval_step
        self.debug = debug
        self.patience = patience

        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        self.model_save_dir = "./model/"
        Path(self.model_save_dir).mkdir(parents=True, exist_ok=True)
        self.model_save_path = self.model_save_dir + f"{data_type}_{timestamp}_{wandb_id}.pt"

    def _update_save_best_score(self, f1_score: float, epoch: int):
        if self.best_f1_score < f1_score:
            self.best_f1_score = f1_score
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.model_save_path)
            logger.info("model is saved here: %s, best epoch: %s, best f1 score: %f" % (self.model_save_path, self.best_epoch, self.best_f1_score))

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
                logger.info('======== Epoch {:} / {:} ========'.format(epoch, self.epochs))
                logger.info("Training start...")
                self.model.train()
                loss_vals = []
                for step, batch in enumerate(tqdm(self.train_dataloader)):
                    device = self.device
                    if self.model_type == "box":
                        xy_rel_id = torch.stack(batch[12], dim=-1).to(device) # [batch_size, 2]
                        flag = batch[15]  # 0: HiEve, 1: MATRES
                        vol_A_B, vol_B_A, _, _, _, _ = self.model(batch, device, self.data_type) # [batch_size, # of datasets]
                        loss = self.bce_loss(vol_A_B, vol_B_A, xy_rel_id, flag)
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
                            if self.loss_type:
                                loss += self.lambda_dict["lambda_trans"] * self.loss_func_trans(alpha, beta, gamma).sum()
                        elif self.data_type == "matres":
                            loss = self.lambda_dict["lambda_anno"] * (self.loss_anno_dict["matres"](alpha, xy_rel_id) + self.loss_anno_dict["matres"](beta, yz_rel_id) + self.loss_anno_dict["matres"](gamma, xz_rel_id))
                            if self.loss_type:
                                loss += self.lambda_dict["lambda_trans"] * self.loss_func_trans(alpha, beta, gamma).sum()
                        elif self.data_type == "joint":
                            loss = self.lambda_dict["lambda_anno"] * self._get_anno_loss(batch_size, flag, alpha, beta, gamma, xy_rel_id, yz_rel_id, xz_rel_id)
                            if self.loss_type:
                                loss += self.lambda_dict["lambda_trans"] * self._get_trans_loss_h(alpha[:, 0:4], beta[:, 0:4], gamma[:, 0:4])
                                loss += self.lambda_dict["lambda_trans"] * self._get_trans_loss_t(alpha[:, 4:8], beta[:, 4:8], gamma[:, 4:8])
                                if self.loss_type == 2:
                                    loss += (self.lambda_dict["lambda_cross"] * self.loss_func_cross(alpha, beta, gamma)).sum()

                    loss_vals.append(loss.item())
                    loss.backward()
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
            valid_metrics.update(self.evaluator.evaluate("hieve", "valid"))
            cv_valid_metrics.update(self.evaluator.evaluate("hieve", "cv-valid"))

            if not self.debug:
                test_metrics.update(self.evaluator.evaluate("hieve", "test"))
                cv_test_metrics.update(self.evaluator.evaluate("hieve", "cv-test"))

        if self.data_type == "matres" or self.data_type == "joint":
            valid_metrics.update(self.evaluator.evaluate("matres", "valid"))
            cv_valid_metrics.update(self.evaluator.evaluate("matres", "cv-valid"))

            if not self.debug:
                test_metrics.update(self.evaluator.evaluate("matres", "test"))
                cv_test_metrics.update(self.evaluator.evaluate("matres", "cv-test"))

        logger.info("valid_metrics: {0}".format(valid_metrics))
        logger.info("cv_valid_metrics: {0}".format(cv_valid_metrics))
        wandb.log(valid_metrics, commit=False)

        if not self.debug:
            logger.info("test_metrics: {0}".format(test_metrics))
            logger.info("cv_test_metrics: {0}".format(cv_test_metrics))
            wandb.log(test_metrics, commit=False)

        if self.data_type != "joint":   # single task
            f1_score = valid_metrics[f"[valid-{self.data_type}] F1 Score"]
        else:                           # joint task
            f1_score = valid_metrics[f"[valid-hieve] F1 Score"] + valid_metrics[f"[valid-matres] F1 Score"]

        self._update_save_best_score(f1_score, epoch)
        wandb.log({f"[{self.data_type}] Best F1 Score": self.best_f1_score}, commit=False)


class OneThresholdEvaluator:
    def __init__(self, train_type: str, model_type: str, model: Module, device: torch.device,
                 valid_dataloader_dict: Dict[str, DataLoader], test_dataloader_dict: Dict[str, DataLoader],
                 valid_cv_dataloader_dict: Dict[str, DataLoader], test_cv_dataloader_dict: Dict[str, DataLoader],
                 hieve_threshold: float, matres_threshold: float):
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
        self.hieve_threshold = hieve_threshold
        self.matres_threshold = matres_threshold

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
        eval_start_time = time.time()
        logger.info(f"Validation-[{eval_type}-{data_type}] start... ")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                device = self.device

                xy_rel_id = torch.stack(batch[12], dim=-1).to(device) # [batch_size, 2]
                yz_rel_id = torch.stack(batch[13], dim=-1).to(device)
                xz_rel_id = torch.stack(batch[14], dim=-1).to(device)
                flag = batch[15]  # 0: HiEve, 1: MATRES
                vol_A_B, vol_B_A, vol_B_C, vol_C_B, vol_A_C, vol_C_A = self.model(batch, device, self.train_type) # [batch_size, 2]

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
                    vol_A_B, vol_B_A = vol_A_B.squeeze(), vol_B_A.squeeze()  # [batch_size]
                    vol_B_C, vol_C_B = vol_B_C.squeeze(), vol_C_B.squeeze()
                    vol_A_C, vol_C_A = vol_A_C.squeeze(), vol_C_A.squeeze()

                if data_type == "hieve":
                    threshold = self.hieve_threshold
                if data_type == "matres":
                    threshold = self.matres_threshold

                xy_preds, xy_targets, xy_constraint_dict = threshold_evalution(vol_A_B, vol_B_A, xy_rel_id, threshold)
                yz_preds, yz_targets, yz_constraint_dict = threshold_evalution(vol_B_C, vol_C_B, yz_rel_id, threshold)
                xz_preds, xz_targets, xz_constraint_dict = threshold_evalution(vol_A_C, vol_C_A, xz_rel_id, threshold)
                pred_vals.extend(xy_preds)
                rel_ids.extend(xy_targets)

                if constraint_violation:
                    constraint_violation.update_violation_count_box(xy_constraint_dict, yz_constraint_dict, xz_constraint_dict)

            if constraint_violation:
                logger.info(f"[{eval_type}-{data_type}] constraint-violation: %s" % constraint_violation.violation_dict)
                logger.info(f"[{eval_type}-{data_type}] all_cases: %s" % constraint_violation.all_case_count)

        metrics = metric(data_type, eval_type, self.model_type, y_true=rel_ids, y_pred=pred_vals)
        logger.info("done!")
        metrics[f"[{eval_type}] Elapsed Time"] = (time.time() - eval_start_time)
        return metrics


class TwoThresholdEvaluator:
    def __init__(self, train_type: str, model_type: str, model: Module, device: torch.device,
                 valid_dataloader_dict: Dict[str, DataLoader], test_dataloader_dict: Dict[str, DataLoader],
                 valid_cv_dataloader_dict: Dict[str, DataLoader], test_cv_dataloader_dict: Dict[str, DataLoader],
                 hieve_threshold1: float, hieve_threshold2: float, matres_threshold1: float,  matres_threshold2: float):
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
        self.hieve_threshold1 = hieve_threshold1
        self.hieve_threshold2 = hieve_threshold2
        self.matres_threshold1 = matres_threshold1
        self.matres_threshold2 = matres_threshold2

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
        eval_start_time = time.time()
        logger.info(f"Validation-[{eval_type}-{data_type}] start... ")
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                device = self.device

                xy_rel_id = torch.stack(batch[12], dim=-1).to(device) # [batch_size, 2]
                yz_rel_id = torch.stack(batch[13], dim=-1).to(device)
                xz_rel_id = torch.stack(batch[14], dim=-1).to(device)
                flag = batch[15]  # 0: HiEve, 1: MATRES
                vol_A_B, vol_B_A, vol_B_C, vol_C_B, vol_A_C, vol_C_A = self.model(batch, device, self.train_type) # [batch_size, 2]

                if vol_A_B.shape[-1] == 2:
                    if data_type == "hieve":
                        vol_A_B, vol_B_A = vol_A_B[:, 0][flag == 0], vol_B_A[:, 0][flag == 0]  # [batch_size]
                        vol_B_C, vol_C_B = vol_B_C[:, 0][flag == 0], vol_C_B[:, 0][flag == 0]
                        vol_A_C, vol_C_A = vol_A_C[:, 0][flag == 0], vol_C_A[:, 0][flag == 0]
                        xy_rel_id = xy_rel_id[flag == 0]
                        yz_rel_id = yz_rel_id[flag == 0]
                        xz_rel_id = xz_rel_id[flag == 0]
                    elif data_type == "matres":
                        vol_A_B, vol_B_A = vol_A_B[:, 1][flag == 1], vol_B_A[:, 1][flag == 1]  # [batch_size]
                        vol_B_C, vol_C_B = vol_B_C[:, 1][flag == 1], vol_C_B[:, 1][flag == 1]
                        vol_A_C, vol_C_A = vol_A_C[:, 1][flag == 1], vol_C_A[:, 1][flag == 1]
                        xy_rel_id = xy_rel_id[flag == 1]
                        yz_rel_id = yz_rel_id[flag == 1]
                        xz_rel_id = xz_rel_id[flag == 1]
                else:
                    vol_A_B, vol_B_A = vol_A_B.squeeze(), vol_B_A.squeeze()  # [batch_size]
                    vol_B_C, vol_C_B = vol_B_C.squeeze(), vol_C_B.squeeze()
                    vol_A_C, vol_C_A = vol_A_C.squeeze(), vol_C_A.squeeze()

                if data_type == "hieve":
                    threshold1 = self.hieve_threshold1
                    threshold2 = self.hieve_threshold2
                if data_type == "matres":
                    threshold1 = self.matres_threshold1
                    threshold2 = self.matres_threshold2

                xy_preds, xy_targets, xy_constraint_dict = two_threshold_evalution(vol_A_B, vol_B_A, xy_rel_id, threshold1, threshold2)
                yz_preds, yz_targets, yz_constraint_dict = two_threshold_evalution(vol_B_C, vol_C_B, yz_rel_id, threshold1, threshold2)
                xz_preds, xz_targets, xz_constraint_dict = two_threshold_evalution(vol_A_C, vol_C_A, xz_rel_id, threshold1, threshold2)
                pred_vals.extend(xy_preds)
                rel_ids.extend(xy_targets)
                if constraint_violation:
                    constraint_violation.update_violation_count_box(xy_constraint_dict, yz_constraint_dict, xz_constraint_dict)

            if constraint_violation:
                logger.info(f"[{eval_type}-{data_type}] constraint-violation: %s" % constraint_violation.violation_dict)
                logger.info(f"[{eval_type}-{data_type}] all_cases: %s" % constraint_violation.all_case_count)

        metrics = metric(data_type, eval_type, self.model_type, y_true=rel_ids, y_pred=pred_vals)
        logger.info("done!")
        metrics[f"[{eval_type}] Elapsed Time"] = (time.time() - eval_start_time)
        return metrics


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
        eval_start_time = time.time()
        logger.info(f"Validation-[{eval_type}-{data_type}] start... ")
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
                if constraint_violation:
                    constraint_violation.update_violation_count_vector(alpha_indices, beta_indices, gamma_indices)
            if constraint_violation:
                logger.info(f"[{eval_type}-{data_type}] constraint-violation: %s" % constraint_violation.violation_dict)
                logger.info(f"[{eval_type}-{data_type}] all_cases: %s" % constraint_violation.all_case_count)

        metrics = metric(data_type, eval_type, self.model_type, y_true=rel_ids, y_pred=pred_vals)
        logger.info("done!")
        metrics[f"[{eval_type}] Elapsed Time"] = (time.time() - eval_start_time)
        return metrics
