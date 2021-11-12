import logging
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger()

def metric(data_type, eval_type, model_type, y_true, y_pred):
    """
    confusion matrix ex)
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]
    confusion_matrix(y_true, y_pred):
    array([ [2, 0, 0],
            [0, 0, 1],
            [1, 0, 2]])
    => (i, j) indicates the number of samples with true label being i-th class and predicted label being j-th class.
    """
    metrics = {}
    if model_type == "box":
        CM = confusion_matrix(y_true, y_pred, labels=["00", "01", "10"])
    else:
        CM = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    logger.info("confusion_matrix: \n{0}".format(CM))
    if model_type == "box" or model_type == "vector":
        Acc, P, R, F1, _ = CM_metric_box_3class(CM)
    else:
        Acc, P, R, F1, _ = CM_metric_3class(CM)
    metrics[f"[{eval_type}-{data_type}] Precision"] = P
    metrics[f"[{eval_type}-{data_type}] Recall"] = R
    metrics[f"[{eval_type}-{data_type}] F1 Score"] = F1

    if data_type == "hieve" or data_type == "esl":
        if model_type == "box" or model_type == "vector":
            P, R, F1 = CM_metric_box_2class(CM)
        else:
            P, R, F1 = CM_metric_2class(CM)
        metrics[f"[{eval_type}-{data_type}] Precision (PC-CP)"] = P
        metrics[f"[{eval_type}-{data_type}] Recall (PC-CP)"] = R
        metrics[f"[{eval_type}-{data_type}] F1 Score (PC-CP)"] = F1

    logger.info("classifiction_report: \n{0}".format(classification_report(y_true, y_pred)))
    return metrics


def get_micro_metric(metrics: list, supports: list) -> float:
    total_true_positive = np.sum(np.array(metrics) * np.array(supports))
    total_support = np.sum(supports)

    return total_true_positive / total_support


def get_macro_metric(metrics: list) -> float:
    return np.mean(metrics)


def get_f1_score(total_precision: float, total_recall: float) -> float:
    return 2 * (total_precision * total_recall) / (total_precision + total_recall)


def CM_metric_3class(CM):
    all_ = CM.sum()
    rows = CM.shape[0]
    Acc = 0
    P_numerator, P_denominator = 0, 0
    R_numerator, R_denominator = 0, 0
    for i in range(rows):
        Acc += CM[i][i]
        P_denominator += CM[i][:rows-1].sum()
        if i == rows-1: continue  # NoRel, Vague case
        P_numerator += CM[i][i]
        R_numerator += CM[i][i]
        R_denominator += CM[i].sum()

    Acc /= all_
    P = P_numerator / P_denominator
    R = R_numerator / R_denominator
    F1 = 2 * P * R / (P + R)

    # all_ = CM.sum()
    # Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    # P = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0][0:3].sum() + CM[1][0:3].sum() + CM[2][0:3].sum() + CM[3][0:3].sum())
    # R = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0].sum() + CM[1].sum() + CM[2].sum())
    # F1 = 2 * P * R / (P + R)

    return Acc, P, R, F1, CM


def CM_metric_2class(CM):
    P = 1.0 * (CM[0][0] + CM[1][1]) / (CM[0][0:2].sum() + CM[1][0:2].sum() + CM[2][0:2].sum() + CM[3][0:2].sum())
    R = 1.0 * (CM[0][0] + CM[1][1]) / (CM[0].sum() + CM[1].sum())
    F1 = 2 * P * R / (P + R)

    return P, R, F1

def CM_metric_box_3class(CM):
    """
    keys:
    00, 01, 10, 11
    """
    all_ = CM.sum()
    rows = CM.shape[0]
    Acc = 0
    P_numerator, P_denominator = 0, 0
    R_numerator, R_denominator = 0, 0
    for i in range(rows):
        Acc += CM[i][i]
        P_denominator += CM[i][1:rows].sum()
        if i==0: continue # NoRel, Vague case
        P_numerator += CM[i][i]
        R_numerator += CM[i][i]
        R_denominator += CM[i].sum()

    Acc /= all_
    P = P_numerator/P_denominator
    R = R_numerator/R_denominator
    F1 = 2 * P * R / (P + R)

    # Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    # P = 1.0 * (CM[1][1] + CM[2][2] + CM[3][3]) / (CM[0][1:4].sum() + CM[1][1:4].sum() + CM[2][1:4].sum() + CM[3][1:4].sum())
    # R = 1.0 * (CM[1][1] + CM[2][2] + CM[3][3]) / (CM[1].sum() + CM[2].sum() + CM[3].sum())

    return Acc, P, R, F1, CM


def CM_metric_box_2class(CM):
    """
    keys:
    00, 01, 10, 11
    """
    rows = CM.shape[0] # 4
    P_numerator, P_denominator = 0, 0
    R_numerator, R_denominator = 0, 0
    for i in range(rows):
        P_denominator += CM[i][1:rows-1].sum()
        if i==0 or i==3: continue # NoRel, Vague or Coref case
        P_numerator += CM[i][i]
        R_numerator += CM[i][i]
        R_denominator += CM[i].sum()

    P = P_numerator/P_denominator
    R = R_numerator/R_denominator
    F1 = 2 * P * R / (P + R)

    # Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    # P = 1.0 * (CM[1][1] + CM[2][2]) / (CM[0][1:3].sum() + CM[1][1:3].sum() + CM[2][1:3].sum() + CM[3][1:3].sum())
    # R = 1.0 * (CM[1][1] + CM[2][2]) / (CM[1].sum() + CM[2].sum())

    return P, R, F1


class ConstraintViolation:
    """
    constraint-violation
    0 = 10, 1 = 01, 2 = 11, 3 = 00
    [vector model]          [box model]
    0, 0 - [1,2,3]          10, 10 - [01,11,00]
    0, 1 - [none]           10, 01 - [none]
    0, 2 - [1,2,3]          10, 11 - [01,11,00]
    0, 3 - [1,2]            10, 00 - [01,11]
    1, 0 - [none]           01, 10 - [none]
    1, 1 - [0,2,3]          01, 01 - [10,11,00]
    1, 2 - [0,2,3]          01, 11 - [10,11,00]
    1, 3 - [0,2]            01, 00 - [10,11]
    2, 0 - [1,2,3]          11, 10 - [01,11,00]
    2, 1 - [0,2,3]          11, 01 - [10,11,00]
    2, 2 - [0,1,3]          11, 11 - [10,01,00]
    2, 3 - [0,1,2]          11, 00 - [10,01,11]
    3, 0 - [1,2]            00, 10 - [01,11]
    3, 1 - [0,2]            00, 01 - [10,11]
    3, 2 - [0,1,2]          00, 11 - [10,01,11]
    3, 3 - [none]           00, 00 - [none]

    constraint_violdation:
    [box]    violation_dict {key=violation case, value=[all_count for first two, count per each case]}
    [vector] violation_dict {key=violation case, value=count per each case}
             all_case_count {key=possible case, }
    """
    def __init__(self, model_type):
        super().__init__()
        if model_type == "box" or model_type == "vector":
            self.violation_dict = {
                ("10", "10", "01"): 0, ("10", "10", "11"): 0, ("10", "10", "00"): 0,
                ("10", "11", "01"): 0, ("10", "11", "11"): 0, ("10", "11", "00"): 0,
                ("10", "00", "01"): 0, ("10", "00", "11"): 0,
                ("01", "01", "10"): 0, ("01", "01", "11"): 0, ("01", "01", "00"): 0,
                ("01", "11", "10"): 0, ("01", "11", "11"): 0, ("01", "11", "00"): 0,
                ("01", "00", "10"): 0, ("01", "00", "11"): 0,
                ("11", "10", "01"): 0, ("11", "10", "11"): 0, ("11", "10", "00"): 0,
                ("11", "01", "10"): 0, ("11", "01", "11"): 0, ("11", "01", "00"): 0,
                ("11", "11", "10"): 0, ("11", "11", "01"): 0, ("11", "11", "00"): 0,
                ("11", "00", "10"): 0, ("11", "00", "01"): 0, ("11", "00", "11"): 0,
                ("00", "10", "01"): 0, ("00", "10", "11"): 0,
                ("00", "01", "10"): 0, ("00", "01", "11"): 0,
                ("00", "11", "10"): 0, ("00", "11", "01"): 0, ("00", "11", "11"): 0,
            }
            self.all_case_count = {
                ("10", "10"): 0, ("10", "01"): 0, ("10", "11"): 0, ("10", "00"): 0,
                ("01", "10"): 0, ("01", "01"): 0, ("01", "11"): 0, ("01", "00"): 0,
                ("11", "10"): 0, ("11", "01"): 0, ("11", "11"): 0, ("11", "00"): 0,
                ("00", "10"): 0, ("00", "01"): 0, ("00", "11"): 0, ("00", "00"): 0,
            }
        else:
            self.violation_dict = {
                (0, 0, 1): 0, (0, 0, 2): 0, (0, 0, 3): 0,
                (0, 2, 1): 0, (0, 2, 2): 0, (0, 2, 3): 0,
                (0, 3, 1): 0, (0, 3, 2): 0,
                (1, 1, 0): 0, (1, 1, 2): 0, (1, 1, 3): 0,
                (1, 2, 0): 0, (1, 2, 2): 0, (1, 2, 3): 0,
                (1, 3, 0): 0, (1, 3, 2): 0,
                (2, 0, 1): 0, (2, 0, 2): 0, (2, 0, 3): 0,
                (2, 1, 0): 0, (2, 1, 2): 0, (2, 1, 3): 0,
                (2, 2, 0): 0, (2, 2, 1): 0, (2, 2, 3): 0,
                (2, 3, 0): 0, (2, 3, 1): 0, (2, 3, 2): 0,
                (3, 0, 1): 0, (3, 0, 2): 0,
                (3, 1, 0): 0, (3, 1, 2): 0,
                (3, 2, 0): 0, (3, 2, 1): 0, (3, 2, 2): 0,
            }
            self.all_case_count = {
                (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0,
                (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0,
                (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0,
                (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0
            }

    def update_violation_count_box(self, xy_constraint_dict, yz_constraint_dict, xz_constraint_dict):
        # update each violation dict key using xy, yz, xz constraint dict
        for key, value in self.violation_dict.items():
            xy, yz, xz = key
            xy_indices = xy_constraint_dict[xy]
            yz_indices = yz_constraint_dict[yz]
            xz_indices = xz_constraint_dict[xz]
            self.violation_dict[key] += len(xy_indices & yz_indices & xz_indices)

        for key, value in self.all_case_count.items():
            xy, yz = key
            xy_indices = xy_constraint_dict[xy]
            yz_indices = yz_constraint_dict[yz]
            self.all_case_count[key] += len(xy_indices & yz_indices)

    def update_violation_count_vector(self, alpha_indices, beta_indices, gamma_indices):
        # update each violation dict key using xy, yz, xz constraint dict
        assert len(alpha_indices) == len(beta_indices) == len(gamma_indices)
        for i in range(len(alpha_indices)):
            xy = alpha_indices[i]
            yz = beta_indices[i]
            xz = gamma_indices[i]
            violation_key = (xy, yz, xz)
            if violation_key in self.violation_dict.keys():
                self.violation_dict[violation_key] += 1

            total_key = (xy, yz)
            self.all_case_count[total_key] += 1


class CrossCategoryConstraintViolation:
    """
    constraint-violation
    0 = 10, 1 = 01, 2 = 11, 3 = 00
    [box model]
    h10, h10 - [h01,h11,h00,m01]
    h10, h01 - [none]
    h10, h11 - [h01,h11,h00,m01]
    h10, h00 - [h01,h11]

    h10, m10 - [m01,m11,m00,h01,h11]
    h10, m01 - [none]
    h10, m11 - [m01,m11,m00,h01,h11]
    h10, m00 - [none]

    h01, h10 - [none]
    h01, h01 - [h10,h11,h00,m10]
    h01, h11 - [h10,h11,h00,m10]
    h01, h00 - [h10,h11]

    h01, m10 - [none]
    h01, m01 - [m10,m11,m00,h10,h11]
    h01, m11 - [m10,m11,m00,h10,h11]
    h01, m00 - [none]

    h11, h10 - [h01,h11,h00,m01]
    h11, h01 - [h10,h11,h00,m10]
    h11, h11 - [h10,h01,h00,m10,m01,m00]
    h11, h00 - [h10,h01,h11]

    h11, m10 - [m01,m11,m00,h01,h11]
    h11, m01 - [m10,m11,m00,h10,h11]
    h11, m11 - [m10,m01,m00]
    h11, m00 - [m10,m01,m11]

    h00, h10 - [h01,h11]
    h00, h01 - [h10,h11]
    h00, h11 - [h10,h01,h11]
    h00, h00 - [none]

    h00, m10 - [none]
    h00, m01 - [none]
    h00, m11 - [none]
    h00, m00 - [none]

    m10, h10 - [m01,m11,m00,h01,h11]
    m10, h01 - [none]
    m10, h11 - [m01,m11,m00,h01,h11]
    m10, h00 - [none]

    m10, m10 - [m01,m11,m00,h01,h11]
    m10, m01 - [none]
    m10, m11 - [m01,m11,m00,h01,h11]
    m10, m00 - [m01,m11]

    m01, h10 - [none]
    m01, h01 - [m10,m11,m00,h10,h11]
    m01, h11 - [m10,m11,m00,h10,h11]
    m01, h00 - [none]

    m01, m10 - [none]
    m01, m01 - [m10,m11,m00,h10,h11]
    m01, m11 - [m10,m11,m00,h10,h11]
    m01, m00 - [m10,m11]

    m11, h10 - [m01]
    m11, h01 - [m10]
    m11, h11 - [m10,m01,m00]
    m11, h00 - [none]

    m11, m10 - [m01,m11,m00,h01,h11]
    m11, m01 - [m10,m11,m00,h10,h11]
    m11, m11 - [m10,m01,m00]
    m11, m00 - [m10,m01,m11,h11]


    m00, h10 - [none]
    m00, h01 - [none]
    m00, h11 - [m10,m01,m11,h11]
    m00, h00 - [none]

    m00, m10 - [m01,m11]
    m00, m01 - [m10,m11]
    m00, m11 - [m10,m01,m11]
    m00, m00 - [none]
    """
    def __init__(self, model_type):
        super().__init__()
        if model_type == "box" or model_type == "vector":
            self.violation_dict = {
                ("h10", "h10", "h01"): 0, ("h10", "h10", "h11"): 0, ("h10", "h10", "h00"): 0, ("h10", "h10", "m01"): 0,
                ("h10", "h11", "h01"): 0, ("h10", "h11", "h11"): 0, ("h10", "h11", "h00"): 0, ("h10", "h11", "m01"): 0,
                ("h10", "h00", "h01"): 0, ("h10", "h00", "h11"): 0,

                ("h10", "m10", "m01"): 0, ("h10", "m10", "m11"): 0, ("h10", "m10", "m00"): 0, ("h10", "m10", "h01"): 0, ("h10", "m10", "h11"): 0,
                ("h10", "m11", "m01"): 0, ("h10", "m11", "m11"): 0, ("h10", "m11", "m00"): 0, ("h10", "m11", "h01"): 0, ("h10", "m11", "h11"): 0,

                ("h01", "h01", "h10"): 0, ("h01", "h01", "h11"): 0, ("h01", "h01", "h00"): 0, ("h01", "h01", "m10"): 0,
                ("h01", "h11", "h10"): 0, ("h01", "h11", "h11"): 0, ("h01", "h11", "h00"): 0, ("h01", "h11", "m10"): 0,
                ("h01", "h00", "h10"): 0, ("h01", "h00", "h11"): 0,

                ("h01", "m01", "m10"): 0, ("h01", "m01", "m11"): 0, ("h01", "m01", "m00"): 0, ("h01", "m01", "h10"): 0, ("h01", "m01", "h11"): 0,
                ("h01", "m11", "m10"): 0, ("h01", "m11", "m11"): 0, ("h01", "m11", "m00"): 0, ("h01", "m11", "h10"): 0, ("h01", "m11", "h11"): 0,

                ("h11", "h10", "h01"): 0, ("h11", "h10", "h11"): 0, ("h11", "h10", "h00"): 0, ("h11", "h10", "m01"): 0,
                ("h11", "h01", "h10"): 0, ("h11", "h01", "h11"): 0, ("h11", "h01", "h00"): 0, ("h11", "h01", "m10"): 0,
                ("h11", "h11", "h10"): 0, ("h11", "h11", "h01"): 0, ("h11", "h11", "h00"): 0,
                ("h11", "h11", "m10"): 0, ("h11", "h11", "m01"): 0, ("h11", "h11", "m00"): 0,
                ("h11", "h00", "h10"): 0, ("h11", "h00", "h01"): 0, ("h11", "h00", "h11"): 0,

                ("h11", "m10", "m01"): 0, ("h11", "m10", "m11"): 0, ("h11", "m10", "m00"): 0, ("h11", "m10", "h01"): 0, ("h11", "m10", "h11"): 0,
                ("h11", "m01", "m10"): 0, ("h11", "m01", "m11"): 0, ("h11", "m01", "m00"): 0, ("h11", "m01", "h10"): 0, ("h11", "m01", "h11"): 0,
                ("h11", "m11", "m10"): 0, ("h11", "m11", "m01"): 0, ("h11", "m11", "m00"): 0,
                ("h11", "m00", "m10"): 0, ("h11", "m00", "m01"): 0, ("h11", "m00", "m11"): 0,

                ("h00", "h10", "h01"): 0, ("h00", "h10", "h11"): 0,
                ("h00", "h01", "h10"): 0, ("h00", "h01", "h11"): 0,
                ("h00", "h11", "h10"): 0, ("h00", "h11", "h01"): 0, ("h00", "h11", "h11"): 0,

                ("m10", "h10", "m01"): 0, ("m10", "h10", "m11"): 0, ("m10", "h10", "m00"): 0, ("m10", "h10", "h01"): 0, ("m10", "h10", "h11"): 0,
                ("m10", "h11", "m01"): 0, ("m10", "h11", "m11"): 0, ("m10", "h11", "m00"): 0, ("m10", "h11", "h01"): 0, ("m10", "h11", "h11"): 0,

                ("m10", "m10", "m01"): 0, ("m10", "m10", "m11"): 0, ("m10", "m10", "m00"): 0, ("m10", "m10", "h01"): 0, ("m10", "m10", "h11"): 0,
                ("m10", "m11", "m01"): 0, ("m10", "m11", "m11"): 0, ("m10", "m11", "m00"): 0, ("m10", "m11", "h01"): 0, ("m10", "m11", "h11"): 0,
                ("m10", "m00", "m01"): 0, ("m10", "m00", "m11"): 0,

                ("m01", "h01", "m10"): 0, ("m01", "h01", "m11"): 0, ("m01", "h01", "m00"): 0, ("m01", "h01", "h10"): 0, ("m01", "h01", "h11"): 0,
                ("m01", "h11", "m10"): 0, ("m01", "h11", "m11"): 0, ("m01", "h11", "m00"): 0, ("m01", "h11", "h10"): 0, ("m01", "h11", "h11"): 0,

                ("m01", "m01", "m10"): 0, ("m01", "m01", "m11"): 0, ("m01", "m01", "m00"): 0, ("m01", "m01", "h10"): 0, ("m01", "m01", "h11"): 0,
                ("m01", "m11", "m10"): 0, ("m01", "m11", "m11"): 0, ("m01", "m11", "m00"): 0, ("m01", "m11", "h10"): 0, ("m01", "m11", "h11"): 0,
                ("m01", "m00", "m10"): 0, ("m01", "m00", "m11"): 0,

                ("m11", "h10", "m01"): 0,
                ("m11", "h01", "m10"): 0,
                ("m11", "h11", "m10"): 0, ("m11", "h11", "m01"): 0, ("m11", "h11", "m00"): 0,

                ("m11", "m10", "m01"): 0, ("m11", "m10", "m11"): 0, ("m11", "m10", "m00"): 0, ("m11", "m10", "h01"): 0, ("m11", "m10", "h11"): 0,
                ("m11", "m01", "m10"): 0, ("m11", "m01", "m11"): 0, ("m11", "m01", "m00"): 0, ("m11", "m01", "h10"): 0, ("m11", "m01", "h11"): 0,
                ("m11", "m11", "m10"): 0, ("m11", "m11", "m01"): 0, ("m11", "m11", "m00"): 0,
                ("m11", "m00", "m10"): 0, ("m11", "m00", "m01"): 0, ("m11", "m00", "m11"): 0, ("m11", "m00", "h11"): 0,

                ("m00", "h11", "m10"): 0, ("m00", "h11", "m01"): 0, ("m00", "h11", "m11"): 0, ("m00", "h11", "h11"): 0,

                ("m00", "m10", "m01"): 0, ("m00", "m10", "m11"): 0,
                ("m00", "m01", "m10"): 0, ("m00", "m01", "m11"): 0,
                ("m00", "m11", "m10"): 0, ("m00", "m11", "m01"): 0, ("m00", "m11", "m11"): 0,
            }
            self.all_case_count = {
                ("h10", "h10"): 0, ("h10", "h01"): 0, ("h10", "h11"): 0, ("h10", "h00"): 0,
                ("h01", "h10"): 0, ("h01", "h01"): 0, ("h01", "h11"): 0, ("h01", "h00"): 0,
                ("h11", "h10"): 0, ("h11", "h01"): 0, ("h11", "h11"): 0, ("h11", "h00"): 0,
                ("h00", "h10"): 0, ("h00", "h01"): 0, ("h00", "h11"): 0, ("h00", "h00"): 0,

                ("m10", "m10"): 0, ("m10", "m01"): 0, ("m10", "m11"): 0, ("m10", "m00"): 0,
                ("m01", "m10"): 0, ("m01", "m01"): 0, ("m01", "m11"): 0, ("m01", "m00"): 0,
                ("m11", "m10"): 0, ("m11", "m01"): 0, ("m11", "m11"): 0, ("m11", "m00"): 0,
                ("m00", "m10"): 0, ("m00", "m01"): 0, ("m00", "m11"): 0, ("m00", "m00"): 0,

                ("h10", "m10"): 0, ("h10", "m01"): 0, ("h10", "m11"): 0, ("h10", "m00"): 0,
                ("h01", "m10"): 0, ("h01", "m01"): 0, ("h01", "m11"): 0, ("h01", "m00"): 0,
                ("h11", "m10"): 0, ("h11", "m01"): 0, ("h11", "m11"): 0, ("h11", "m00"): 0,
                ("h00", "m10"): 0, ("h00", "m01"): 0, ("h00", "m11"): 0, ("h00", "m00"): 0,

                ("m10", "h10"): 0, ("m10", "h01"): 0, ("m10", "h11"): 0, ("m10", "h00"): 0,
                ("m01", "h10"): 0, ("m01", "h01"): 0, ("m01", "h11"): 0, ("m01", "h00"): 0,
                ("m11", "h10"): 0, ("m11", "h01"): 0, ("m11", "h11"): 0, ("m11", "h00"): 0,
                ("m00", "h10"): 0, ("m00", "h01"): 0, ("m00", "h11"): 0, ("m00", "h00"): 0,
            }
        else:
            self.violation_dict = {
                (0, 0, 1): 0, (0, 0, 2): 0, (0, 0, 3): 0, (0, 0, 5): 0,
                (0, 2, 1): 0, (0, 2, 2): 0, (0, 2, 3): 0, (0, 2, 5): 0,
                (0, 3, 1): 0, (0, 3, 2): 0,

                (0, 4, 5): 0, (0, 4, 6): 0, (0, 4, 7): 0, (0, 4, 1): 0, (0, 4, 2): 0,
                (0, 6, 5): 0, (0, 6, 6): 0, (0, 6, 7): 0, (0, 6, 1): 0, (0, 6, 2): 0,

                (1, 1, 0): 0, (1, 1, 2): 0, (1, 1, 3): 0, (1, 1, 4): 0,
                (1, 2, 0): 0, (1, 2, 2): 0, (1, 2, 3): 0, (1, 2, 4): 0,
                (1, 3, 0): 0, (1, 3, 2): 0,

                (1, 5, 4): 0, (1, 5, 6): 0, (1, 5, 7): 0, (1, 5, 0): 0, (1, 5, 2): 0,
                (1, 6, 4): 0, (1, 6, 6): 0, (1, 6, 7): 0, (1, 6, 0): 0, (1, 6, 2): 0,

                (2, 0, 1): 0, (2, 0, 2): 0, (2, 0, 3): 0, (2, 0, 5): 0,
                (2, 1, 0): 0, (2, 1, 2): 0, (2, 1, 3): 0, (2, 1, 4): 0,
                (2, 2, 0): 0, (2, 2, 1): 0, (2, 2, 3): 0,
                (2, 2, 4): 0, (2, 2, 5): 0, (2, 2, 7): 0,
                (2, 3, 0): 0, (2, 3, 1): 0, (2, 3, 2): 0,

                (2, 4, 5): 0, (2, 4, 6): 0, (2, 4, 7): 0, (2, 4, 1): 0, (2, 4, 2): 0,
                (2, 5, 4): 0, (2, 5, 6): 0, (2, 5, 7): 0, (2, 5, 0): 0, (2, 5, 2): 0,
                (2, 6, 4): 0, (2, 6, 5): 0, (2, 6, 7): 0,
                (2, 7, 4): 0, (2, 7, 5): 0, (2, 7, 6): 0,

                (3, 0, 1): 0, (3, 0, 2): 0,
                (3, 1, 0): 0, (3, 1, 2): 0,
                (3, 2, 0): 0, (3, 2, 1): 0, (3, 2, 2): 0,

                (4, 0, 5): 0, (4, 0, 6): 0, (4, 0, 7): 0, (4, 0, 1): 0, (4, 0, 2): 0,
                (4, 2, 5): 0, (4, 2, 6): 0, (4, 2, 7): 0, (4, 2, 1): 0, (4, 2, 2): 0,

                (4, 4, 5): 0, (4, 4, 6): 0, (4, 4, 7): 0, (4, 4, 1): 0, (4, 4, 2): 0,
                (4, 6, 5): 0, (4, 6, 6): 0, (4, 6, 7): 0, (4, 6, 1): 0, (4, 6, 2): 0,
                (4, 7, 5): 0, (4, 7, 6): 0,

                (5, 1, 4): 0, (5, 1, 6): 0, (5, 1, 7): 0, (5, 1, 0): 0, (5, 1, 2): 0,
                (5, 2, 4): 0, (5, 2, 6): 0, (5, 2, 7): 0, (5, 2, 0): 0,
                (5, 2, 2): 0,

                (5, 5, 4): 0, (5, 5, 6): 0, (5, 5, 7): 0, (5, 5, 0): 0,
                (5, 5, 2): 0,
                (5, 6, 4): 0, (5, 6, 6): 0, (5, 6, 7): 0, (5, 6, 0): 0,
                (5, 6, 2): 0,
                (5, 7, 4): 0, (5, 7, 6): 0,

                (6, 0, 5): 0,
                (6, 1, 4): 0,
                (6, 2, 4): 0, (6, 2, 5): 0, (6, 2, 7): 0,

                (6, 4, 5): 0, (6, 4, 6): 0, (6, 4, 7): 0, (6, 4, 1): 0,
                (6, 4, 2): 0,
                (6, 5, 4): 0, (6, 5, 6): 0, (6, 5, 7): 0, (6, 5, 0): 0,
                (6, 5, 2): 0,
                (6, 6, 4): 0, (6, 6, 5): 0, (6, 6, 7): 0,
                (6, 7, 4): 0, (6, 7, 5): 0, (6, 7, 6): 0, (6, 7, 2): 0,

                (7, 2, 4): 0, (7, 2, 5): 0, (7, 2, 6): 0, (7, 2, 2): 0,

                (7, 4, 5): 0, (7, 4, 6): 0,
                (7, 5, 4): 0, (7, 5, 6): 0,
                (7, 6, 4): 0, (7, 6, 5): 0, (7, 6, 6): 0,
            }
            self.all_case_count = {
                (0, 0): 0, (0, 1): 0, (0, 2): 0, (0, 3): 0,
                (1, 0): 0, (1, 1): 0, (1, 2): 0, (1, 3): 0,
                (2, 0): 0, (2, 1): 0, (2, 2): 0, (2, 3): 0,
                (3, 0): 0, (3, 1): 0, (3, 2): 0, (3, 3): 0,

                (4, 4): 0, (4, 5): 0, (4, 6): 0, (4, 7): 0,
                (5, 4): 0, (5, 5): 0, (5, 6): 0, (5, 7): 0,
                (6, 4): 0, (6, 5): 0, (6, 6): 0, (6, 7): 0,
                (7, 4): 0, (7, 5): 0, (7, 6): 0, (7, 7): 0,

                (0, 4): 0, (0, 5): 0, (0, 6): 0, (0, 7): 0,
                (1, 4): 0, (1, 5): 0, (1, 6): 0, (1, 7): 0,
                (2, 4): 0, (2, 5): 0, (2, 6): 0, (2, 7): 0,
                (3, 4): 0, (3, 5): 0, (3, 6): 0, (3, 7): 0,

                (4, 0): 0, (4, 1): 0, (4, 2): 0, (4, 3): 0,
                (5, 0): 0, (5, 1): 0, (5, 2): 0, (5, 3): 0,
                (6, 0): 0, (6, 1): 0, (6, 2): 0, (6, 3): 0,
                (7, 0): 0, (7, 1): 0, (7, 2): 0, (7, 3): 0,
            }

    def update_violation_count_box(self, h_xy_cv_list, h_yz_cv_list, h_xz_cv_list, m_xy_cv_list, m_yz_cv_list, m_xz_cv_list):
        # update each violation dict key using xy, yz, xz constraint dict
        for key, value in self.violation_dict.items():
            xy, yz, xz = key
            for h_xy_cv_dict, h_yz_cv_dict, h_xz_cv_dict, m_xy_cv_dict, m_yz_cv_dict, m_xz_cv_dict in zip(h_xy_cv_list, h_yz_cv_list, h_xz_cv_list,
                                                                                                          m_xy_cv_list, m_yz_cv_list, m_xz_cv_list):
                if xy.startswith("h"):
                    xy_indices = h_xy_cv_dict[xy[1:]]
                elif xy.startswith("m"):
                    xy_indices = m_xy_cv_dict[xy[1:]]
                if yz.startswith("h"):
                    yz_indices = h_yz_cv_dict[yz[1:]]
                elif yz.startswith("m"):
                    yz_indices = m_yz_cv_dict[yz[1:]]
                if xz.startswith("h"):
                    xz_indices = h_xz_cv_dict[xz[1:]]
                elif xz.startswith("m"):
                    xz_indices = m_xz_cv_dict[xz[1:]]
                self.violation_dict[key] += len(xy_indices & yz_indices & xz_indices)

        for key, value in self.all_case_count.items():
            xy, yz = key
            for h_xy_cv_dict, h_yz_cv_dict, m_xy_cv_dict, m_yz_cv_dict in zip(h_xy_cv_list, h_yz_cv_list, m_xy_cv_list, m_yz_cv_list):
                if xy.startswith("h"):
                    xy_indices = h_xy_cv_dict[xy[1:]]
                elif xy.startswith("m"):
                    xy_indices = m_xy_cv_dict[xy[1:]]
                if yz.startswith("h"):
                    yz_indices = h_yz_cv_dict[yz[1:]]
                elif yz.startswith("m"):
                    yz_indices = m_yz_cv_dict[yz[1:]]
                # multiply 2 because we have two cases for each (xy, yz) ie. hh -> hhh, hhm
                self.all_case_count[key] += len(xy_indices & yz_indices) * 2

    def update_violation_count_vector(self, h_alpha_list, h_beta_list, h_gamma_list, m_alpha_list, m_beta_list, m_gamma_list):
        # update each violation dict key using xy, yz, xz constraint dict
        for key, value in self.violation_dict.items():
            xy, yz, xz = key
            for i, (ha, hb, hg, ma, mb, mg) in enumerate(zip(h_alpha_list, h_beta_list, h_gamma_list, m_alpha_list, m_beta_list, m_gamma_list)):
                if xy < 4:
                    a = ha
                elif xy >= 4:
                    a = ma+4

                if yz < 4:
                    b = hb
                elif yz >= 4:
                    b = mb+4

                if xz < 4:
                    g = hg
                elif xz >= 4:
                    g = mg+4

                if a == xy and b == yz and g == xz:
                    self.violation_dict[key] += 1

        for key, value in self.all_case_count.items():
            xy, yz = key
            for i, (ha, hb, ma, mb) in enumerate(zip(h_alpha_list, h_beta_list, m_alpha_list, m_beta_list)):
                if xy < 4:
                    a = ha
                elif xy >= 4:
                    a = ma + 4

                if yz < 4:
                    b = hb
                elif yz >= 4:
                    b = mb + 4

                if a == xy and b == yz:
                    # +2 because we have two cases for each (xy, yz) ie. hh -> hhh, hhm
                    self.all_case_count[key] += 2

