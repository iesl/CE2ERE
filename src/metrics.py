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
    CM = confusion_matrix(y_true, y_pred)
    logger.info("confusion_matrix: \n{0}".format(CM))
    if model_type == "box" or model_type == "vector":
        Acc, P, R, F1, _ = CM_metric_box(CM)
    else:
        Acc, P, R, F1, _ = CM_metric(CM)
    metrics[f"[{eval_type}-{data_type}] Precision"] = P
    metrics[f"[{eval_type}-{data_type}] Recall"] = R
    metrics[f"[{eval_type}-{data_type}] F1 Score"] = F1

    result_dict = classification_report(y_true, y_pred, output_dict=True)
    logger.info("classifiction_report: \n{0}".format(classification_report(y_true, y_pred)))
    if data_type == "hieve":
        if model_type == "box" or model_type == "vector":
            pc_results = result_dict["10"]  # Parent-Child: precision, recall, f1, support
            cp_results = result_dict["01"]  # Child-Parent: ~
        else:
            pc_results = result_dict["0"]   # Parent-Child: precision,recall, f1, support
            cp_results = result_dict["1"]   # Child-Parent: ~

        pc_precision, pc_recall, pc_f1, pc_support = pc_results.values()
        cp_precision, cp_recall, cp_f1, cp_support = cp_results.values()
        f1_scores = [pc_f1, cp_f1]
        macro_f1_score = get_macro_metric(f1_scores)
        logger.info("macro f1 score: \n{0}".format(macro_f1_score))

    return metrics


def get_micro_metric(metrics: list, supports: list) -> float:
    total_true_positive = np.sum(np.array(metrics) * np.array(supports))
    total_support = np.sum(supports)

    return total_true_positive / total_support


def get_macro_metric(metrics: list) -> float:
    return np.mean(metrics)


def get_f1_score(total_precision: float, total_recall: float) -> float:
    return 2 * (total_precision * total_recall) / (total_precision + total_recall)


def generate_result_table(precisions: list, recalls: list, supports: list, f1s: list,
                          micro_precision: float, macro_precision: float,
                          micro_recall: float,    macro_recall: float,
                          micro_f1: float,        macro_f1: float) -> str:
    total_support = np.sum(supports[:3])
    line1 = "                 precision    recall  f1-score   support\n\n"
    line2 = "   Parent-Child       {p:.2f}      {r:.2f}      {f1:.2f}   {s}\n".format(
        p=precisions[0], r=recalls[0], f1=f1s[0], s=supports[0]
    )
    line3 = "   Child-Parent       {p:.2f}      {r:.2f}      {f1:.2f}   {s}\n".format(
        p=precisions[1], r=recalls[1], f1=f1s[1], s=supports[1]
    )
    line4 = "          CoRef       {p:.2f}      {r:.2f}      {f1:.2f}   {s}\n".format(
        p=precisions[2], r=recalls[2], f1=f1s[2], s=supports[2]
    )
    line5 = "NoRel [ignored]       {p:.2f}      {r:.2f}      {f1:.2f}   {s}\n\n".format(
        p=precisions[3], r=recalls[3], f1=f1s[3], s=supports[3]
    )
    line6 = "      micro avg       {p:.2f}      {r:.2f}      {f1:.2f}   {s}\n".format(
        p=micro_precision, r=micro_recall, f1=micro_f1, s=total_support
    )
    line7 = "      macro avg       {p:.2f}      {r:.2f}      {f1:.2f}   {s}".format(
        p=macro_precision, r=macro_recall, f1=macro_f1, s=total_support
    )

    return line1+line2+line3+line4+line5+line6+line7


def CM_metric(CM):
    all_ = CM.sum()
    Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    P = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0][0:3].sum() + CM[1][0:3].sum() + CM[2][0:3].sum() + CM[3][0:3].sum())
    R = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0].sum() + CM[1].sum() + CM[2].sum())
    F1 = 2 * P * R / (P + R)

    return Acc, P, R, F1, CM

def CM_metric_box(CM):
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