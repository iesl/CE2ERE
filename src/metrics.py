from sklearn.metrics import confusion_matrix, classification_report


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
    if data_type == "matres":
        if len(y_true) == 0 or len(y_pred) == 0:
            metrics[f"[{eval_type}-MATRES] Precision"] = 0
            metrics[f"[{eval_type}-MATRES] Recall"] = 0
            metrics[f"[{eval_type}-MATRES] F1 Score"] = 0
            return metrics, None

        CM = confusion_matrix(y_true, y_pred)
        if model_type == "box" or model_type == "vector":
            Acc, P, R, F1, _ = CM_metric_box(CM)
        else:
            Acc, P, R, F1, _ = CM_metric(CM)
        metrics[f"[{eval_type}-MATRES] Precision"] = P
        metrics[f"[{eval_type}-MATRES] Recall"] = R
        metrics[f"[{eval_type}-MATRES] F1 Score"] = F1
        return metrics, CM

    if data_type == "hieve":
        if len(y_true) == 0 or len(y_pred) == 0:
            metrics[f"[{eval_type}-HiEve] Precision"] = 0
            metrics[f"[{eval_type}-HiEve] Recall"] = 0
            metrics[f"[{eval_type}-HiEve] F1 Score"] = 0
            return metrics, None

        result_dict = classification_report(y_true, y_pred, output_dict=True)
        result_table = classification_report(y_true, y_pred)

        if model_type == "box" or model_type == "vector":
            if "10" not in result_dict.keys():
                metrics[f"[{eval_type}-HiEve] Precision"] = 0
                metrics[f"[{eval_type}-HiEve] Recall"] = 0
                metrics[f"[{eval_type}-HiEve] F1 Score"] = 0
            else:
                P_PC = result_dict['10']['precision']   # Parent-Child - precision
                P_CP = result_dict['01']['precision']   # Child-Parent - precision
                P_CR = result_dict['11']['precision']   # CoRef - precision
                R_PC = result_dict['10']['recall']      # Parent-Child - recall
                R_CP = result_dict['01']['recall']      # Parent-Child - recall
                R_CR = result_dict['11']['recall']      # CoRef - recall
                F1_PC = result_dict['10']['f1-score']   # Parent-Child - f1 score
                F1_CP = result_dict['01']['f1-score']   # Child-Parent - f1 score
                F1_CR = result_dict['11']['f1-score']   # CoRef - f1 score
                metrics[f"[{eval_type}-HiEve] Precision"] = (P_PC+P_CP)/2
                metrics[f"[{eval_type}-HiEve] Recall"] = (R_PC+R_CP)/2
                metrics[f"[{eval_type}-HiEve] F1 Score"] = (F1_PC+F1_CP)/2
        else:
            P_PC = result_dict['0']['precision'] # Parent-Child - precision
            P_CP = result_dict['1']['precision']  # Child-Parent - precision

            R_PC = result_dict['0']['recall'] # Parent-Child - recall
            R_CP = result_dict['1']['recall'] # Child-Parent - recall

            F1_PC = result_dict['0']['f1-score'] # Parent-Child
            F1_CP = result_dict['1']['f1-score'] # Child-Parent
            F1_CR = result_dict['2']['f1-score'] # CoRef
            F1_NR = result_dict['3']['f1-score'] # NoRel
            F1_PC_CP_avg = (F1_PC + F1_CP) / 2.0

            metrics[f"[{eval_type}-HiEve] Precision"] = (P_PC + P_CP) / 2
            metrics[f"[{eval_type}-HiEve] Recall"] = (R_PC + R_CP) / 2
            metrics[f"[{eval_type}-HiEve] F1-PC"] = F1_PC
            metrics[f"[{eval_type}-HiEve] F1-CP"] = F1_CP
            metrics[f"[{eval_type}-HiEve] F1 Score"] = F1_PC_CP_avg
        return metrics, result_table

    return None, None


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
        if model_type == "box" or  model_type == "vector":
            self.violation_dict = {
                ("10", "10", "01"): [0, 0], ("10", "10", "11"): [0, 0], ("10", "10", "00"): [0, 0],
                ("10", "01"): [0, 0],
                ("10", "11", "01"): [0, 0], ("10", "11", "11"): [0, 0], ("10", "11", "00"): [0, 0],
                ("10", "00", "01"): [0, 0], ("10", "00", "11"): [0, 0],
                ("01", "10"): [0, 0],
                ("01", "01", "10"): [0, 0], ("01", "01", "11"): [0, 0], ("01", "01", "00"): [0, 0],
                ("01", "11", "10"): [0, 0], ("01", "11", "11"): [0, 0], ("01", "11", "00"): [0, 0],
                ("01", "00", "10"): [0, 0], ("01", "00", "11"): [0, 0],
                ("11", "10", "01"): [0, 0], ("11", "10", "11"): [0, 0], ("11", "10", "00"): [0, 0],
                ("11", "01", "10"): [0, 0], ("11", "01", "11"): [0, 0], ("11", "01", "00"): [0, 0],
                ("11", "11", "10"): [0, 0], ("11", "11", "01"): [0, 0], ("11", "11", "00"): [0, 0],
                ("11", "00", "10"): [0, 0], ("11", "00", "01"): [0, 0], ("11", "00", "11"): [0, 0],
                ("00", "10", "01"): [0, 0], ("00", "10", "11"): [0, 0],
                ("00", "01", "10"): [0, 0], ("00", "01", "11"): [0, 0],
                ("00", "11", "10"): [0, 0], ("00", "11", "01"): [0, 0], ("00", "11", "11"): [0, 0],
                ("00", "00"): [0, 0],
            }
            self.all_case_count = None
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
            if len(key) == 2: # (10, 01), (01, 10), (00, 00) cases
                xy, yz = key
                xy_indices = xy_constraint_dict[xy]
                yz_indices = yz_constraint_dict[yz]
                all_cases = len(xy_indices & yz_indices)
                value[0] += all_cases
            else:
                xy, yz, xz = key
                xy_indices = xy_constraint_dict[xy]
                yz_indices = yz_constraint_dict[yz]
                xz_indices = xz_constraint_dict[xz]
                all_cases = len(xy_indices & yz_indices)
                value[0] += all_cases
                value[1] += len(xy_indices & yz_indices & xz_indices)

    def update_violation_count_vector(self, alpha_indices, beta_indices, gamma_indices):
        # update each violation dict key using xy, yz, xz constraint dict
        assert len(alpha_indices) == len(beta_indices) == len(gamma_indices)
        for i in range(len(alpha_indices)):
            xy = alpha_indices[i]
            yz = alpha_indices[i]
            xz = gamma_indices[i]

            key = (xy, yz, xz)
            if key in self.violation_dict.keys():
                self.violation_dict[(xy, yz, xz)] += 1
            self.all_case_count[(xy, yz)] += 1
