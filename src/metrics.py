from sklearn.metrics import confusion_matrix, classification_report


def metric(data_type, eval_type, y_true, y_pred):
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
        CM = confusion_matrix(y_true, y_pred)
        Acc, P, R, F1, _ = CM_metric(CM)

        metrics[f"[{eval_type}-MATRES] Precision"] = P
        metrics[f"[{eval_type}-MATRES] Recall"] = R
        metrics[f"[{eval_type}-MATRES] F1 Score"] = F1
        return metrics, CM

    if data_type == "hieve":
        result_dict = classification_report(y_true, y_pred, output_dict=True)
        result_table = classification_report(y_true, y_pred)

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
        metrics[f"[{eval_type}-HiEve] F1-CP"] = F1_CP
        metrics[f"[{eval_type}-HiEve] F1-PC"] = F1_PC
        metrics[f"[{eval_type}-HiEve] F1-CP"] = F1_CP
        metrics[f"[{eval_type}-HiEve] F1-PC-CP-AVG"] = F1_PC_CP_avg
        return metrics, result_table

    return None, None


def CM_metric(CM):
    all_ = CM.sum()
    Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    P = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0][0:3].sum() + CM[1][0:3].sum() + CM[2][0:3].sum() + CM[3][0:3].sum())
    R = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0].sum() + CM[1].sum() + CM[2].sum())
    F1 = 2 * P * R / (P + R)

    return Acc, P, R, F1, CM
