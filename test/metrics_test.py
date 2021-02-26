from metrics import metric


def metric_data():
    return {"y_true": [1, 0, 1, 2, 3], "y_pred": [1, 0, 1, 2, 3]}


def test_matres_metric():
    y_true = metric_data()["y_true"]
    y_pred = metric_data()["y_pred"]
    metrics, CM = metric("matres", y_true, y_pred)
    print()
    print(metrics)
    print(CM)


def test_matres_metric():
    y_true = metric_data()["y_true"]
    y_pred = metric_data()["y_pred"]
    metrics, res_table = metric("hieve", y_true, y_pred)
    print()
    print(metrics)
    print(res_table)