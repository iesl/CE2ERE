from metrics import metric
import torch


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


def volume_data():
    return {
        "p_A_B": torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        "p_B_A": torch.tensor([10, 12, 14, 16, 1, 2, 3, 4, 5, 8]),
        "p_B_C": torch.tensor([1, 2, 3, 5, 7, 9, 11, 13, 17, 19]),
        "p_C_B": torch.tensor([9, 11, 13, 15, 17, 1, 2, 3, 4, 5]),
        "p_A_C": torch.tensor([1, 2, 2, 2, 2, 2, 15, 16, 17, 18]),
        "p_C_A": torch.tensor([2, 2, 2, 2, 2, 12, 14, 16, 17, 18]),
        "threshold": 5
     }


def test_constraint_violation():
    volume_dict = volume_data()
    p_A_B = volume_dict["p_A_B"]
    p_B_A = volume_dict["p_B_A"]
    p_B_C = volume_dict["p_B_C"]
    p_C_B = volume_dict["p_C_B"]
    p_A_C = volume_dict["p_A_C"]
    p_C_A = volume_dict["p_C_A"]
    threshold = volume_dict["threshold"]

    # P(A|B) P(B|A) "PC" "10" case
    mask = (p_A_B >= threshold) & (p_B_A <= threshold)
    assert (mask.nonzero().squeeze() == torch.tensor([4, 5, 6, 7, 8])).all()
    xy_dict = {"10": set(mask.nonzero().squeeze().tolist())}

    # "PC" "10" case
    mask = (p_B_C >= threshold) & (p_C_B <= threshold)
    assert (mask.nonzero().squeeze() == torch.tensor([5, 6, 7, 8, 9])).all()
    yz_dict = {"10": set(mask.nonzero().squeeze().tolist())}

    # "CR" "11" case
    mask = (p_A_C >= threshold) & (p_C_A >= threshold)
    assert (mask.nonzero().squeeze() == torch.tensor([6, 7, 8, 9])).all()
    xz_dict = {"11": set(mask.nonzero().squeeze().tolist())}

    violation_dict = {("10", "10", "11"): [0, 0]}

    for key, value in violation_dict.items():
        xy, yz, xz = key
        xy_indices = xy_dict[xy]
        yz_indices = yz_dict[yz]
        xz_indices = xz_dict[xz]
        assert xy_indices & yz_indices == {5,6,7,8}
        assert xy_indices & yz_indices & xz_indices == {6, 7, 8}

        value[0] += len(xy_indices & yz_indices)
        value[1] += len(xy_indices & yz_indices & xz_indices)

    assert violation_dict[("10", "10", "11")][0] == 4
    assert violation_dict[("10", "10", "11")][1] == 3