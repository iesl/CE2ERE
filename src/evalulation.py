def threshold_evalution(volume1, volume2, relation_label, threshold):
    """
    1.  hieve:   volume1: P(A|B), volume2: P(B|A)
        matres:  volume1: P(B|A), volume2: P(A|B)
    2. relation_labels: xy_relation_id, yz_relation_id, xz_relation_id
    3. threshold: log0.5: -0.301029996, log0.25: -0.602059991, log0.1: -1
    3. preds: prediction labels
    4. targets: target labels
    """
    preds, targets = [], []
    constraint_dict = {}
    # HiEve - case1: P(A|B) > threshold1 && P(B|A) < threshold1 => A and B are PC, B and A are CP
    # MATRES - case1: P(B|A) > threshold1 && P(A|B) < threshold1 => A is before B, B is after A
    mask = (volume1 > threshold) & (volume2 < threshold)
    update_evaluation_list(mask, preds, targets, relation_label, constraint_dict, "10")

    # HiEve - case2: P(A|B) < threshold1 && P(B|A) > threshold1 => A and B are CP, B and A are PC
    # MATRES - case2: P(B|A) < threshold1 && P(A|B) > threshold1 => A is after B, B is before A
    mask = (volume1 < threshold) & (volume2 > threshold)
    update_evaluation_list(mask, preds, targets, relation_label, constraint_dict, "01")

    # HiEve - case3: P(A|B) > threshold1 && P(B|A) > threshold1 => CoRef
    # MATRES - case3: P(A|B) > threshold1 && P(B|A) > threshold1 => Equal
    mask = (volume1 > threshold) & (volume2 > threshold)
    update_evaluation_list(mask, preds, targets, relation_label, constraint_dict, "11")

    # HiEve - case4: P(A|B) < threshold1 && P(B|A) < threshold1 => NoRel
    # MATRES - case4: P(B|A) < threshold1 && P(A|B) < threshold1 => Vague
    mask = (volume1 < threshold) & (volume2 < threshold)
    update_evaluation_list(mask, preds, targets, relation_label, constraint_dict, "00")

    return preds, targets, constraint_dict


def update_evaluation_list(mask, preds, targets, relation_label, constraint_dict, key):
    mask_indices = mask.nonzero()
    preds.extend(mask_indices.shape[0] * [key])
    if mask_indices.shape[0] == 1:
        relation_label_list = [[x.item() for x in relation_label[mask_indices.squeeze()]]]
        mask_constraint_indices = [mask_indices.squeeze()]
    else:
        relation_label_list = relation_label[mask_indices.squeeze()].tolist()
        mask_constraint_indices = mask_indices.squeeze().tolist()
    targets.extend([''.join(map(str, item)) for item in relation_label_list])
    constraint_dict[key] = set(mask_constraint_indices)