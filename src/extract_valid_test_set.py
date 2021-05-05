from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import *

import torch
import random
import pickle

from tqdm import tqdm

from data_loader import padding, get_hieve_files, hieve_file_reader, get_matres_files, read_matres_files, \
    get_tml_dir_path, matres_file_reader


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def box_rel_id_to_vec_rel_id(box_rel_id):
    rel_id_dict = {(1,0): 0, (0,1): 1, (1,1): 2, (0,0): 3}
    return rel_id_dict[box_rel_id]


def extract_hieve_valid_test_constraint_set(data_dict: Dict[str, Any], downsample: float) -> List[Tuple]:
    train_box_set = []
    train_vec_set = []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    relation_dict = data_dict["relation_dict"]
    num_event = len(event_dict)

    for x in range(1, num_event+1):
        for y in range(x+1, num_event+1):
            for z in range(y+1, num_event+1):
                x_sntc_id = event_dict[x]["sent_id"]
                y_sntc_id = event_dict[y]["sent_id"]
                z_sntc_id = event_dict[z]["sent_id"]

                # get list of subword ids related to the specific sentence id
                # then padding the list of subword ids
                x_sntc = padding(sntc_dict[x_sntc_id]["roberta_subword_to_ID"], isPosTag=False)
                y_sntc = padding(sntc_dict[y_sntc_id]["roberta_subword_to_ID"], isPosTag=False)
                z_sntc = padding(sntc_dict[z_sntc_id]["roberta_subword_to_ID"], isPosTag=False)

                x_position = event_dict[x]["roberta_subword_id"]
                y_position = event_dict[y]["roberta_subword_id"]
                z_position = event_dict[z]["roberta_subword_id"]

                x_sntc_pos_tag = padding(sntc_dict[x_sntc_id]["roberta_subword_pos"], isPosTag=True)
                y_sntc_pos_tag = padding(sntc_dict[y_sntc_id]["roberta_subword_pos"], isPosTag=True)
                z_sntc_pos_tag = padding(sntc_dict[z_sntc_id]["roberta_subword_pos"], isPosTag=True)

                # rel_id: {"SuperSub": 0, "SubSuper": 1, "Coref": 2, "NoRel": 3}
                xy_rel_id = relation_dict[(x, y)]["relation"]
                yz_rel_id = relation_dict[(y, z)]["relation"]
                xz_rel_id = relation_dict[(x, z)]["relation"]

                if xy_rel_id == (0,0) and yz_rel_id == (0,0): pass # x-y: NoRel and y-z: NoRel
                elif xy_rel_id == (0,0) or yz_rel_id == (0,0) or xz_rel_id == (0,0): # if one of them is NoRel
                    if random.uniform(0, 1) < downsample:
                        box_to_append = \
                            str(x), str(y), str(z), \
                            x_sntc, y_sntc, z_sntc, \
                            x_position, y_position, z_position, \
                            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag, \
                            xy_rel_id, yz_rel_id, xz_rel_id, \
                            0  # 0: HiEve, 1: MATRES
                        train_box_set.append(box_to_append)
                        vec_to_append = \
                            str(x), str(y), str(z), \
                            x_sntc, y_sntc, z_sntc, \
                            x_position, y_position, z_position, \
                            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag, \
                            box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(yz_rel_id), box_rel_id_to_vec_rel_id(xz_rel_id), \
                            0  # 0: HiEve, 1: MATRES
                        train_vec_set.append(vec_to_append)
                else:
                    box_to_append = \
                        str(x), str(y), str(z), \
                        x_sntc, y_sntc, z_sntc, \
                        x_position, y_position, z_position, \
                        x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag, \
                        xy_rel_id, yz_rel_id, xz_rel_id, \
                        0  # 0: HiEve, 1: MATRES
                    train_box_set.append(box_to_append)
                    vec_to_append = \
                        str(x), str(y), str(z), \
                        x_sntc, y_sntc, z_sntc, \
                        x_position, y_position, z_position, \
                        x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag, \
                        box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(yz_rel_id), box_rel_id_to_vec_rel_id(xz_rel_id), \
                        0  # 0: HiEve, 1: MATRES
                    train_vec_set.append(vec_to_append)

    return train_box_set, train_vec_set


def extract_matres_valid_test_constraint_set(data_dict: Dict[str, Any], eiid_to_event_trigger_dict: Dict[int, str],
                         eiid_pair_to_rel_id_dict: Dict[Tuple[int], int]) -> List[Tuple]:
    """
    eiid_to_event_trigger_dict: eiid = trigger_word
    eiid_pair_to_rel_id_dict: (eiid1, eiid2) = relation_type_id
    """
    box_set, vec_set = [], []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    eiid_dict = data_dict["eiid_dict"]

    eiid_keys = eiid_to_event_trigger_dict.keys()
    eiid_pair_keys = eiid_pair_to_rel_id_dict.keys()
    for eiid1 in eiid_keys:
        for eiid2 in eiid_keys:
            for eiid3 in eiid_keys:
                if eiid1 != eiid2 and eiid2 != eiid3 and eiid1 != eiid3:
                    pair1 = (eiid1, eiid2)
                    pair2 = (eiid2, eiid3)
                    pair3 = (eiid1, eiid3)
                    if pair1 in eiid_pair_keys and pair2 in eiid_pair_keys and pair3 in eiid_pair_keys:
                        xy_rel_id = eiid_pair_to_rel_id_dict[pair1]
                        yz_rel_id = eiid_pair_to_rel_id_dict[pair2]
                        xz_rel_id = eiid_pair_to_rel_id_dict[pair3]

                        x_evnt_id = eiid_dict[eiid1]["eID"]
                        y_evnt_id = eiid_dict[eiid2]["eID"]
                        z_evnt_id = eiid_dict[eiid3]["eID"]

                        x_sntc_id = event_dict[x_evnt_id]["sent_id"]
                        y_sntc_id = event_dict[y_evnt_id]["sent_id"]
                        z_sntc_id = event_dict[z_evnt_id]["sent_id"]

                        x_sntc = padding(sntc_dict[x_sntc_id]["roberta_subword_to_ID"], isPosTag=False)
                        y_sntc = padding(sntc_dict[y_sntc_id]["roberta_subword_to_ID"], isPosTag=False)
                        z_sntc = padding(sntc_dict[z_sntc_id]["roberta_subword_to_ID"], isPosTag=False)

                        x_position = event_dict[x_evnt_id]["roberta_subword_id"]
                        y_position = event_dict[y_evnt_id]["roberta_subword_id"]
                        z_position = event_dict[z_evnt_id]["roberta_subword_id"]

                        x_sntc_pos_tag = padding(sntc_dict[x_sntc_id]["roberta_subword_pos"], isPosTag=True)
                        y_sntc_pos_tag = padding(sntc_dict[y_sntc_id]["roberta_subword_pos"], isPosTag=True)
                        z_sntc_pos_tag = padding(sntc_dict[z_sntc_id]["roberta_subword_pos"], isPosTag=True)

                        box_to_append = \
                            x_evnt_id, y_evnt_id, z_evnt_id,\
                            x_sntc, y_sntc, z_sntc,\
                            x_position, y_position, z_position,\
                            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag,\
                            xy_rel_id, yz_rel_id, xz_rel_id,\
                            1  # 0: HiEve, 1: MATRES
                        box_set.append(box_to_append)
                        vec_to_append = \
                            x_evnt_id, y_evnt_id, z_evnt_id,\
                            x_sntc, y_sntc, z_sntc,\
                            x_position, y_position, z_position,\
                            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag,\
                            box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(yz_rel_id), box_rel_id_to_vec_rel_id(xz_rel_id),\
                            1  # 0: HiEve, 1: MATRES
                        vec_set.append(vec_to_append)

    return box_set, vec_set


def extract_hieve_valid_test_set(data_dict: Dict[str, Any], downsample: float) -> List[Tuple]:
    train_box_set = []
    train_vec_set = []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    relation_dict = data_dict["relation_dict"]
    num_event = len(event_dict)

    for x in range(1, num_event + 1):
        for y in range(x + 1, num_event + 1):
            x_sntc_id = event_dict[x]["sent_id"]
            y_sntc_id = event_dict[y]["sent_id"]

            x_sntc = padding(sntc_dict[x_sntc_id]["roberta_subword_to_ID"])
            y_sntc = padding(sntc_dict[y_sntc_id]["roberta_subword_to_ID"])

            x_position = event_dict[x]["roberta_subword_id"]
            y_position = event_dict[y]["roberta_subword_id"]

            x_sntc_pos_tag = padding(sntc_dict[x_sntc_id]["roberta_subword_pos"], isPosTag=True)
            y_sntc_pos_tag = padding(sntc_dict[y_sntc_id]["roberta_subword_pos"], isPosTag=True)

            xy_rel_id = relation_dict[(x, y)]["relation"]

            if xy_rel_id == (0,0):
                if random.uniform(0, 1) < downsample:
                    box_to_append = \
                        str(x), str(y), str(x), \
                        x_sntc, y_sntc, x_sntc, \
                        x_position, y_position, x_position, \
                        x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag, \
                        xy_rel_id, xy_rel_id, xy_rel_id, \
                        0  # 0: HiEve, 1: MATRES
                    train_box_set.append(box_to_append)
                    vec_to_append = \
                        str(x), str(y), str(x), \
                        x_sntc, y_sntc, x_sntc, \
                        x_position, y_position, x_position, \
                        x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag, \
                        box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(xy_rel_id), \
                        0  # 0: HiEve, 1: MATRES
                    train_vec_set.append(vec_to_append)
            else:
                box_to_append = \
                    str(x), str(y), str(x), \
                    x_sntc, y_sntc, x_sntc, \
                    x_position, y_position, x_position, \
                    x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag, \
                    xy_rel_id, xy_rel_id, xy_rel_id, \
                    0  # 0: HiEve, 1: MATRES
                train_box_set.append(box_to_append)
                vec_to_append = \
                    str(x), str(y), str(x), \
                    x_sntc, y_sntc, x_sntc, \
                    x_position, y_position, x_position, \
                    x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag, \
                    box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(xy_rel_id), \
                    0  # 0: HiEve, 1: MATRES
                train_vec_set.append(vec_to_append)

    return train_box_set, train_vec_set


def extract_matres_valid_test_set(data_dict: Dict[str, Any], eiid_pair_to_rel_id_dict: Dict[Tuple[int], int]) -> List[Tuple]:
    """
    eiid_to_event_trigger_dict: eiid = trigger_word
    eiid_pair_to_rel_id_dict: (eiid1, eiid2) = relation_type_id
    """
    box_set, vec_set = [], []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    eiid_dict = data_dict["eiid_dict"]

    for (eiid1, eiid2) in eiid_pair_to_rel_id_dict.keys():
        xy_rel_id = eiid_pair_to_rel_id_dict[(eiid1, eiid2)]

        x_evnt_id = eiid_dict[eiid1]["eID"]
        y_evnt_id = eiid_dict[eiid2]["eID"]

        x_sntc_id = event_dict[x_evnt_id]["sent_id"]
        y_sntc_id = event_dict[y_evnt_id]["sent_id"]

        x_sntc = padding(sntc_dict[x_sntc_id]["roberta_subword_to_ID"], isPosTag=False)
        y_sntc = padding(sntc_dict[y_sntc_id]["roberta_subword_to_ID"], isPosTag=False)

        x_position = event_dict[x_evnt_id]["roberta_subword_id"]
        y_position = event_dict[y_evnt_id]["roberta_subword_id"]

        x_sntc_pos_tag = padding(sntc_dict[x_sntc_id]["roberta_subword_pos"], isPosTag=True)
        y_sntc_pos_tag = padding(sntc_dict[y_sntc_id]["roberta_subword_pos"], isPosTag=True)

        box_to_append = \
            x_evnt_id, y_evnt_id, x_evnt_id,\
            x_sntc, y_sntc, x_sntc,\
            x_position, y_position, x_position,\
            x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag,\
            xy_rel_id, xy_rel_id, xy_rel_id,\
            1  # 0: HiEve, 1: MATRES
        box_set.append(box_to_append)
        vec_to_append = \
            x_evnt_id, y_evnt_id, x_evnt_id,\
            x_sntc, y_sntc, x_sntc,\
            x_position, y_position, x_position,\
            x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag,\
            box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(xy_rel_id), box_rel_id_to_vec_rel_id(xy_rel_id),\
            1  # 0: HiEve, 1: MATRES
        vec_set.append(vec_to_append)

    return box_set, vec_set


def main():
    set_seed(10)
    data_dir = Path("../data").expanduser()
    hieve_dir, hieve_files = get_hieve_files(data_dir)

    all_valid_box_violation_set, all_test_box_violation_set = [], []
    all_valid_vec_violation_set, all_test_vec_violation_set = [], []
    all_valid_box_set, all_test_box_set = [], []
    all_valid_vec_set, all_test_vec_set = [], []

    train_range, valid_range, test_range = range(0, 60), range(60, 80), range(80, 100)
    downsample = 0.4
    for i, file in enumerate(tqdm(hieve_files)):
        data_dict = hieve_file_reader(hieve_dir, file, "box")
        doc_id = i
        if doc_id in valid_range:
            valid_box_set, valid_vec_set = extract_hieve_valid_test_constraint_set(data_dict, downsample)
            all_valid_box_violation_set.extend(valid_box_set)
            all_valid_vec_violation_set.extend(valid_vec_set)

            valid_box_set, valid_vec_set = extract_hieve_valid_test_set(data_dict, downsample)
            all_valid_box_set.extend(valid_box_set)
            all_valid_vec_set.extend(valid_vec_set)
        elif doc_id in test_range:
            test_box_set, test_vec_set = extract_hieve_valid_test_constraint_set(data_dict, downsample)
            all_test_box_violation_set.extend(test_box_set)
            all_test_vec_violation_set.extend(test_vec_set)

            test_box_set, test_vec_set = extract_hieve_valid_test_set(data_dict, downsample)
            all_test_box_set.extend(test_box_set)
            all_test_vec_set.extend(test_vec_set)

    # constraint violation valid/test dataset
    with open('hieve_valid_cv_vector.pickle', 'wb') as handle:
        pickle.dump(all_valid_vec_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hieve_valid_cv_box.pickle', 'wb') as handle:
        pickle.dump(all_valid_box_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hieve_test_cv_vector.pickle', 'wb') as handle:
        pickle.dump(all_test_vec_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hieve_test_cv_box.pickle', 'wb') as handle:
        pickle.dump(all_test_box_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # valid/test dataset
    with open('hieve_valid_vector.pickle', 'wb') as handle:
        pickle.dump(all_valid_vec_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hieve_valid_box.pickle', 'wb') as handle:
        pickle.dump(all_valid_box_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hieve_test_vector.pickle', 'wb') as handle:
        pickle.dump(all_test_vec_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('hieve_test_box.pickle', 'wb') as handle:
        pickle.dump(all_test_box_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    all_tml_dir_path_dict, all_tml_file_dict, all_txt_file_path = get_matres_files(data_dir)
    eiid_to_event_trigger, eiid_pair_to_rel_id = read_matres_files(all_txt_file_path, "box")

    all_valid_box_violation_set, all_test_box_violation_set = [], []
    all_valid_vec_violation_set, all_test_vec_violation_set = [], []
    all_valid_box_set, all_test_box_set = [], []
    all_valid_vec_set, all_test_vec_set = [], []
    for i, fname in enumerate(tqdm(eiid_pair_to_rel_id.keys())):
        file_name = fname + ".tml"
        dir_path = get_tml_dir_path(file_name, all_tml_dir_path_dict, all_tml_file_dict)
        data_dict = matres_file_reader(dir_path, file_name, eiid_to_event_trigger)

        eiid_to_event_trigger_dict = eiid_to_event_trigger[fname]
        eiid_pair_to_rel_id_dict = eiid_pair_to_rel_id[fname]
        if file_name in all_tml_file_dict["tb"]:
            pass
        elif file_name in all_tml_file_dict["aq"]:
            valid_box_set, valid_vec_set = extract_matres_valid_test_constraint_set(data_dict, eiid_to_event_trigger_dict, eiid_pair_to_rel_id_dict)
            all_valid_box_violation_set.extend(valid_box_set)
            all_valid_vec_violation_set.extend(valid_vec_set)
            valid_box_set, valid_vec_set = extract_matres_valid_test_set(data_dict, eiid_pair_to_rel_id_dict)
            all_valid_box_set.extend(valid_box_set)
            all_valid_vec_set.extend(valid_vec_set)
        elif file_name in all_tml_file_dict["pl"]:
            test_box_set, test_vec_set = extract_matres_valid_test_constraint_set(data_dict, eiid_to_event_trigger_dict, eiid_pair_to_rel_id_dict)
            all_test_box_violation_set.extend(test_box_set)
            all_test_vec_violation_set.extend(test_vec_set)
            test_box_set, test_vec_set = extract_matres_valid_test_set(data_dict, eiid_pair_to_rel_id_dict)
            all_test_box_set.extend(test_box_set)
            all_test_vec_set.extend(test_vec_set)
        else:
            raise ValueError(f"file_name={file_name} does not exist in MATRES dataset!")

    # constraint violation valid/test dataset
    with open('matres_valid_cv_vector.pickle', 'wb') as handle:
        pickle.dump(all_valid_vec_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('matres_valid_cv_box.pickle', 'wb') as handle:
        pickle.dump(all_valid_box_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('matres_test_cv_vector.pickle', 'wb') as handle:
        pickle.dump(all_test_vec_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('matres_test_cv_box.pickle', 'wb') as handle:
        pickle.dump(all_test_box_violation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # valid/test dataset
    with open('matres_valid_vector.pickle', 'wb') as handle:
        pickle.dump(all_valid_vec_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('matres_valid_box.pickle', 'wb') as handle:
        pickle.dump(all_valid_box_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('matres_test_vector.pickle', 'wb') as handle:
        pickle.dump(all_test_vec_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('matres_test_box.pickle', 'wb') as handle:
        pickle.dump(all_test_box_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

def check():
    with open('/Users/ehwang/PycharmProjects/CE2ERE/data/matres_valid_test_set/matres_valid_vector.pickle', 'rb') as handle:
        valid_vector_origin = pickle.load(handle)
        print(len(valid_vector_origin))
    with open('/Users/ehwang/PycharmProjects/CE2ERE/data/matres_valid_test_set/matres_valid_box.pickle', 'rb') as handle:
        valid_box_origin = pickle.load(handle)
        print(len(valid_box_origin))
    with open('/Users/ehwang/PycharmProjects/CE2ERE/data/matres_valid_test_set/matres_test_vector.pickle', 'rb') as handle:
        test_vector_origin = pickle.load(handle)
        print(len(test_vector_origin))
    with open('/Users/ehwang/PycharmProjects/CE2ERE/data/matres_valid_test_set/matres_test_box.pickle', 'rb') as handle:
        test_box_origin = pickle.load(handle)
        print(len(test_box_origin))

    # cv: valid/test for constraint violation evaluation. Uses the train dataloader which includes x,y,z
    # regular valid/test only includes x, y, x
    with open('matres_valid_cv_vector.pickle', 'rb') as handle:
        valid_vector = pickle.load(handle)
        print(len(valid_vector))
    with open('matres_valid_cv_box.pickle', 'rb') as handle:
        valid_box = pickle.load(handle)
        print(len(valid_box))
    with open('matres_test_cv_vector.pickle', 'rb') as handle:
        test_vector = pickle.load(handle)
        print(len(test_vector))
    with open('matres_test_cv_box.pickle', 'rb') as handle:
        test_box = pickle.load(handle)
        print(len(test_box))

    assert (valid_vector_origin == valid_vector)
    assert (valid_box_origin == valid_box)
    assert (test_box_origin == test_box)
    assert (test_vector_origin == test_vector)


if __name__ == '__main__':
    # main()
    check()