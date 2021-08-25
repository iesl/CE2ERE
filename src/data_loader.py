import ast
import json
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

from EventDataset import EventDataset
from data_reader import *
from utils import *
from typing import List, Tuple, Dict, Any, Optional, Union
from natsort import natsorted

logger = logging.getLogger()

# Padding function
def padding(subword_ids: List[int], isPosTag: Optional[bool] = False, max_sent_len: Optional[int] = 120):
    if isPosTag == False:
        one_list = [1] * max_sent_len
        one_list[0:len(subword_ids)] = subword_ids
        return torch.tensor(one_list, dtype=torch.long)
    else:
        one_list = ["None"] * max_sent_len
        one_list[0:len(subword_ids)] = subword_ids
        return one_list

def get_hieve_train_set(data_dict: Dict[str, Any], downsample: float, model_type: str, symm_train: Optional[int]=0) -> List[Tuple]:
    train_set = []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    relation_dict = data_dict["relation_dict"]
    num_event = len(event_dict)

    for x in range(1, num_event+1):
        for y in range(x+1, num_event+1):
            for z in range(y+1, num_event+1):
                append_hieve_train_dataset(train_set, downsample, model_type, x, y, z, event_dict, sntc_dict, relation_dict)
                if symm_train and (model_type == "box" or model_type == "vector"):
                    if relation_dict[(x, y)]["relation"] == (1, 0) or relation_dict[(x, y)]["relation"] == (0, 1):
                        if (y, x) in relation_dict.keys() and (x, z) in relation_dict.keys() and (y, z) in relation_dict.keys():
                            append_hieve_train_dataset(train_set, downsample, model_type, y, x, z, event_dict, sntc_dict, relation_dict)
                    if relation_dict[(y, z)]["relation"] == (1, 0) or relation_dict[(y, z)]["relation"] == (0, 1):
                        if (x, z) in relation_dict.keys() and (z, y) in relation_dict.keys() and (x, y) in relation_dict.keys():
                            append_hieve_train_dataset(train_set, downsample, model_type, x, z, y, event_dict, sntc_dict, relation_dict)
                    if relation_dict[(x, z)]["relation"] == (1, 0) or relation_dict[(x, z)]["relation"] == (0, 1):
                        if (z, y) in relation_dict.keys() and (y, x) in relation_dict.keys() and (z, x) in relation_dict.keys():
                            append_hieve_train_dataset(train_set, downsample, model_type, z, y, x, event_dict, sntc_dict, relation_dict)
                elif symm_train and model_type == "bilstm":
                    if relation_dict[(x, y)]["relation"] == 0 or relation_dict[(x, y)]["relation"] == 1:
                        if (y, x) in relation_dict.keys() and (x, z) in relation_dict.keys() and (y, z) in relation_dict.keys():
                            append_hieve_train_dataset(train_set, downsample, model_type, y, x, z, event_dict, sntc_dict, relation_dict)
                    if relation_dict[(y, z)]["relation"] == 0 or relation_dict[(y, z)]["relation"] == 1:
                        if (x, z) in relation_dict.keys() and (z, y) in relation_dict.keys() and (x, y) in relation_dict.keys():
                            append_hieve_train_dataset(train_set, downsample, model_type, x, z, y, event_dict, sntc_dict, relation_dict)
                    if relation_dict[(x, z)]["relation"] == 0 or relation_dict[(x, z)]["relation"] == 1:
                        if (z, y) in relation_dict.keys() and (y, x) in relation_dict.keys() and (z, x) in relation_dict.keys():
                            append_hieve_train_dataset(train_set, downsample, model_type, z, y, x, event_dict, sntc_dict, relation_dict)
    return train_set


def append_hieve_train_dataset(train_set, downsample, model_type, x, y, z, event_dict, sntc_dict, relation_dict):
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

    to_append = \
        str(x), str(y), str(z), \
        x_sntc, y_sntc, z_sntc, \
        x_position, y_position, z_position, \
        x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag, \
        xy_rel_id, yz_rel_id, xz_rel_id, \
        0  # 0: HiEve, 1: MATRES

    if model_type == "box" or model_type == "vector":
        if xy_rel_id == (0, 0) and yz_rel_id == (0, 0):
            pass  # x-y: NoRel and y-z: NoRel
        elif xy_rel_id == (0, 0) or yz_rel_id == (0, 0) or xz_rel_id == (0, 0):  # if one of them is NoRel
            if random.uniform(0, 1) < downsample:
                train_set.append(to_append)
        else:
            train_set.append(to_append)
    else:
        if xy_rel_id == 3 and yz_rel_id == 3:
            pass  # x-y: NoRel and y-z: NoRel
        elif xy_rel_id == 3 or yz_rel_id == 3 or xz_rel_id == 3:  # if one of them is NoRel
            if random.uniform(0, 1) < downsample:
                train_set.append(to_append)
        else:
            train_set.append(to_append)


def get_hieve_valid_test_set(data_dict: Dict[str, Any], downsample: float, model_type: str, symm_eval: int) -> List[Tuple]:
    final_set = []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    relation_dict = data_dict["relation_dict"]
    num_event = len(event_dict)

    for x in range(1, num_event+1):
        for y in range(x+1, num_event+1):
            append_hieve_eval_dataset(final_set, downsample, model_type, x, y, event_dict, sntc_dict, relation_dict)
            if symm_eval and (model_type == "box" or model_type == "vector"):
                if relation_dict[(x, y)]["relation"] == (1, 0) or relation_dict[(x, y)]["relation"] == (0, 1):
                    append_hieve_eval_dataset(final_set, downsample, model_type, y, x, event_dict, sntc_dict, relation_dict)
            elif symm_eval and (model_type == "bilstm"):
                if relation_dict[(x, y)]["relation"] == 0 or relation_dict[(x, y)]["relation"] == 1:
                    append_hieve_eval_dataset(final_set, downsample, model_type, y, x, event_dict, sntc_dict, relation_dict)
    return final_set


def append_hieve_eval_dataset(final_set, downsample, model_type, x, y, event_dict, sntc_dict, relation_dict):
    x_sntc_id = event_dict[x]["sent_id"]
    y_sntc_id = event_dict[y]["sent_id"]

    x_sntc = padding(sntc_dict[x_sntc_id]["roberta_subword_to_ID"])
    y_sntc = padding(sntc_dict[y_sntc_id]["roberta_subword_to_ID"])

    x_position = event_dict[x]["roberta_subword_id"]
    y_position = event_dict[y]["roberta_subword_id"]

    x_sntc_pos_tag = padding(sntc_dict[x_sntc_id]["roberta_subword_pos"], isPosTag=True)
    y_sntc_pos_tag = padding(sntc_dict[y_sntc_id]["roberta_subword_pos"], isPosTag=True)

    xy_rel_id = relation_dict[(x, y)]["relation"]

    to_append = \
        str(x), str(y), str(x), \
        x_sntc, y_sntc, x_sntc, \
        x_position, y_position, x_position, \
        x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag, \
        xy_rel_id, xy_rel_id, xy_rel_id, \
        0  # 0: HiEve, 1: MATRES

    if model_type == "box" or model_type == "vector":
        if xy_rel_id == (0, 0):
            if random.uniform(0, 1) < downsample:
                final_set.append(to_append)
        else:
            final_set.append(to_append)
    else:
        if xy_rel_id == 3:
            if random.uniform(0, 1) < downsample:
                final_set.append(to_append)
        else:
            final_set.append(to_append)


def get_matres_train_set(data_dict: Dict[str, Any], eiid_to_event_trigger_dict: Dict[int, str],
                         eiid_pair_to_rel_id_dict: Dict[Tuple[int], int], model_type: str, symm_train: Optional[int]=0) -> List[Tuple]:
    """
    eiid_to_event_trigger_dict: eiid = trigger_word
    eiid_pair_to_rel_id_dict: (eiid1, eiid2) = relation_type_id
    """
    train_set = []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    eiid_dict = data_dict["eiid_dict"]

    eiid_keys = eiid_to_event_trigger_dict.keys()
    for eiid1 in eiid_keys:
        for eiid2 in eiid_keys:
            for eiid3 in eiid_keys:
                if eiid1 != eiid2 and eiid2 != eiid3 and eiid1 != eiid3:
                    append_matres_train_dataset(train_set, eiid1, eiid2, eiid3, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
                    if not symm_train: continue
                    eiid_pair_keys = eiid_pair_to_rel_id_dict.keys()
                    if model_type == "box" or model_type == "vector":
                        if (eiid1, eiid2) in eiid_pair_keys and (eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == (1, 0) or eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == (0, 1)):
                            append_matres_train_dataset(train_set, eiid2, eiid1, eiid3, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
                        if (eiid2, eiid3) in eiid_pair_keys and (eiid_pair_to_rel_id_dict[(eiid2, eiid3)] == (1, 0) or eiid_pair_to_rel_id_dict[(eiid2, eiid3)] == (0, 1)):
                            append_matres_train_dataset(train_set, eiid1, eiid3, eiid2, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
                        if (eiid1, eiid3) in eiid_pair_keys and (eiid_pair_to_rel_id_dict[(eiid1, eiid3)] == (1, 0) or eiid_pair_to_rel_id_dict[(eiid1, eiid3)] == (0, 1)):
                            append_matres_train_dataset(train_set, eiid3, eiid2, eiid1, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
                    else:
                        if (eiid1, eiid2) in eiid_pair_keys and (eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == 0 or eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == 1):
                            append_matres_train_dataset(train_set, eiid2, eiid1, eiid3, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
                        if (eiid2, eiid3) in eiid_pair_keys and (eiid_pair_to_rel_id_dict[(eiid2, eiid3)] == 0 or eiid_pair_to_rel_id_dict[(eiid2, eiid3)] == 1):
                            append_matres_train_dataset(train_set, eiid1, eiid3, eiid2, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
                        if (eiid1, eiid3) in eiid_pair_keys and (eiid_pair_to_rel_id_dict[(eiid1, eiid3)] == 0 or eiid_pair_to_rel_id_dict[(eiid1, eiid3)] == 1):
                            append_matres_train_dataset(train_set, eiid3, eiid2, eiid1, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
    return train_set


def append_matres_train_dataset(train_set, eiid1, eiid2, eiid3, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict):
    eiid_pair_keys = eiid_pair_to_rel_id_dict.keys()
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

        to_append = \
            x_evnt_id, y_evnt_id, z_evnt_id, \
            x_sntc, y_sntc, z_sntc, \
            x_position, y_position, z_position, \
            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag, \
            xy_rel_id, yz_rel_id, xz_rel_id, \
            1  # 0: HiEve, 1: MATRES

        train_set.append(to_append)


def get_matres_valid_test_set(data_dict: Dict[str, Any], eiid_pair_to_rel_id_dict: Dict[Tuple[int], int], model_type: str, symm_eval: int):
    final_set = []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    eiid_dict = data_dict["eiid_dict"]

    for (eiid1, eiid2) in eiid_pair_to_rel_id_dict.keys():
        append_matres_eval_dataset(final_set, eiid1, eiid2, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
        # box
        if model_type == "box" or model_type == "vector":
            if symm_eval and (eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == (1, 0) or eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == (0, 1)):
                append_matres_eval_dataset(final_set, eiid2, eiid1, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
        else:
            if symm_eval and (eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == 0 or eiid_pair_to_rel_id_dict[(eiid1, eiid2)] == 1):
                append_matres_eval_dataset(final_set, eiid2, eiid1, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict)
    return final_set


def append_matres_eval_dataset(final_set, eiid1, eiid2, event_dict, sntc_dict, eiid_dict, eiid_pair_to_rel_id_dict):
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

    to_append = \
        x_evnt_id, y_evnt_id, x_evnt_id, \
        x_sntc, y_sntc, x_sntc, \
        x_position, y_position, x_position, \
        x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag, \
        xy_rel_id, xy_rel_id, xy_rel_id, \
        1  # 0: HiEve, 1: MATRES

    final_set.append(to_append)


def hieve_data_loader(args: Dict[str, Any], data_dir: Union[Path, str]) -> Tuple[List[Any]]:
    hieve_dir = data_dir / "hievents_v2/processed/"
    all_train_set, all_valid_set, all_test_set = [], [], []
    all_valid_cv_set, all_test_cv_set = [], []

    hieve_files = natsorted([f for f in listdir(hieve_dir) if isfile(join(hieve_dir, f)) and f[-4:] == "tsvx"])
    train_range, valid_range, test_range = [], [], []
    with open(data_dir / "hievents_v2/sorted_dict.json") as f:
        sorted_dict = json.load(f)
    i = 0
    for (key, value) in sorted_dict.items():
        i += 1
        key = int(key)
        if i <= 20:
            test_range.append(key)
        elif i <= 40:
            valid_range.append(key)
        else:
            train_range.append(key)

    hieve_train, hieve_valid, hieve_test = [], [], []
    for i, file in enumerate(hieve_files):
        if i in train_range:
            hieve_train.append(file)
        elif i in valid_range:
            hieve_valid.append(file)
        elif i in test_range:
            hieve_test.append(file)

    # logger.info("train files: "+str(hieve_train))
    # logger.info("valid files: "+str(hieve_valid))
    # logger.info("test files: "+str(hieve_test))

    start_time = time.time()
    print("HiEve train files processing...")
    for i, file in enumerate(tqdm(hieve_train)):
        data_dict = hieve_file_reader(hieve_dir, file, args.model, args.symm_train)  # data_reader.py
        train_set = get_hieve_train_set(data_dict, args.downsample, args.model, args.symm_train)
        all_train_set.extend(train_set)
    print("done!")

    with temp_seed(10):
        print("HiEve valid files processing...")
        for i, file in enumerate(tqdm(hieve_valid)):
            data_dict = hieve_file_reader(hieve_dir, file, args.model, args.symm_eval)
            valid_set = get_hieve_valid_test_set(data_dict, 0.4, args.model, args.symm_eval)
            all_valid_set.extend(valid_set)

    with temp_seed(10):
        for i, file in enumerate(tqdm(hieve_valid)):
            data_dict = hieve_file_reader(hieve_dir, file, args.model, args.symm_eval)
            cv_valid_set = get_hieve_train_set(data_dict, 0.4, args.model, args.symm_eval)
            all_valid_cv_set.extend(cv_valid_set)
        print("done!")

    with temp_seed(10):
        print("HiEve test files processing...")
        for i, file in enumerate(tqdm(hieve_test)):
            data_dict = hieve_file_reader(hieve_dir, file, args.model, args.symm_eval)
            test_set = get_hieve_valid_test_set(data_dict, 0.4, args.model, args.symm_eval)
            all_test_set.extend(test_set)

    with temp_seed(10):
        for i, file in enumerate(tqdm(hieve_test)):
            data_dict = hieve_file_reader(hieve_dir, file, args.model, args.symm_eval)
            cv_test_set = get_hieve_train_set(data_dict, 0.4, args.model, args.symm_eval)
            all_test_cv_set.extend(cv_test_set)
        print("done!")

    elapsed_time = format_time(time.time() - start_time)
    logger.info("HiEve Preprocessing took {:}".format(elapsed_time))
    logger.info(f'HiEve training instance num: {len(all_train_set)}, '
          f'valid instance num: {len(all_valid_set)}, '
          f'test instance num: {len(all_test_set)}, '
          f'cv-valid instance num: {len(all_valid_cv_set)}, '
          f'cv-test instance num: {len(all_test_cv_set)}, ')

    if args.debug:
        logger.info("debug mode on")
        all_train_set = all_train_set[0:100]
        all_valid_set = all_train_set
        all_test_set = all_train_set
        all_valid_cv_set = all_train_set
        all_test_cv_set = all_train_set
        logger.info("hieve length debugging mode: %d".format(len(all_train_set)))

    return all_train_set, all_valid_set, all_test_set, all_valid_cv_set, all_test_cv_set


def matres_data_loader(args: Dict[str, Any], data_dir: Union[Path, str]) -> Tuple[List[Any]]:
    all_tml_dir_path_dict, all_tml_file_dict, all_txt_file_path = get_matres_files(data_dir)
    eiid_to_event_trigger, eiid_pair_to_rel_id = read_matres_files(all_txt_file_path, args.model, args.symm_eval or args.symm_train)

    all_train_set, all_valid_set, all_test_set = [], [], []
    all_valid_cv_set, all_test_cv_set = [], []
    start_time = time.time()
    for i, fname in enumerate(tqdm(eiid_pair_to_rel_id.keys())):
        file_name = fname + ".tml"
        dir_path = get_tml_dir_path(file_name, all_tml_dir_path_dict, all_tml_file_dict) # get directory corresponding to filename
        data_dict = matres_file_reader(dir_path, file_name, eiid_to_event_trigger)

        eiid_to_event_trigger_dict = eiid_to_event_trigger[fname]
        eiid_pair_to_rel_id_dict = eiid_pair_to_rel_id[fname]
        if file_name in all_tml_file_dict["tb"]:
            train_set = get_matres_train_set(data_dict, eiid_to_event_trigger_dict, eiid_pair_to_rel_id_dict, args.symm_train)
            all_train_set.extend(train_set)
        elif file_name in all_tml_file_dict["aq"]:
            valid_set = get_matres_valid_test_set(data_dict, eiid_pair_to_rel_id_dict, args.model, args.symm_eval)
            all_valid_set.extend(valid_set)
            cv_valid_set = get_matres_train_set(data_dict, eiid_to_event_trigger_dict, eiid_pair_to_rel_id_dict, args.model, args.symm_eval)
            all_valid_cv_set.extend(cv_valid_set)
        elif file_name in all_tml_file_dict["pl"]:
            test_set = get_matres_valid_test_set(data_dict, eiid_pair_to_rel_id_dict, args.model, args.symm_eval)
            all_test_set.extend(test_set)
            cv_test_set = get_matres_train_set(data_dict, eiid_to_event_trigger_dict, eiid_pair_to_rel_id_dict, args.model, args.symm_eval)
            all_test_cv_set.extend(cv_test_set)
        else:
            raise ValueError(f"file_name={file_name} does not exist in MATRES dataset!")

    elapsed_time = format_time(time.time() - start_time)
    logger.info("MATRES Preprocessing took {:}".format(elapsed_time))
    logger.info(f'MATRES training instance num: {len(all_train_set)}, '
          f'valid instance num: {len(all_valid_set)}, '
          f'test instance num: {len(all_test_set)}, '
          f'cv-valid instance num: {len(all_valid_cv_set)}, '
          f'cv-test instance num: {len(all_test_cv_set)}')
    if args.debug:
        logger.info("debug mode on")
        all_train_set = all_train_set[0:100]
        all_valid_set = all_train_set
        all_test_set = all_train_set
        all_valid_cv_set = all_train_set
        all_test_cv_set = all_train_set

    return all_train_set, all_valid_set, all_test_set, all_valid_cv_set, all_test_cv_set


def get_dataloaders(log_batch_size: int, train_set: List, valid_set_dict: Dict[str, List], test_set_dict: Dict[str, List],
                    valid_cv_set_dict: Dict[str, List], test_cv_set_dict: Dict[str, List]) -> Tuple[DataLoader]:

    train_dataloader = DataLoader(EventDataset(train_set), batch_size=2 ** log_batch_size, shuffle=True)
    valid_dataloader_dict, test_dataloader_dict = {}, {}

    # create validation dataloader
    for data_type, valid_set in valid_set_dict.items():
        if data_type == "hieve":
            valid_dataloader = DataLoader(EventDataset(valid_set), batch_size=2 ** log_batch_size, shuffle=False)
        elif data_type == "matres":
            valid_dataloader = DataLoader(EventDataset(valid_set), batch_size=2 ** log_batch_size, shuffle=False)
        else:
            raise ValueError(f"dataset={data_type} is not supported at this time!")
        valid_dataloader_dict[data_type] = valid_dataloader

    # create test dataloader
    for data_type, test_set in test_set_dict.items():
        if data_type == "hieve":
            test_dataloader = DataLoader(EventDataset(test_set), batch_size=2 ** log_batch_size, shuffle=False)
        elif data_type == "matres":
            test_dataloader = DataLoader(EventDataset(test_set), batch_size=2 ** log_batch_size, shuffle=False)
        else:
            raise ValueError(f"dataset={data_type} is not supported at this time!")
        test_dataloader_dict[data_type] = test_dataloader

    valid_cv_dataloader_dict, test_cv_dataloader_dict = {}, {}
    # create validation constraint violdation dataloader
    for data_type, valid_set in valid_cv_set_dict.items():
        if data_type == "hieve":
            valid_dataloader = DataLoader(EventDataset(valid_set), batch_size=2 ** log_batch_size, shuffle=False)
        elif data_type == "matres":
            valid_dataloader = DataLoader(EventDataset(valid_set), batch_size=2 ** log_batch_size, shuffle=False)
        else:
            raise ValueError(f"dataset={data_type} is not supported at this time!")
        valid_cv_dataloader_dict[data_type] = valid_dataloader

    # create test constraint violdation dataloader
    for data_type, test_set in test_cv_set_dict.items():
        if data_type == "hieve":
            test_dataloader = DataLoader(EventDataset(test_set), batch_size=2 ** log_batch_size, shuffle=False)
        elif data_type == "matres":
            test_dataloader = DataLoader(EventDataset(test_set), batch_size=2 ** log_batch_size, shuffle=False)
        else:
            raise ValueError(f"dataset={data_type} is not supported at this time!")
        test_cv_dataloader_dict[data_type] = test_dataloader

    return train_dataloader, valid_dataloader_dict, test_dataloader_dict, valid_cv_dataloader_dict, test_cv_dataloader_dict