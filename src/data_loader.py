import random
import time

from typing import *
from tqdm import tqdm
from data_reader import hieve_file_reader
from utils import *

import torch

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

def get_hieve_train_set(data_dict: Dict[str, Any], downsample: float) -> List[Tuple]:
    train_set = []
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

                to_append = (
                    str(x), str(y), str(z),
                    x_sntc, y_sntc, z_sntc,
                    x_position, y_position, z_position,
                    x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag,
                    xy_rel_id, yz_rel_id, xz_rel_id,
                    0 # 0: HiEve, 1: MATRES
                )

                if xy_rel_id == 3 and yz_rel_id == 3: pass # x-y: NoRel and y-z: NoRel
                elif xy_rel_id == 3 or yz_rel_id == 3 or xz_rel_id == 3: # if one of them is NoRel
                    if random.uniform(0, 1) < downsample:
                        train_set.append(to_append)
                else:
                    train_set.append(to_append)
    return train_set


def get_hieve_valid_test_set(data_dict: Dict[str, Any], undersmp_ratio: float,
                             doc_id: int, valid_range: List[int], test_range: List[int]) -> List[Tuple]:
    valid_set, test_set = [], []
    event_dict = data_dict["event_dict"]
    sntc_dict = data_dict["sentences"]
    relation_dict = data_dict["relation_dict"]
    num_event = len(event_dict)

    for x in range(1, num_event+1):
        for y in range(x+1, num_event+1):
            x_sntc_id = event_dict[x]["sent_id"]
            y_sntc_id = event_dict[y]["sent_id"]

            x_sntc = padding(sntc_dict[x_sntc_id]["roberta_subword_to_ID"])
            y_sntc = padding(sntc_dict[y_sntc_id]["roberta_subword_to_ID"])

            x_position = event_dict[x]["roberta_subword_id"]
            y_position = event_dict[y]["roberta_subword_id"]

            x_sntc_pos_tag = padding(sntc_dict[x_sntc_id]["roberta_subword_pos"], isPosTag=True)
            y_sntc_pos_tag = padding(sntc_dict[y_sntc_id]["roberta_subword_pos"], isPosTag=True)

            xy_rel_id = relation_dict[(x, y)]["relation"]

            to_append = (
                str(x), str(y), str(x),
                x_sntc, y_sntc, x_sntc,
                x_position, y_position, x_position,
                x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag,
                xy_rel_id, xy_rel_id, xy_rel_id,
                0 # 0: HiEve, 1: MATRES
            )

            if doc_id in valid_range:
                if xy_rel_id == 3:
                    if random.uniform(0, 1) < undersmp_ratio:
                        valid_set.append(to_append)
                else:
                    valid_set.append(to_append)
            elif doc_id in test_range:
                if xy_rel_id == 3:
                    if random.uniform(0, 1) < undersmp_ratio:
                        test_set.append(to_append)
                else:
                    test_set.append(to_append)
            else:
                raise ValueError("doc_id=%d is out of range!" % (doc_id))

    return valid_set, test_set


def hieve_data_loader(args: Dict[str, Any], data_dir: Union[Path, str]):
    hieve_dir, hieve_files = get_files(data_dir, "hieve")
    all_train_set, all_valid_set, all_test_set = [], [], []
    train_range, valid_range, test_range = range(0, 60), range(60, 80), range(80, 100)

    start_time = time.time()
    for i, file in enumerate(tqdm(hieve_files)):
        data_dict = hieve_file_reader(hieve_dir, file)
        doc_id = i

        if doc_id in train_range:
            train_set = get_hieve_train_set(data_dict, args.downsample)
            all_train_set.extend(train_set)
        else:
            valid_set, test_set = get_hieve_valid_test_set(data_dict, 0.4, doc_id, valid_range, test_range)
            all_valid_set.extend(valid_set)
            all_test_set.extend(test_set)
    elapsed_time = format_time(time.time() - start_time)

    print("HiEve Preprocessing took {:}".format(elapsed_time))
    print(f'HiEve training instance num: {len(all_train_set)}')

    return all_train_set, all_valid_set, all_test_set
