import random
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from EventDataset import EventDataset
from data_reader import *
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

def get_hieve_train_set(data_dict: Dict[str, Any], downsample: float, device: torch.device) -> List[Tuple]:
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

                to_append = \
                    str(x), str(y), str(z),\
                    x_sntc, y_sntc, z_sntc,\
                    x_position, y_position, z_position,\
                    x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag,\
                    xy_rel_id, yz_rel_id, xz_rel_id,\
                    0 # 0: HiEve, 1: MATRES


                if xy_rel_id == 3 and yz_rel_id == 3: pass # x-y: NoRel and y-z: NoRel
                elif xy_rel_id == 3 or yz_rel_id == 3 or xz_rel_id == 3: # if one of them is NoRel
                    if random.uniform(0, 1) < downsample:
                        train_set.append(to_append)
                else:
                    train_set.append(to_append)
    return train_set


def get_hieve_valid_test_set(data_dict: Dict[str, Any], undersmp_ratio: float, device: torch.device) -> List[Tuple]:
    final_set = []
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

            to_append = \
                str(x), str(y), str(x),\
                x_sntc, y_sntc, x_sntc,\
                x_position, y_position, x_position,\
                x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag,\
                xy_rel_id, xy_rel_id, xy_rel_id,\
                0 # 0: HiEve, 1: MATRES


            if xy_rel_id == 3:
                if random.uniform(0, 1) < undersmp_ratio:
                    final_set.append(to_append)
            else:
                final_set.append(to_append)

    return final_set


def get_matres_train_set(data_dict: Dict[str, Any], eiid_to_event_trigger_dict: Dict[int, str],
                         eiid_pair_to_rel_id_dict: Dict[Tuple[int], int], device: torch.device) -> List[Tuple]:
    """
    eiid_to_event_trigger_dict: eiid = trigger_word
    eiid_pair_to_rel_id_dict: (eiid1, eiid2) = relation_type_id
    """
    train_set = []
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

                        to_append = \
                            x_evnt_id, y_evnt_id, z_evnt_id,\
                            x_sntc, y_sntc, z_sntc,\
                            x_position, y_position, z_position,\
                            x_sntc_pos_tag, y_sntc_pos_tag, z_sntc_pos_tag,\
                            xy_rel_id, yz_rel_id, xz_rel_id,\
                            1  # 0: HiEve, 1: MATRES


                        train_set.append(to_append)
    return train_set


def get_matres_valid_test_set(data_dict: Dict[str, Any], eiid_pair_to_rel_id_dict: Dict[Tuple[int], int], device: torch.device):
    final_set = []
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

        to_append = \
            x_evnt_id, y_evnt_id, x_evnt_id,\
            x_sntc, y_sntc, x_sntc,\
            x_position, y_position, x_position,\
            x_sntc_pos_tag, y_sntc_pos_tag, x_sntc_pos_tag,\
            xy_rel_id, xy_rel_id, xy_rel_id,\
            1 # 0: HiEve, 1: MATRES

        final_set.append(to_append)

    return final_set


def hieve_data_loader(args: Dict[str, Any], data_dir: Union[Path, str], device: torch.device) -> Tuple[List[Any]]:
    hieve_dir, hieve_files = get_hieve_files(data_dir)
    all_train_set, all_valid_set, all_test_set = [], [], []

    train_range, valid_range, test_range = range(0, 60), range(60, 80), range(80, 100)

    start_time = time.time()
    for i, file in enumerate(tqdm(hieve_files)):
        data_dict = hieve_file_reader(hieve_dir, file)
        doc_id = i

        if doc_id in train_range:
            train_set = get_hieve_train_set(data_dict, args.downsample, device)
            all_train_set.extend(train_set)
        elif doc_id in valid_range:
            valid_set = get_hieve_valid_test_set(data_dict, args.downsample, device)
            all_valid_set.extend(valid_set)
        elif doc_id in test_range:
            test_set = get_hieve_valid_test_set(data_dict, args.downsample, device)
            all_test_set.extend(test_set)
        else:
            raise ValueError(f"doc_id={doc_id} is out of range!")

    elapsed_time = format_time(time.time() - start_time)
    print("HiEve Preprocessing took {:}".format(elapsed_time))
    print(f'HiEve training instance num: {len(all_train_set)}, '
          f'valid instance num: {len(all_valid_set)}, '
          f'test instance num: {len(all_test_set)}')

    if args.debug:
        print("debug mode on")
        all_train_set = all_train_set[0:100]
        all_valid_set = all_train_set
        all_test_set = all_train_set

    return all_train_set, all_valid_set, all_test_set


def matres_data_loader(args: Dict[str, Any], data_dir: Union[Path, str], device: torch.device) -> Tuple[List[Any]]:
    all_tml_dir_path_dict, all_tml_file_dict, all_txt_file_path = get_matres_files(data_dir)
    eiid_to_event_trigger, eiid_pair_to_rel_id = read_matres_files(all_txt_file_path)

    all_train_set, all_valid_set, all_test_set = [], [], []
    start_time = time.time()
    for i, fname in enumerate(tqdm(eiid_pair_to_rel_id.keys())):
        file_name = fname + ".tml"
        dir_path = get_tml_dir_path(file_name, all_tml_dir_path_dict, all_tml_file_dict)
        data_dict = matres_file_reader(dir_path, file_name, eiid_to_event_trigger, eiid_pair_to_rel_id)

        eiid_to_event_trigger_dict = eiid_to_event_trigger[fname]
        eiid_pair_to_rel_id_dict = eiid_pair_to_rel_id[fname]
        if file_name in all_tml_file_dict["tb"]:
            train_set = get_matres_train_set(data_dict, eiid_to_event_trigger_dict, eiid_pair_to_rel_id_dict, device)
            all_train_set.extend(train_set)
        elif file_name in all_tml_file_dict["aq"]:
            valid_set = get_matres_valid_test_set(data_dict, eiid_pair_to_rel_id_dict, device)
            all_valid_set.extend(valid_set)
        elif file_name in all_tml_file_dict["pl"]:
            test_set = get_matres_valid_test_set(data_dict, eiid_pair_to_rel_id_dict, device)
            all_test_set.extend(test_set)
        else:
            raise ValueError(f"file_name={file_name} does not exist in MATRES dataset!")

    elapsed_time = format_time(time.time() - start_time)
    print("MATRES Preprocessing took {:}".format(elapsed_time))
    print(f'MATRES training instance num: {len(all_train_set)}, '
          f'valid instance num: {len(all_valid_set)}, '
          f'test instance num: {len(all_test_set)}')
    if args.debug:
        print("debug mode on")
        all_train_set = all_train_set[0:100]
        all_valid_set = all_train_set
        all_test_set = all_train_set

    return all_train_set, all_valid_set, all_test_set


def get_dataloaders(log_batch_size: int, train_set: List, valid_set_dict: Dict[str, List], test_set_dict: Dict[str, List]) -> Tuple[DataLoader]:

    train_dataloader = DataLoader(EventDataset(train_set), batch_size=2 ** log_batch_size, shuffle=True)
    valid_dataloader_dict, test_dataloader_dict = {}, {}

    for data_type, valid_set in valid_set_dict.items():
        if data_type == "hieve":
            valid_dataloader = DataLoader(EventDataset(valid_set), batch_size=2 ** log_batch_size, shuffle=True)
        elif data_type == "matres":
            valid_dataloader = DataLoader(EventDataset(valid_set), batch_size=2 ** log_batch_size, shuffle=True)
        else:
            raise ValueError(f"dataset={data_type} is not supported at this time!")
        valid_dataloader_dict[data_type] = valid_dataloader

    for data_type, test_set in test_set_dict.items():
        if data_type == "hieve":
            test_dataloader = DataLoader(EventDataset(test_set), batch_size=2 ** log_batch_size, shuffle=True)
        elif data_type == "matres":
            test_dataloader = DataLoader(EventDataset(test_set), batch_size=2 ** log_batch_size, shuffle=True)
        else:
            raise ValueError(f"dataset={data_type} is not supported at this time!")
        test_dataloader_dict[data_type] = test_dataloader

    return train_dataloader, valid_dataloader_dict, test_dataloader_dict