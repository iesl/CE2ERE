import ast
import json
import torch
import random

from tqdm import tqdm
from pathlib import Path


def set_seed():
    torch.manual_seed(42)
    random.seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


def main():
    set_seed()
    data_dir = Path("../data").expanduser()
    with open(data_dir / "hievents_v2/hieve_file_list.txt") as f:
        hieve_files = ast.literal_eval(f.read())

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

    train_files = []
    valid_files = []
    test_files = []
    for i, file in enumerate(tqdm(hieve_files)):
        doc_id = i
        if doc_id in train_range:
            train_files.append(file)
        elif doc_id in valid_range:
            valid_files.append(file)
        elif doc_id in test_range:
            test_files.append(file)

    print("train:", train_files)
    print("valid:", valid_files)
    print("test:", test_files)


if __name__ == '__main__':
    main()