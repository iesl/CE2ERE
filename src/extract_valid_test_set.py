import ast
import json
import torch
import random

from tqdm import tqdm
from pathlib import Path

from utils import get_hieve_files


def set_seed():
    torch.manual_seed(10)
    random.seed(10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(10)


def main():
    set_seed()
    data_dir = Path("../data").expanduser()
    hieve_dir, hieve_files = get_hieve_files(data_dir)

    train_range = range(0, 60)
    valid_range = range(60, 80)
    test_range = range(80, 100)

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