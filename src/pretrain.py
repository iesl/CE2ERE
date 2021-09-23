import argparse
import os
import random

import pandas as pd
import spacy
import tqdm
import math

from os import listdir
from os.path import isfile, join
from pathlib import Path

from natsort import natsorted
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import LineByLineTextDataset

nlp = spacy.load("en_core_web_sm")
random.seed(10)

def create_txt_file(data_dir, txt_path):
    hieve_dir = data_dir / "hievents_v2/processed/"
    hieve_files = natsorted([f for f in listdir(hieve_dir) if isfile(join(hieve_dir, f)) and f[-4:] == "tsvx"])
    # write lines to .txt file
    print(f"Writing file contents into {os.path.abspath(txt_path)}...")
    with open(txt_path, "w") as output:
        for file in tqdm.tqdm(hieve_files):
            # print(file)
            file_path = hieve_dir / file
            for line in open(file_path, mode="r"):
                line = line.split("\t")
                type = line[0].lower()
                if type == "text":
                    text = line[1]
                    tokens = nlp(text)
                    for sent in tokens.sents:
                        output.write(f"{sent.string.strip()}\n")
    print("done!")


def get_num_lines(txt_path):
    num_lines = len(list(open(txt_path)))
    print("# of lines:", num_lines)
    return num_lines


def get_train_test_split(data_dir, txt_path):
    with open(txt_path, "r") as input:
        lines = input.readlines()
        lines = [line.rstrip() for line in lines]

    train, test = train_test_split(lines, test_size=0.2)
    train_file = data_dir / "article_train_split.txt"
    test_file = data_dir / "article_test_split.txt"
    with open(train_file, "w") as f:
        for line in train:
            f.write(f"{line.strip()}\n")

    with open(test_file, "w") as f:
        for line in test:
            f.write(f"{line.strip()}\n")

    return {
        "train": os.path.abspath(train_file),
        "test": os.path.abspath(test_file),
    }


def setup_tokenizer(file_path_dict):
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    # dataset = LineByLineTextDataset(
    #     tokenizer=tokenizer,
    #     file_path=file_path_dict["train"],
    #     block_size=512,
    # )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("pretrain arg parser")
    parser.add_argument("--data_dir", help="data directory", type=str)
    parser.add_argument("--create_file", default=False, action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser()

    roberta_model_save_dir = "./roberta_retrained/"
    Path(roberta_model_save_dir).mkdir(parents=True, exist_ok=True)
    model_save_path = roberta_model_save_dir

    txt_path = data_dir / "all_article_text.txt"

    if args.create_file:
        create_txt_file(data_dir, txt_path)
    get_num_lines(txt_path)
    file_path_dict = get_train_test_split(data_dir, txt_path)

    # setup_tokenizer(file_path_dict)
